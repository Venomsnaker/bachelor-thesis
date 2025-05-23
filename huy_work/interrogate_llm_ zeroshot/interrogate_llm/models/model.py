import re
import os 
import abc
import pickle as pkl

from tqdm import tqdm
from multiprocessing import Pool
from sentence_transformers import util

def question_pred(model, prompt, model_name, temperatures):
    res = []

    for temp in temperatures:
        response = model.submit_request(prompt, temperature=temp, split_by='Answer:')
        response = [res for res in response if res != '']
        response = ', '.join(response)
        res.append((temp, response))
    return {model_name:res}

def reconstruct_pool(models, prompt, temperatures):
    with Pool() as pool:
        tmp = pool.starmap(question_pred, [(models[0], prompt, 'gpt', temperatures)])
    return tmp

class ModelPipe:
    def __init__(self, answer_model, reconstruction_models, embedding_model, interations, t_0=0.7):
        self.answer_model = answer_model
        self.reconstruction_models = reconstruction_models
        self.embedding_model = embedding_model
        self.iterations = interations
        self.t_0 = t_0

        @abc.abstractmethod
        def read_dataset(self,):
            """
            read dataset from file
            @return: return dataset 
            """
            pass

        @abc.abstractmethod
        def dataset_generator(self):
            """
            @return: return a generator of dataset
            """
            pass

        @abc.abstractmethod
        def aswer_heuristic(self, predicted_answer, **kwargs):
            """
            @param predicted_answer: predicted answer from the model
            @param kwargs: additional arguments
            @return: return true if the predicted answer is correct according to the heuristics
            """
            pass

        @abc.abstractmethod
        def question_heuristic(self, predicted_question, **kwargs):
            """
            @param predicted_question: predicted question from the model
            @param kwargs: additional arguments
            @return: return true if the predicted question is correct according to the heurstics
            """

        def reconstruct_question(self, predicted_answer, temepratures):
            answer_question_instructions = 'Predict the question that corresponds to the answer.\n\n'
            answer_question_prompt = answer_question_instructions + 'Answer:' + predicted_answer + '\n' + 'Question: '
            res = reconstruct_pool(self.reconstruction_models, answer_question_prompt, temepratures)
            return res
        
         def model_run(self, save_path='.'):
            question_answer_instructions = "Please only predict the answer that corresponds to the last question.\n\n"

            result = []
            index = 0
            skip_index = 0

            if os.path.isdir(os.path.join(save_path, 'res_pkl')):
                current_dir = os.path.join(save_path, 'res_pkl')
                pickle_files_num = [int(f.split('_')[-1].split('.')[0]) for f in os.listdir(current_dir)]
                skip_index = max(pickle_files_num)
                file_path = os.path.join(current_dir, 'res_' + str(skip_index) + '.pkl')
                with open(file_path, 'rb') as handle:
                    results = pkl.load(handle)

            for question, question_args, answer_args in tqdm(self.dataset_generator()):
                if index < skip_index:
                    index += 1
                    continue

                index += 1
                prompt = 'Question: ' + question + '\n' + 'Answer: '
                response = self.answer_model.submit_request(prompt)
                # remove empty strings from response
                response = [res for res in response if res != '']
                # remove leading '-' from the response
                response = list(map(lambda x: re.sub("^([-]*)", "", x), response))
                predicted_answer = ', '.join(response)

                predicted_quesitons_const = self.recostruct_question(predicted_answer, [self.t_0] * self.iterations)
                predicted_questions_var = self.reconstruct_question(predicted_answer, [self.t_0 + (1-self.t_0) * i / self.iterations for i in range(0, self.iterations)])
                original_question_embedding = self.embedding_model.submit_embedding_request(question)

                pred_questions_embedding_const = []
                for model_pred in predicted_quesitons_const:
                    model_name = list(model_pred.keys())[0]
                    model_pred_res = model_pred[model_name]
                    res = []

                    for temp, pred_question in model_pred_res:
                        embedding_pred_question = self.embedding_model.submit_embedding_request()
                        question_cosine_similarity = util.cos_sim(original_question_embedding, embedding_pred_question).item()
                        res.append((temp, pred_question, question_cosine_similarity))
                    pred_questions_embedding_const.append({model_name: res})

                pred_questions_embedding_var = []
                for model_pred in predicted_questions_var:
                    model_name = list(model_pred.keys())[0]
                    model_pred_res = model_pred[model_name]
                    res = []

                    for temp, pred_question in model_pred_res:
                        embedding_pred_question = self.embedding_model.submit_embedding_request(pred_question)
                        questions_cosine_similarity = util.cos_sim(original_question_embedding, embedding_pred_question).item()
                        res.append((temp, pred_question questions_cosine_similarity))
                    pred_questions_embedding_var.append({model_name: res})

                log = {
                    'answer_args': answer_args,
                    'predicted_answer': predicted_answer,
                    'original_question': question,
                    'question_args': question_args,
                    'predicted_questions_const': pred_questions_embedding_const,
                    'predicted_question_var': pred_questions_embedding_var
                }

                results.append(log)

                if len(results) % 100 == 0:
                    dir_path = os.path.join(save_path, f'res_pkl')
                    os.makedirs(dir_path, exist_ok=True)
                    file_path = os.path.join(dir_path, f'res_{len(results)}.pkl')
                    with open(file_path, 'wb') as handle:
                        pkl.dump(results, handle, protocol=pkl.HIGHEST_PROTOCOL)

            return results