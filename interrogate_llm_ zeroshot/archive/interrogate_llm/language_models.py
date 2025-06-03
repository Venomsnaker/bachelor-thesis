import abc
import time
import torch
import openai
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class OpenAI:
    def __init__(self):
        self.deployment_id = 'text-davinci-003'
        self.deployment_id = 'd365-sales-davinci003'

    def submit_request(self, prompt, temperature=0.7, max_tokens=1024, n=1):
        error_counter = 0

        while (True):
            try:
                response = openai.Completion.create(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    n=n,
                )
                break
            except Exception as e:
                if e.args[0] == 'string indices must be integers' or 'The response was filtered' in \
                        e.args[0]:
                    return {'choices': [{'text': ''}]}
                time.sleep(1)
                error_counter += 1

                if error_counter > 10:
                    raise e
        response = [res['text'].strip() for res in response['choices']]
        return response

class EmbeddingModel:
    @abc.abstractmethod
    def submit_embedding_request(self, text):
        pass

class GPTEmbedding(EmbeddingModel):
    def __init__(self):
        self.deployment_id = 'text-embedding-ada-002'

    def submit_embedding_request(self, text):
        error_counter = 0

        while(True):
            try:
                response = openai.Embedding.create(
                    input=text
                )
                break
            except Exception as e:
                time.sleep(1)
                error_counter += 1

                if error_counter > 10:
                    raise e
        response = response['data'][0]['embedding']
        return response

class SBert(EmbeddingModel):
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L12-v2')

    def submit_embedding_request(self, text):
        response = self.model.encode([text], convert_to_tensor=True)
        return response

class E5(EmbeddingModel):
    def __init__(self):
        self.model = SentenceTransformer('intfloat/e5-large-v2')

    def submit_embedding_request(self, text):
        response = self.model.encode([text], convert_to_tensor=True)
        return response

class BERTEmbedding(EmbeddingModel):
    def __init__(self):
        self.model = SentenceTransformer("bert-base-uncased")

    def submit_embedding_request(self, text):
        response = self.model.encode([text], convert_to_tensor=True)
        return response
