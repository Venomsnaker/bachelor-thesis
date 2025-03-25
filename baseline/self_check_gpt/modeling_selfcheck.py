import spacy
import bert_score
import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, List, Set, Tuple, Union
from transformers import logging
logging.set_verbosity_error()

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer
from utils import expand_list1, expand_list2, NLIConfig, LLMPromptConfig
from modeling_ngram import UnigramModel, NgramModel
from modeling_selfcheck_apiprompt import SelfCheckAPIPrompt

class SelfCheckBERTScore:
    def __init__(self, default_model="en", rescale_with_baseline=True):
        self.nlp = spacy.load("en_core_web_sm")
        self.default_model = default_model
        self.rescale_with_baseline = rescale_with_baseline
        print("SelfCheck-BERTScore initializedd")

    @torch.no_grad()
    def predict(
        self,
        sentences: List[str],
        sampled_passages: List[str],
    ):
        num_sentences = len(sentences)
        num_samples = len(sampled_passages)
        bertscore_array = np.zeros((num_sentences, num_samples))

        for s in range(num_samples):
            sample_passage = sampled_passages[s]
            sentences_sample = [sent for sent in self.nlp(sample_passage).sents]
            sentences_sample = [sent.text.strip() for sent in sentences_sample if len(sent) > 3]
            num_sentences_sample = len(sentences_sample)
            refs = expand_list1(sentences, num_sentences_sample)
            cands = expand_list2(sentences_sample, num_sentences)

            P, R, F1 = bert_score.score(
                cands, refs,
                lang=self.default_model,
                verbose=False,
                rescale_with_baseline=self.rescale_with_baseline
            )
            F1_arr = F1.reshape(num_sentences, num_sentences_sample)
            F1_arr_max_axis1 = F1_arr.max(axis=1).values
            F1_arr_max_axis1 = F1_arr_max_axis1.numpy()
            bertscore_array[:,s] = F1_arr_max_axis1
        bertscore_mean_per_sent = bertscore_array.mean(axis=-1)
        one_minus_bertscore_mean_per_sent = 1.0 - bertscore_mean_per_sent
        return one_minus_bertscore_mean_per_sent

class SelfCheckNgram:
    def __init__(self, n: int, lowercase: bool = True):
        self.n = n
        self.lowercase = lowercase
        print(f"SelfCheck-{n}gram initialized")

    def predict(
            self, 
            sentences: List[str],
            passage: str,
            sampled_passages: List[str],
    ):
        if self.n == 1:
            ngram_model = UnigramModel(lowercase=self.lowercase)
        elif self.n > 1:
            ngram_model = NgramModel(n=self.n, lowercase=self.lowercase)
        else:
            raise ValueError("n must be integer >= 1")
        ngram_model.add(passage)

        for sampled_passage in sampled_passages:
            ngram_model.add(sampled_passage)
        ngram_model.train(k=0)
        ngram_pred = ngram_model.evaluate(sentences)
        return ngram_pred
    
class SelfCheckNLI:
    def __init__(
        self,
        nli_model: str = None,
        device = None
    ):
        nli_model = nli_model if nli_model is not None else NLIConfig.nli_model
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(nli_model)
        self.model = DebertaV2ForSequenceClassification.from_pretrained(nli_model)
        self.model.eval()
        if device is None:
            device = torch.device("cpu")
        self.model.to(device)
        self.device = device
        print("SelfCheck-NLI initialized to device", device)

    @torch.no_grad()
    def predict(
        self,
        sentences: List[str],
        sampled_passages: List[str],
    ):
        num_sentences = len(sentences)
        num_samples = len(sampled_passages)
        scores = np.zeros((num_sentences, num_samples))

        for sent_i, sentence in enumerate(sentences):
            for sample_i, sample in enumerate(sampled_passages):
                inputs = self.tokenizer.batch_encode_plus(
                    batch_text_or_text_pairs=[(sentence, sample)],
                    add_special_tokens=True, padding="longest",
                    truncation=True, return_tensors="pt",
                    return_token_type_ids=True, return_attention_mask=True,
                )
                inputs = inputs.to(self.device)
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)
                prob_ = probs[0][1].item()
                scores[sent_i, sample_i] = prob_
        scores_per_sentence = scores.mean(axis=-1)
        return scores_per_sentence
      
class SelfCheckLLMPrompt:
    def __init__(
        self,
        model: str = None,
        device = None
    ):
        model = model if model is not None else LLMPromptConfig.model
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto")
        self.model.eval()
        if device is None:
            device = torch.device("cpu")
        self.model.to(device)
        self.device = device
        self.prompt_template = "Context: {context}\n\nSentence: {sentence}\n\nIs the sentence supported by the context above? Answer Yes or No.\n\nAnswer: "
        self.text_mapping = {'yes': 0.0, 'no': 1.0, 'n/a': 0.5}
        self.not_defined_text = set()
        print(f"SelfCheck-LLMPrompt ({model}) initialized to device {device}")

    def set_prompt_template(self, prompt_template: str):
        self.prompt_template = prompt_template

    @torch.no_grad()
    def predict(
        self,
        sentences: List[str],
        sampled_passages: List[str],
        verbose: bool = False,
    ):
        num_sentences = len(sentences)
        num_samples = len(sampled_passages)
        scores = np.zeros((num_sentences, num_samples))
        disable = not verbose

        for sent_i in tqdm(range(num_sentences), disable=disable):
            sentence = sentences[sent_i]

            for sample_i, sample in enumerate(sampled_passages):
                sample = sample.replace("\n", " ")
                prompt = self.prompt_template.format(context=sample, sentence=sentence)
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                generate_ids = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=5,
                    do_sample=False
                )
                ouput_text = self.tokenizer.batch_decode(
                    generate_ids, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]
                generate_text = ouput_text.replace(prompt, "")
                score_ = self.text_postprocessing(generate_text)
                scores[sent_i, sample_i] = score_
        scores_per_sentence = scores.mean(aixs=-1)
        return scores_per_sentence

    def text_postprocessing(
        self,
        text,
    ):
        text = text.lower().strip()

        if text[:3] == "yes":
            text = 'yes'
        elif text[:2] == "no":
            text = "no"
        else:
            if text not in self.not_defined_text:
                print(f"Warning: {text} not defined")
                self.not_defined_text.add(text)
            text = 'n/a'
        return self.text_mapping[text]