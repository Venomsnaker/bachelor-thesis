from openai import OpenAI
from groq import Groq
from tqdm import tqdm
from typing import Dict, List, Set, Tuple, Union
import numpy as np
import os
import time
from dotenv import load_dotenv

API_CALL_SLEEP = 1

class SelfCheckAPIPrompt:
    def __init__(
        self,
        client_type = "openai",
        model = "solar-pro", # "gpt-3.5-turbo"
        base_url="https://api.upstage.ai/v1/solar",
        api_key = None,
    ):
        if client_type == "openai":
            self.client=OpenAI(
                base_url=base_url,
                api_key=api_key,
            )
            print("Initiate OpenAI client... model = {}".format(model)) 
        elif client_type == "groq":
            self.client = Groq(api_key=api_key)
            print("Initiate Groq client ... model = {}".format(model))
            
        self.client_type = client_type
        self.model = model
        self.prompt_template = "Context: {context}\n\nSentence: {sentence}\n\nIs the sentence supported by the context above? Answer Yes or No.\n\nAnswer: "
        self.text_mapping = {'yes': 0.0, 'no': 1.0, 'n/a': 0.5}
        self.not_defined_text = set()

    def set_prompt_template(self, prompt_template: str):
        self.prompt_template = prompt_template
        
    def get_respond(self, prompt: str, temperature=0, max_tokens=float('inf')):
        if self.client_type == "openai" or self.client_type == 'groq':
            params = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "temperature": temperature
            }
            
            if max_tokens != float('inf'):
                params["max_tokens"] = max_tokens
            
            response = self.client.chat.completions.create(**params)
            return response.choices[0].message.content # return response['choices'][0]['message']['content']
        else:
            raise ValueError("client not implemented")
        
    def predict(
        self, 
        sentences: List[str],
        sample_passages: List[str],
        verbose: bool = False,
    ):
        num_sentences = len(sentences)
        num_samples = len(sample_passages)
        scores = np.zeros((num_sentences, num_samples))
        disable = not verbose

        for sent_i in tqdm(range(num_sentences), disable=disable):
            sentence = sentences[sent_i]

            for sample_passage_idx, sample_passage in enumerate(sample_passages):
                # Set up prompt
                prompt = self.prompt_template.format(context=sample_passage.replace("\n", " ") , sentence=sentence)
                generated_text = self.get_respond(prompt)
                score = self.text_postprocessing(generated_text)
                scores[sent_i, sample_passage_idx] = score
                
            # Manage API call
            if num_sentences > 10:
                time.sleep(API_CALL_SLEEP)
        scores_per_sentence = scores.mean(axis=-1)
        return scores_per_sentence

    def text_postprocessing(
            self,
            text,
    ):
        text = text.lower().strip()
        
        if text[:3] == 'yes':
            text = 'yes'
        elif text[:2] == 'no':
            text = 'no'
        else:
            if text not in self.not_defined_text:
                print(f"warning: {text} not defined")
                self.not_defined_text.add(text)
            text = 'n/a'
        return self.text_mapping[text]
