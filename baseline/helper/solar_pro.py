from openai import OpenAI
from tqdm import tqdm
from typing import List
import numpy as np

class SolarPro:
    def __init__(
        self,
        client_type = "openai",
        model="solar-pro",
        base_url="https://api.upstage.ai/v1/solar",
        api_key=None,
    ):
        self.client=OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        print(f"Initiate OpenAI client... model = {model}")
        self.client_type = client_type
        self.model = model
    
    def get_respond(self, prompt: str, temperature: float, max_tokens=float('inf')):
        if self.client_type == "openai":
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
            return response.choices[0].message.content
        else:
            raise ValueError("client not implemented")

        