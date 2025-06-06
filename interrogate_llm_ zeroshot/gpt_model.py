from openai import OpenAI
import time

class OpenAIClient:
    def __init__(
            self,
            api_key,
            model = 'gpt-4.1-mini-2025-04-14',
            retries = 3):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.retries = retries

    def generate_response(self, prompt: str, temeprature = 1, n = 5):
        retries = 3

        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {'role': 'system', 'content': 'You are a helpful assistant.'},
                        {'role': 'user', 'content': prompt}
                    ],
                    temperature=temeprature,
                    n=n 
                )
                return [choice.message.content for choice in response.choices]
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(1)
                else:
                    raise e

    
    def set_temperature(self, temperature: float):
        if 0 <= temperature <= 2:
            self.temperature = temperature
        else:
            raise ValueError("Temperature must be between 0 and 2.")

class OpenAIEmbeddingClient:
    def __init__(
            self, 
            api_key,
            model = 'text-embedding-3-large',):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        
    def get_embedding(self, input):
        retries = 3

        for attempt in range(retries):
            try: 
                embedding = self.client.embeddings.create(
                    model=self.model,
                    input=input)
                return embedding.data[0].embedding
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(1)
                else:
                    raise e
