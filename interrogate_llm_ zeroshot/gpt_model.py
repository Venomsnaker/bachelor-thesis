from openai import OpenAI
import time

class OpenAIClient:
    def __init__(
            self,
            api_key,
            model = 'gpt-4o-2024-11-20',
            temperature = 0.7,
            retries = 3):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.retries = retries

    def generate_response(self, prompt: str):
        retries = 3
        
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {'role': 'system', 'content': 'You are a helpful assistant.'},
                        {'role': 'user', 'content': prompt}
                    ],
                    temperature=self.temperature
                )
                return response.choices[0].message.content
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

