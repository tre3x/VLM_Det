import time
import requests
import ast

class RateLimiter:
    def __init__(self, max_requests, time_window):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = 0

    def wait(self):
        if self.requests >= self.max_requests:
            time.sleep(self.time_window)
            self.requests = 0
        self.requests += 1

class GeminiAPI:
    def __init__(self, api_key, model, retries=10, delay=5):
        self.api_key = api_key
        self.model = model
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        self.headers = {
            "Content-Type": "application/json",
        }
        self.rate_limiter = RateLimiter(max_requests=15, time_window=5)  # Adjust as needed
        self.retries = retries
        self.delay = delay

    def get_shape_information(self, inputs: dict, example_pairs) -> str:
        self.rate_limiter.wait()

        examples = []
        
        examples.extend([{"text": inputs['prompt']}])
        for shape, img_crop_list in example_pairs.items():
            for crop in img_crop_list:
                examples.append([{"inline_data": {"mime_type": "image/jpeg", "data":crop}}])
                examples.append({"text": f'{{"Intended classification output below: "}}' })
                examples.append({"text": f'{shape}' })
        
        examples.extend([
            {"text": inputs['prompt']},
            {
                "inline_data": {
                    "mime_type": "image/jpeg", 
                    "data": inputs['image']
                }
            }
        ])

        payload = {
            "contents": [
                {
                    "parts": examples
                }
            ],
            "generationConfig": {
                "temperature": 1.0,
                "maxOutputTokens": 4096,
                "response_mime_type": "application/json",
            }
        }

        # Retry mechanism
        attempt = 0
        while attempt < self.retries:
            try:
                response = requests.post(f"{self.url}?key={self.api_key}", headers=self.headers, json=payload)
                if response.status_code == 200:
                    result = response.json()
                    if "candidates" in result and result["candidates"]:
                        text = ast.literal_eval(result["candidates"][0]['content']['parts'][0]['text'])
                        if "Classification" in text.keys():
                            return text["Classification"]
                        if "classification" in text.keys():
                            return text["classification"]
                        return 'None'
                    
                    else:
                        raise Exception(f"Unexpected API response format: {result}")
                else:
                    raise Exception(f"Failed to call the API: {response.status_code} - {response.text}")

            except Exception as e:
                print(f"API call failed on attempt {attempt+1}/{self.retries}: {str(e)}")
                if attempt + 1 == self.retries:
                    print(f"Skipping this image after {self.retries} failed attempts.")
                    return None  # Skip the current image
                else:
                    time.sleep(self.delay)  # Delay before retrying
                    attempt += 1