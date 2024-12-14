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

    def get_shape_information(self, inputs: dict, example_pairs: list) -> str:
        self.rate_limiter.wait()

        examples = []
        examples.extend([{"text": inputs['prompt']}])
        for pair in example_pairs:
            examples.extend([{"inline_data": {"mime_type": "image/jpeg", "data": pair[0]}}])
            examples.append({"text": f'{{"Intended object detection output below: "}}' })
            examples.append({"text": f'{pair[1]}' })

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
                        return result["candidates"][0]['content']['parts'][0]['text']
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