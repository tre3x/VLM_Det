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

class GPTAPI:
    def __init__(self, api_key, model, retries=10, delay=5):
        self.api_key = api_key
        self.model = model
        self.url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        self.rate_limiter = RateLimiter(max_requests=20, time_window=1)
        self.retries = retries
        self.delay = delay

    def get_shape_information(self, inputs: dict, example_pairs: list) -> str:
        self.rate_limiter.wait()

        examples = []
        for pair in example_pairs:
            examples.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{pair[0]}", "detail": "high"}})
            examples.append({"type": "text", "text": f'{{"Intended object detection output below: "}}' })
            examples.append({"type": "text", "text": f'{pair[1]}' })

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": inputs['prompt']},
                        *examples,
                        {"type": "text", "text": inputs['prompt']},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{inputs['image']}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ]
        }

        # Retry mechanism
        attempt = 0
        while attempt < self.retries:
            try:
                response = requests.post(self.url, headers=self.headers, json=payload)
                if response.status_code == 200:
                    result = response.json()
                    if "choices" in result and result["choices"]:
                        return result["choices"][0]['message']['content']
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