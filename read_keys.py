import os
import json
    
def read_keys():
    with open('key.json') as file:
        key = json.load(file)
        openai_key = key["openai"]
        gemini_key = key["gemini"]

    os.environ['OPENAI_API_KEY'] = openai_key
    os.environ['GOOGLE_API_KEY'] = gemini_key