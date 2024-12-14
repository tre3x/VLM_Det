import os
import cv2
import random
import json

from config import prompts
from utils import load_image, map_shape_to_bbox, generate_classification_examples
from read_keys import read_keys

from region_proposal import detector
from api.classification import GPTAPI, GeminiAPI




class classifier():
    def __init__(self, example_image_dir, example_label_dir, shot=0, target_size = (124, 124)):
        self.shot = shot
        self.prompt = prompts.CELL_CLASSIFICATION
        self.example_image_dir = example_image_dir
        self.example_label_dir = example_label_dir
        self.target_size = target_size
        read_keys()

    def classify(self, patch, padding=0, vlm_model_type="gpt", vlm="gpt-4o-2024-08-06"):
        if vlm_model_type=="gpt":
            api_key = os.environ["OPENAI_API_KEY"]
            api = GPTAPI(api_key, vlm)
            
        if vlm_model_type=="gemini":
            api_key = os.environ["GOOGLE_API_KEY"]
            api = GeminiAPI(api_key, vlm)

            examples = generate_classification_examples(shots=self.shot, example_image_dir = self.example_image_dir, 
                                              example_label_dir = self.example_label_dir, target_size=self.target_size, offset=padding)

            inputs = {"prompt": self.prompt, "image": patch}
            pred_shape = api.get_shape_information(inputs, examples)

        return pred_shape