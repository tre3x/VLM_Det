import os
import cv2
import random
import json
from utils import load_image, map_shape_to_bbox
from region_proposal import detector
from api import GPTAPI, GeminiAPI
from config import prompts


class classifier():
    def __init__(self, patch, example_image_dir, example_label_dir, shot=0, target_size = (124, 124)):
        self.shot = shot
        self.patch = patch
        self.prompt = prompts.CELL_CLASSIFICATION
        self.example_image_dir = example_image_dir
        self.example_label_dir = example_label_dir
        self.target_size = target_size
        self.read_keys()

    def read_keys(self):
        with open('key.json') as file:
            key = json.load(file)
            openai_key = key["openai"]
            gemini_key = key["gemini"]

        os.environ['OPENAI_API_KEY'] = openai_key
        os.environ['GOOGLE_API_KEY'] = gemini_key

    def generate_examples(self, shots, example_image_dir, example_label_dir, offset=0):
        img_bbox = {}
        examples = {
                "Round": [],
                "Spindle": [],
                "Polygonal": []
            }
        num_each = int(shots/3)

        sampled_image_names = random.sample(os.listdir(example_image_dir), 20)

        for sampled_image_name in sampled_image_names:
            img_name = sampled_image_name.split('.')[0]
            label_file = os.path.join(example_label_dir, "{}.txt".format(img_name))
            image_file = os.path.join(example_image_dir, sampled_image_name)
            img = cv2.imread(image_file)
            _, img_width, img_height = load_image(img, target_size=self.target_size)

            shapes_dict = {
                "Round": [],
                "Spindle": [],
                "Polygonal": []
            }

            with open(label_file, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    
                    shape_type = int(parts[0])
                    x_center, y_center, box_width, box_height = map(float, parts[1:])
                    x_center_abs = int(x_center * img_width)
                    y_center_abs = int(y_center * img_height)
                    width_abs = int(box_width * img_width)
                    height_abs = int(box_height * img_height)

                    x1 = int(x_center_abs - (width_abs/2))
                    y1 = int(y_center_abs - (height_abs/2))
                    x2 = int(x_center_abs + (width_abs/2))
                    y2 = int(y_center_abs + (height_abs/2))
                    
                    if shape_type == 0:
                        shape_name = "Round"
                    elif shape_type == 1:
                        shape_name = "Spindle"
                    elif shape_type == 2:
                        shape_name = "Polygonal"
                    else:
                        continue

                    shapes_dict[shape_name].extend([(x1, y1, x2, y2)])
            
            img_bbox[sampled_image_name] = shapes_dict

        shapes = ["Round", "Spindle", "Polygonal"]
        for shape in shapes:
            for image_name, image_bbox in img_bbox.items():
                image_file = os.path.join(example_image_dir, image_name)
                img = cv2.imread(image_file)

                if len(image_bbox[shape]) >= num_each-len(examples[shape]): num_sample = num_each-len(examples[shape])
                else: num_sample = len(image_bbox[shape])
                
                for bbox in random.sample(image_bbox[shape], num_sample):
                    h, w = img.shape[:2]
                    x1, y1, x2, y2 = max(bbox[0]-offset, 0), max(bbox[1]-offset, 0), min(bbox[2]+offset, w), min(bbox[3]+offset, h)
                    crop = img[y1:y2, x1:x2]
                    crop, _, _ = load_image(crop, target_size=self.target_size)
                    examples[shape].append(crop)


                if len(examples[shape])>=num_each:
                    break

        return examples

    def classify(self, vlm_model_type="gpt", vlm="gpt-4o-2024-08-06"):
        if vlm_model_type=="gpt":
            api_key = os.environ["OPENAI_API_KEY"]
            api = GPTAPI(api_key, vlm)
            
        if vlm_model_type=="gemini":
            api_key = os.environ["GOOGLE_API_KEY"]
            api = GeminiAPI(api_key, vlm)

            examples = self.generate_examples(shots=self.shot, example_image_dir = self.example_image_dir, 
                                              example_label_dir = self.example_label_dir, offset=0)

            inputs = {"prompt": self.prompt, "image": self.patch}
            pred_shape = api.get_shape_information(inputs, examples)

        return pred_shape