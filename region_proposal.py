import os
import cv2
import config
import ast

from utils import load_image, plot, get_min_max_area, generate_detection_examples
from metric import detector_iou
from config import prompts
from read_keys import read_keys

import selective_search
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from api.detection import GeminiAPI



class detector():
    def __init__(self, img_path):
        self.img = cv2.imread(img_path)
        self.bbox = None

    def get_all_bbox_sam(self, sam_checkpoint, sam_model_type, padding, min_area=None, max_area=None):
        sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        sam.to("cuda")
        mask_generator = SamAutomaticMaskGenerator(sam)
        
        masks = mask_generator.generate(self.img)

        bounding_boxes = []
        for mask in masks:
            x1, y1, x2, y2 = mask["bbox"][0], mask["bbox"][1], mask["bbox"][0]+mask["bbox"][2], mask["bbox"][1]+mask["bbox"][3]
            x1, y1, x2, y2 = max(0, x1-padding), max(0, y1-padding), min( self.img.shape[0], x2+padding), min( self.img.shape[1], y2+padding)
            box_area = (x2-x1)*(y2-y1)
            if min_area and min_area:
                if (box_area > min_area) and (box_area < max_area):
                    bounding_boxes.append([x1, y1, x2, y2])
            else:
                bounding_boxes.append([x1, y1, x2, y2])
            
        return bounding_boxes
    
    def get_all_bbox_selective_search(self, max_box, min_area=None, max_area=None):
        boxes = selective_search.selective_search(self.img, mode='single', random_sort=False)
        box_filter = []

        if min_area and min_area:
            for box in boxes:
                box_area = (box[2]-box[0]) * (box[3]-box[1])
                if (box_area > min_area) and (box_area < max_area):
                    box_filter.append([box[0], box[1], box[2], box[3]]) 
        
        if max_box:
            box_filter = selective_search.box_filter(boxes, topN=max_box)

        return box_filter
    
    def get_all_bbox_api(self, model_type="gpt", vlm="gpt-4o-2024-08-06", example_image_dir=None, example_label_dir=None, target_size=(124, 124), shots=0):

        read_keys()
        
        if model_type=="gpt":
            api_key = os.environ["OPENAI_API_KEY"]
            api = GPTAPI(api_key, vlm)
        if model_type=="gemini":
            api_key = os.environ["GOOGLE_API_KEY"]
            api = GeminiAPI(api_key, vlm)
            
        #prompt = prompts.DETECTION_CLASSIFICATION
        prompt = prompts.SINGLE_CLASS_DETECTION_CLASSIFICATION
        
        examples = []
        if shots>0:
            examples = generate_detection_examples(example_image_dir = example_image_dir, 
                                                example_label_dir = example_label_dir, shots=shots, target_size=target_size)
        img, _, _ = load_image(self.img, target_size=target_size)
        inputs = {"prompt": prompt, "image": img}

        preds = api.get_shape_information(inputs, examples)
        preds = ast.literal_eval(preds)

        
        return preds
        
        

if __name__=="__main__":
    img_path = f"/work/mech-ai-scratch/shreyang/AFM/SAM_VLM_API/dataset/sampled/images/sample6.png"
    model_type="gemini"
    vlm="gemini-1.5-pro"
    example_image_dir="./dataset/images/train"
    example_label_dir="./dataset/labels/train"
    
    preds = detector(img_path).get_all_bbox_api(model_type, vlm, example_image_dir, example_label_dir, target_size=(124, 124), shots=3)
    plot(img_path, preds, 
         save_filename="/work/mech-ai-scratch/shreyang/AFM/SAM_VLM_API/results/test/gemini.png", target_size=(124, 124))
    
    '''
    samples = ["sample1", "sample2", "sample3", "sample4", "sample5", "sample6", "sample7", "sample8", "sample9", "sample10"]
    for sample in samples:
        img_path = f"/work/mech-ai-scratch/shreyang/AFM/SAM_VLM_API/dataset/sampled/images/{sample}.png"
        label_path = f"/work/mech-ai-scratch/shreyang/AFM/SAM_VLM_API/dataset/sampled/labels/{sample}_gt.txt"
        min_area, max_area = get_min_max_area(config.pipeline.area_label_dir, config.pipeline.area_image_dir)

        detector_obj = detector(img_path)
        bbox = detector_obj.get_all_bbox_sam("/work/mech-ai-scratch/shreyang/AFM/VLMs/sam-model/sam_vit_h_4b8939.pth", "vit_h", 12,
                                        min_area=min_area, max_area=max_area)
        
        iou = detector_iou(bbox, label_path, img_path)
        print(f"For {sample}, IoU: {iou}")

    plot(img_path, bbox, 
         save_filename="/work/mech-ai-scratch/shreyang/AFM/SAM_VLM_API/results/test/sam_{sample}.png", detection_plot=False)
    '''
