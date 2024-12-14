import os
import cv2
import random
from utils import load_image
from region_classification import classifier
from region_proposal import detector
from config import pipeline
from utils import get_min_max_area, plot
from metric import get_iou

class vlm_detection():
    def __init__(self, img_path, config, custom_config=None):
        self.img_path = img_path
        self.config = config
        self.max_box = config.max_box
        self.detector = config.detector
        self.sam_checkpoint = config.sam_checkpoint
        self.sam_model_type = config.sam_model_type
        self.padding = config.padding
        self.shot = config.shot
        self.vlm_model_type = config.vlm_model_type
        self.vlm = config.vlm
        self.target_size = config.target_size
        self.plot = config.plot
        self.save_filename = config.save_filename
        self.example_image_dir = config.example_image_dir
        self.example_label_dir = config.example_label_dir
        self.min_area = None
        self.max_area = None
        if config.filter_area:
            self.min_area, self.max_area = get_min_max_area(config.area_label_dir, config.area_image_dir)
        if custom_config is not None:
            self.load_custom_config(custom_config)

    def load_custom_config(self, custom_config):
        self.shot = custom_config["shot"]
        self.save_filename = custom_config["save_filename"]

    def get_image_patches(self, img_path, bboxes):
        img_patches = []
        image = cv2.imread(img_path)
        for bbox in bboxes:
            img_patches.append(image[bbox[1]:bbox[3], bbox[0]:bbox[2]])

        return img_patches
        
    def map_shape_to_bbox(self, res, box):
        shapes = {"Spindle": [], "Polygonal": [], "Round": []}
        for r, b in zip(res, box):
            if r == "Spindle":
                shapes["Spindle"].append(b)
            if r == "Polygonal":
                shapes["Polygonal"].append(b)
            if r == "Round":
                shapes["Round"].append(b)
                
        return shapes

    def process(self):
            if self.detector=="sam":
                bbox = detector(self.img_path).get_all_bbox_sam(self.sam_checkpoint, self.sam_model_type, 
                                                                self.padding, self.min_area, self.max_area)
            else: 
                bbox = detector(self.img_path).get_all_bbox_selective_search(self.max_box, self.min_area, self.max_area)

            patches = self.get_image_patches(self.img_path, bbox)
            shape_pred =[]

            for patch in patches:
                img, _, _ = load_image(patch, target_size=self.target_size)
                if img is not None:
                    pred = classifier(self.example_image_dir, self.example_label_dir, self.shot, 
                                      self.target_size).classify(img, self.padding, self.vlm_model_type, self.vlm)
                    shape_pred.append(pred)

            bbox = self.map_shape_to_bbox(shape_pred, bbox)

            if self.plot:
                plot(self.img_path, bbox, self.save_filename, detection_plot=True)

            return bbox
    
if __name__ == "__main__":
    
    ########### FOR TESTING ONE IMAGE ################
    img_path = "/work/mech-ai-scratch/shreyang/AFM/SAM_VLM_API/dataset/sampled/images/sample3.png"
    label_file_path = "/work/mech-ai-scratch/shreyang/AFM/SAM_VLM_API/dataset/sampled/labels/sample3_gt.txt"

    vlm_detection = vlm_detection(img_path, pipeline)
    bbox = vlm_detection.process()

    iou = get_iou(bbox, label_file_path, img_path)
    print(iou)
    
    '''
    ###############################################################
    ######### SCRIPT TO RUN ANALYSIS ON MUTILPLE FILES ############
    ###############################################################
    image_folder_path = "dataset/images/train"
    label_folder_path = "dataset/labels/train"
    
    shots = [0, 6, 12, 18]
    zoom_id = "20"
    
    
    image_files = [f for f in os.listdir(image_folder_path) if f.endswith(".png")]
    label_files = [f for f in os.listdir(label_folder_path) if f.endswith(".txt")]

    def parse_filename(filename):
        parts = filename.rsplit("_", 1)
        if len(parts) != 2 or not parts[1].startswith(zoom_id):
            return None, None
        experiment_id = parts[0]
        zoom = parts[1]
        return experiment_id, zoom

    filtered_images = {}
    for img in image_files:
        experiment_id, zoom = parse_filename(img)
        if experiment_id and zoom:
            filtered_images[experiment_id] = os.path.join(image_folder_path, img)

    filtered_labels = {}
    for lbl in label_files:
        experiment_id, zoom = parse_filename(lbl)
        if experiment_id and zoom:
            filtered_labels[experiment_id] = os.path.join(label_folder_path, lbl)
    
    
    common_experiments = list(set(filtered_images.keys()) & set(filtered_labels.keys()))
    common_experiments = random.sample(common_experiments, 10)
    
    common_experiments = ["8-24-20_11_3T3"]
    output_dir = "./results/"
    for shot in shots:
        output_file_path = os.path.join(output_dir, f"shot_{shot}_iou_results.txt")
        
        with open(output_file_path, "w") as output_file:
            average_iou = 0
            output_file.write(f"Results for Shot {shot}:\n")
            
            for experiment_id in common_experiments:
                img_path = filtered_images[experiment_id]
                label_path = filtered_labels[experiment_id]
                save_filename = f"{output_dir}{experiment_id}_{zoom_id}_{shot}.png"
                custom_config = {
                    "shot": shot,
                    "save_filename": save_filename
                }

                print(f"Processing Experiment ID: {experiment_id}, Image: {img_path}, Label: {label_path}")
                
                vlm_detection_obj = vlm_detection(img_path, pipeline, custom_config)
                bbox = vlm_detection_obj.process()

                iou = get_iou(bbox, label_path, img_path)
                print(f"IoU of {experiment_id}: {iou}")
                
                # Write IoU to the shot-specific file
                output_file.write(f"Experiment ID: {experiment_id}, IoU: {iou:.4f}\n")
                
                average_iou += (iou / len(common_experiments))
            
            # Write the average IoU for the shot
            print(f"Average IoU for Shot {shot}: {average_iou}")
            output_file.write(f"\nAverage IoU for Shot {shot}: {average_iou:.4f}\n")
        
        print(f"IoU results for Shot {shot} saved to {output_file_path}")
    '''