import cv2
from utils import load_image
from region_classification import classifier
from region_proposal import detector
from config import pipeline
from utils import get_min_max_area, plot
from metric import get_iou

class vlm_detection():
    def __init__(self, img_path, config):
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
                bbox = detector(self.img_path, self.target_size).get_all_bbox_sam(self.sam_checkpoint, self.sam_model_type, self.padding, self.min_area, self.max_area)
            else: 
                bbox = detector(self.img_path, self.target_size).get_all_bbox_selective_search(self.max_box, self.min_area, self.max_area)

            patches = self.get_image_patches(self.img_path, bbox)
            shape_pred =[]

            for patch in patches:
                img, _, _ = load_image(patch, target_size=self.target_size)
                if img is not None:
                    pred = classifier(img, self.example_image_dir, self.example_label_dir, self.shot, self.target_size).classify(self.vlm_model_type, self.vlm)
                    shape_pred.append(pred)

            bbox = self.map_shape_to_bbox(shape_pred, bbox)

            if self.plot:
                plot(self.img_path, bbox, self.save_filename, detection_plot=True)

            return bbox
    
if __name__ == "__main__":
    img_path = "/work/mech-ai-scratch/shreyang/AFM/SAM_VLM_API/dataset/sampled/images/sample11.png"
    label_file_path = "/work/mech-ai-scratch/shreyang/AFM/SAM_VLM_API/dataset/sampled/labels/sample11_gt.txt"

    vlm_detection = vlm_detection(img_path, pipeline)
    bbox = vlm_detection.process()

    iou = get_iou(bbox, label_file_path, img_path)
    print(iou)