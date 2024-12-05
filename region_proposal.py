import cv2
from utils import load_image

import selective_search
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

class detector():
    def __init__(self, img_path, target_size=(124, 124)):
        self.img = cv2.imread(img_path)
        self.bbox = None
        self.target_size = target_size

    def get_all_bbox_sam(self, sam_checkpoint, sam_model_type, padding, min_area=None, max_area=None):
        sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        sam.to("cuda")
        mask_generator = SamAutomaticMaskGenerator(sam)
        
        masks = mask_generator.generate( self.img)

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