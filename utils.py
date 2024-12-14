import os
import io
import cv2
import base64
import random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def load_image(image_array, target_size=None):
    try:
        img = Image.fromarray(image_array.astype('uint8'))  # Convert array to PIL image
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if target_size is not None:
            img = img.resize(target_size, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=95)
        width, height = img.size
            
        return base64.b64encode(buffer.getvalue()).decode('utf-8'), width, height
    except Exception as e:
        return None, None, None
    

def map_shape_to_bbox(res, box):
    shapes = {"Spindle": [], "Polygonal": [], "Round": []}
    for r, b in zip(res, box):
        if r == "Spindle":
            shapes["Spindle"].append(b)
        if r == "Polygonal":
            shapes["Polygonal"].append(b)
        if r == "Round":
            shapes["Round"].append(b)
            
    return shapes


def plot(img_path, bboxes, target_size=None, save_filename=None, detection_plot=True):
    image = cv2.imread(img_path)
    if target_size is not None:
        image = cv2.resize(image, target_size)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)


    if not detection_plot: 
        for x1, y1, x2, y2 in bboxes:
            bbox = mpatches.Rectangle(
                (x1, y1), (x2-x1), (y2-y1), fill=False, edgecolor='red', linewidth=1)
            ax.add_patch(bbox)


    else:
        for x1, y1, x2, y2 in bboxes["Round"]:
            bbox = mpatches.Rectangle(
                (x1, y1), (x2-x1), (y2-y1), fill=False, edgecolor='red', linewidth=1)
            ax.add_patch(bbox)
        for x1, y1, x2, y2 in bboxes["Polygonal"]:
            bbox = mpatches.Rectangle(
                (x1, y1), (x2-x1), (y2-y1), fill=False, edgecolor='green', linewidth=1)
            ax.add_patch(bbox)  
        for x1, y1, x2, y2 in bboxes["Spindle"]:
            bbox = mpatches.Rectangle(
                (x1, y1), (x2-x1), (y2-y1), fill=False, edgecolor='blue', linewidth=1)
            ax.add_patch(bbox)   
                

    plt.axis('off')
    if save_filename is not None: plt.savefig(save_filename)
    plt.show()

def compute_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    return intersection_area / union_area

def generate_classification_examples(shots, example_image_dir, example_label_dir, target_size, offset=0):
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
        _, img_width, img_height = load_image(img, target_size=target_size)

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
                crop, _, _ = load_image(crop, target_size=target_size)
                examples[shape].append(crop)


            if len(examples[shape])>=num_each:
                break

    return examples


def generate_detection_examples(example_image_dir, example_label_dir, shots, target_size):
        
        sampled_image_names = random.sample(os.listdir(example_image_dir), shots)
        examples = []

        for file in sampled_image_names:
            img_name = file.split('.')[0]
            label_file = os.path.join(example_label_dir, "{}.txt".format(img_name))
            image_file = os.path.join(example_image_dir, file)
            img = cv2.imread(image_file)
            # read the image and text file
            img, img_width, img_height = load_image(img, target_size=target_size)
            
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

                    x1 = int(x_center_abs - width_abs / 2)
                    y1 = int(y_center_abs - height_abs / 2)
                    x2 = int(x_center_abs + width_abs / 2)
                    y2 = int(y_center_abs + height_abs / 2)
                    
                    if shape_type == 0:
                        shape_name = "Round"
                    elif shape_type == 1:
                        shape_name = "Spindle"
                    elif shape_type == 2:
                        shape_name = "Polygonal"
                    else:
                        continue

                    shapes_dict[shape_name].extend([(x1, y1, x2, y2)])
            
            # form an example dictionary/list with these dat
            examples.append((img, shapes_dict))
            
            # return the examples
        return examples


def get_min_max_area(area_label_dir, area_image_dir):
    min_area = get_label_stat(area_label_dir, area_image_dir, filter=10).run()["min"]
    max_area = get_label_stat(area_label_dir, area_image_dir, filter=10).run()["max"]
    return min_area, max_area


class get_label_stat():
    def __init__(self, label_dir, image_dir, filter=None):
        self.label_dir = label_dir
        self.image_dir = image_dir
        self.filter = filter

    def get_label_path_with_img_dim(self):
        res = []
        if self.filter is not None:
            labels = [filename for filename in os.listdir(self.label_dir) if filename.endswith('{}.txt'.format(self.filter))]
        else:
            labels = os.listdir(self.label_dir)
        for filename in labels:
            label_path = os.path.join(self.label_dir, filename)
            image_path = os.path.join(self.image_dir, filename.split('.')[0]+'.png')

            img = cv2.imread(image_path)
            shape = (img.shape[0], img.shape[1])

            res.append((label_path, shape))

        return res
    
    def get_bbox_area(self, path, dim):
        coords_area = []
        with open(path, 'r') as file:
            for line in file:
                values = line.strip().split()

                x_center = float(values[1])
                y_center = float(values[2])
                width = float(values[3])
                height = float(values[4])

                img_height, img_width = dim[0], dim[1]

                x_center_abs = int(x_center * img_width)
                y_center_abs = int(y_center * img_height)
                width_abs = int(width * img_width)
                height_abs = int(height * img_height)

                x1 = int(x_center_abs - width_abs / 2)
                y1 = int(y_center_abs - height_abs / 2)
                x2 = int(x_center_abs + width_abs / 2)
                y2 = int(y_center_abs + height_abs / 2)
                    
                coords_area.append((x2-x1)*(y2-y1))

        return coords_area
    
    def get_stats(self, bbox_area):
        min_area = min(bbox_area)
        max_area = max(bbox_area)
        average_area = sum(bbox_area)/len(bbox_area)

        return {
            "min": min_area,
            "max": max_area,
            "mean": average_area
        }

    def run(self):
        data = self.get_label_path_with_img_dim()

        bbox_area = []
        for label_path, dim in data:
            bbox_area.extend(self.get_bbox_area(label_path, dim))

        stats = self.get_stats(bbox_area)

        return stats
    
