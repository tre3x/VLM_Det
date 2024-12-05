import cv2

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

def get_iou(bbox, label_file_path, img_path):
    # Read label data
    label_bbox = {"Round": [], "Polygonal": [], "Spindle": []}
        
    with open(label_file_path, 'r') as file:
        for line in file:
            values = line.strip().split()
            shape_class = int(values[0])
            x_center, y_center, width, height = map(float, values[1:])

            img_height, img_width, _ = cv2.imread(img_path).shape
            x_center_abs, y_center_abs = int(x_center * img_width), int(y_center * img_height)
            width_abs, height_abs = int(width * img_width), int(height * img_height)
            x1, y1 = int(x_center_abs - width_abs / 2), int(y_center_abs - height_abs / 2)
            x2, y2 = int(x_center_abs + width_abs / 2), int(y_center_abs + height_abs / 2)

            coords = [x1, y1, x2, y2]
            shape_mapping = {0: "Round", 1: "Spindle", 2: "Polygonal"}
            if shape_class in shape_mapping:
                label_bbox[shape_mapping[shape_class]].append(coords)

    # Compare IoU with self.bbox_processed
    pred_bbox = bbox
    iou_scores = []

    for category in label_bbox:
        boxes1 = label_bbox[category]
        boxes2 = pred_bbox[category]

        if boxes1 and boxes2:
            for box1 in boxes1:
                max_iou = 0.0
                for box2 in boxes2:
                    iou = compute_iou(box1, box2)
                    if iou > max_iou:
                        max_iou = iou
                iou_scores.append(max_iou)

    # Calculate average IoU
    global_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0.0
    return global_iou