class prompts:
    CELL_CLASSIFICATION = """
    You will be provided with an image crop of a microscopic cell from a NIH-3T3 microscopy image. The cropped image dimensions are 124x124 pixels, and it should contain only one cell.

    **Your task is to classify the cell shape into one of the following categories based on its appearance:**

    1. **Round**: Cells appear circular, and sometimes with smooth edges.
    2. **Spindle**: Cells are elongated and tapered, resembling a spindle or stretched ellipse.
    3. **Polygonal**: Cells have multiple angles or sides.
    4. **None**: Any other shape, multiple cells, no visible cell, or ambiguous cases.

    **Guidelines:**
    - If the cell matches the **Round** description, classify it as `Round`.
    - If the cell matches the **Spindle** description, classify it as `Spindle`.
    - If the cell matches the **Polygonal** description, classify it as `Polygonal`.
    - For any other cases, classify it as `None`.

    **Response Format:**
    - Return only the exact label (`Round`, `Spindle`, `Polygonal`, or `None`) as a single string.
    - Do not include any additional text, quotes, or formatting.
    """
    
    DETECTION_CLASSIFICATION = """
    You will be provided with an NIH-3T3 microscopy image. The image dimension is 124x124 pixels.
    Your task is to detect and return bounding boxes for three different cell shapes: **round**, **spindle**, and **polygonal**.

    **Cell Shape Descriptions**:
    - **Round**: Circular or near-circular cells.
    - **Spindle**: Elongated, tapering cells resembling a spindle shape.
    - **Polygonal**: Irregular, multi-sided cells with distinct edges.

    Each bounding box should be represented by coordinates in the format `(x1, y1, x2, y2)`:
    - `(x1, y1)` is the **top-left** corner of the bounding box.
    - `(x2, y2)` is the **bottom-right** corner of the bounding box.

    ### Expected JSON Format:
    {
        "Round": [(x1, y1, x2, y2), (x3, y3, x4, y4), ...],
        "Spindle": [(x1, y1, x2, y2), (x3, y3, x4, y4), ...],
        "Polygonal": [(x1, y1, x2, y2), (x3, y3, x4, y4), ...]
    }

    **Instructions**:
    1. Ensure the JSON output is properly formatted for Python parsing.
    2. Only return the JSON object; do not include any additional text, comments, or preambles.
    """
    
    SINGLE_CLASS_DETECTION_CLASSIFICATION = """
    You will be provided with an NIH-3T3 microscopy image. The image dimensions are **124x124 pixels**.

    Your task is to detect and return bounding boxes for **polygonal-shaped cells**.

    **Cell Shape Description**:
    - **Polygonal**: Irregular, multi-sided cells with distinct edges.

    Each bounding box should be represented by coordinates in the format `(x1, y1, x2, y2)`:
    - `(x1, y1)` is the **top-left** corner of the bounding box.
    - `(x2, y2)` is the **bottom-right** corner of the bounding box.

    ### Expected JSON Format:
    {
        "Round": [(x1, y1, x2, y2), (x3, y3, x4, y4), ...],
        "Spindle": [(x1, y1, x2, y2), (x3, y3, x4, y4), ...],
        "Polygonal": [(x1, y1, x2, y2), (x3, y3, x4, y4), ...]
    }

    **Instructions**:
    1. Ensure the JSON output is properly formatted for Python parsing.
    2. Only return the JSON object; do not include any additional text, comments, or preambles.
    """



class pipeline:
    filter_area = True
    area_label_dir = "./dataset/labels/validation/"
    area_image_dir = "./dataset/images/validation"
    max_box = 80
    detector = "sam"
    sam_checkpoint = "/work/mech-ai-scratch/shreyang/AFM/VLMs/sam-model/sam_vit_h_4b8939.pth"
    sam_model_type = "vit_h"
    padding = 12
    shot = 0
    vlm_model_type = "gemini"
    vlm = "gemini-1.5-pro"
    target_size = (124, 124)
    plot = True
    save_filename = "./results/test.png"
    example_image_dir = "./dataset/images/train"
    example_label_dir = "./dataset/labels/train"