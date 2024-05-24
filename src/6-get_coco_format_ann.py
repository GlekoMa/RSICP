import json
import os
import numpy as np
from os.path import join
from PIL import Image
from utils.io import read_image


def binary_mask_to_rle(binary_mask):
    rle = {"counts": [], "size": list(binary_mask.shape)}
    flattened_mask = binary_mask.ravel(order="F")
    diff_arr = np.diff(flattened_mask)
    nonzero_indices = np.where(diff_arr != 0)[0] + 1
    lengths = np.diff(np.concatenate(([0], nonzero_indices, [len(flattened_mask)])))
    # note that the odd counts are always the numbers of zeros
    if flattened_mask[0] == 1:
        lengths = np.concatenate(([0], lengths))
    rle["counts"] = lengths.tolist()
    return rle


def bboxes_txt_to_dic(path):
    """return like {'ann': [0, 1, 0], 'bbox': [ [193, 113, 282, 228], [...], [...] ]}"""
    with open(path) as f:
        lst = [[int(j) for j in i.split(' ')] for i in f.read().split('\n')]
    dic = {'ann': [i[0] for i in lst],
           'bbox': [i[1:] for i in lst]}
    return dic


def create_coco_json(image_dir, mask_multi_root, bboxes_dir, output_json_path):
    categories = [{"id": 0, "name": "Inscription"}, {"id": 1, "name": "Seal"}]
    coco_dict = {
        "info": {
            "description": "Dataset",
            "url": "",
            "version": "1.0",
            "year": 2024,
            "contributor": "",
            "date_created": "2024-05-22",
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": categories,
    }

    image_id = 1
    annotation_id = 1

    for img_filename in os.listdir(image_dir):
        # Load image and get its dimensions
        img_path = join(image_dir, img_filename)
        img = Image.open(img_path)
        width, height = img.size
        # Add image info
        image_info = {
            "id": image_id,
            "file_name": img_filename,
            "width": width,
            "height": height,
        }
        coco_dict["images"].append(image_info)
        # Load corresponding masks
        img_filename_id = img_filename.split('.')[0]
        mask_multi_dir = join(mask_multi_root, f"{img_filename_id}-mask-multi")
        mask_paths = [join(mask_multi_dir, i) for i in os.listdir(mask_multi_dir)]
        masks = [read_image(i) for i in mask_paths]
        # Load corresponding bboxes
        bboxes_path = join(bboxes_dir, f"{img_filename_id}_bboxes.txt")
        bboxes_dic = bboxes_txt_to_dic(bboxes_path)
        # Iterate over bboxes and masks to create annotations
        for ann, bbox, mask in zip(bboxes_dic['ann'], bboxes_dic['bbox'], masks):
            x_min, y_min, x_max, y_max = bbox
            x, y, w, h = x_min, y_min, x_max - x_min, y_max - y_min
            binary_mask = (mask[:, :, 0] > 0).astype(np.uint8)
            area = float(binary_mask.sum())
            rle = binary_mask_to_rle(binary_mask)
            annotation_info = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": ann,
                "bbox": [x, y, w, h],
                "area": area,
                "segmentation": rle,
                "iscrowd": 0,
            }
            coco_dict["annotations"].append(annotation_info)
            annotation_id += 1
        image_id += 1
    # Save to JSON file
    with open(output_json_path, "w") as json_file:
        json.dump(coco_dict, json_file)


if __name__ == "__main__":
    for i in ["train", "val"]:
        data_root = join("../data/Chinese-Painting-s800-pasted", i)
        image_dir = join(data_root, "imgs_pasted")
        mask_multi_root = join(data_root, "masks_multi")
        bboxes_dir = join(data_root, "bboxes")
        output_json_path = join(data_root, f"json_annotation_{i}.json")
        create_coco_json(image_dir, mask_multi_root, bboxes_dir, output_json_path)
