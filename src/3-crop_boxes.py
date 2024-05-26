# Date: 2024/4/15
import os
from pathlib import Path
from utils.io import read_image, write_png
from utils.bbox_mask import get_boxes_dic


def crop_boxes(img_dir, coco_json_path, output_dir, ann="both"):
    """
    Crop image's seals/inscriptions boxes and save them to new images.

    One input image could generate several seal/inscription images, these new images
    will be named as `{original_image_name}_seals(or incriptions)_1.png`. `ann`
    indicates the kind of objects to be cropped. Can be 'seal', 'inscription', or
    'both'.
    """
    os.makedirs(output_dir)
    boxes_dic = get_boxes_dic(coco_json_path)
    for k, v in boxes_dic.items():
        file = str(Path(img_dir) / k)
        if ann == "both":
            objs = ["seals", "inscriptions"]
        else:
            objs = [ann + "s"]
        for obj in objs:
            img = read_image(file)
            for i in range(len(v[obj])):
                x, y, width, height = [int(i) for i in v[obj][i]]
                box = img[y : y + height, x : x + width, ...]
                filename = f"{Path(file).name.split('.')[0]}_{obj}_{i+1}.png"
                write_png(str(Path(output_dir) / filename), box)


if __name__ == "__main__":
    data_root = Path("../data")
    coco_json_root = data_root / "Chinese-Painting-n240-s800-labeled"
    coco_json_path = coco_json_root / "result.json"
    img_dir = data_root / "Chinese-Painting-s800-si"
    output_dir = data_root / "Seal-Inscription-boxes"
    crop_boxes(img_dir, coco_json_path, output_dir)
