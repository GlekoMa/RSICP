# TODO: just a demo, need a formal checking
import os
from pathlib import Path
from pycocotools.coco import COCO
from torchvision.io import read_image, write_png


def get_boxes_dic(coco_json_path):
    """
    Get a dictionary mapping image file names to lists of bounding boxes
    for seals and inscriptions. Such as
    {
        'file_name1': {'seals': [seal_boxes], 'inscriptions': [inscription_boxes]},
        'file_name2': {'seals': [seal_boxes], 'inscriptions': [inscription_boxes]},
        ...
    }
    """
    coco = COCO(coco_json_path)
    images_id_name_dic = {
        v["id"]: Path(v["file_name"]).name for v in coco.imgs.values()
    }
    images_id_which_has_sr = set([v["image_id"] for v in coco.anns.values()])

    def get_box_dic_part(image_id):
        anns_single = [v for v in coco.anns.values() if v["image_id"] == image_id]
        category_dic = {1: "seals", 0: "inscriptions"}
        return {
            v: [i["bbox"] for i in anns_single if i["category_id"] == k]
            for k, v in category_dic.items()
        }

    return {images_id_name_dic[i]: get_box_dic_part(i) for i in images_id_which_has_sr}


def crop_boxes(img_path, coco_json_path, output_dir, ann="both"):
    """
    Crop image's seals/inscriptions boxes and save them to new images.
    ---
    img_path:
        The path of image.
    coco_json_path:
        The path of json which include the annotation in coco format.
    output_dir:
        The directory of output images to save. One input image could generate
        several seal/inscription images, these new images will be named as
        `{original_image_name}_seals(or incriptions)_1.png`.
    ann:
        The kind of objects to be cropped. Can be 'seal', 'inscription', or 'both'.
    """
    boxes_dic = get_boxes_dic(coco_json_path)
    for k, v in boxes_dic.items():
        file = str(Path(img_path) / k)
        if ann == "both": 
            objs = ["seals", "inscriptions"]
        else:
            objs = [ann + 's']
        for obj in objs:
            img = read_image(file)
            for i in range(len(v[obj])):
                x, y, width, height = [int(i) for i in v[obj][i]]
                box = img[..., y : y + height, x : x + width]
                filename = f"{Path(file).name.split('.')[0]}_{obj}_{i+1}.png"
                write_png(box, str(Path(output_dir) / filename))


if __name__ == "__main__":
    data_root = Path("../data")
    coco_json_root = data_root / "Chinese-Painting-s800-n240"
    coco_json_path = coco_json_root / "result.json"
    img_path = data_root / "Chinese-Painting-s800-n240-sr"
    output_dir = "test"
    os.makedirs(output_dir)
    crop_boxes(img_path, coco_json_path, output_dir)

