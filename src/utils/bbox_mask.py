import os
from os.path import join
from pathlib import Path
from pycocotools.coco import COCO


def get_boxes_dic(coco_json_path: str):
    """
    Get a dictionary mapping image file names to lists of bounding boxes for seals and
    inscriptions. Such as
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
    images_id_which_has_si = set([v["image_id"] for v in coco.anns.values()])

    def get_box_dic_part(image_id):
        anns_single = [v for v in coco.anns.values() if v["image_id"] == image_id]
        category_dic = {1: "seals", 0: "inscriptions"}
        return {
            v: [i["bbox"] for i in anns_single if i["category_id"] == k]
            for k, v in category_dic.items()
        }

    return {images_id_name_dic[i]: get_box_dic_part(i) for i in images_id_which_has_si}
