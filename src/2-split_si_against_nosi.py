# Date: 2024/4/15
import os
import shutil
from os.path import join
from pathlib import Path
from pycocotools.coco import COCO


def split_si_against_nosi(
    images_dir, coco_json_path, output_si_dir, output_nosi_dir, split_ratio
):
    """
    Split images to two kinds: have seals/inscriptions or don't have (si VS nosi).
    `split_ratio` is the train:val ratio of the nosi data.
    """
    ## Load annotations json
    coco = COCO(coco_json_path)
    # Get filenames of two kinds images
    images_id_path_dic = {
        v["id"]: Path(images_dir) / Path(v["file_name"]).name
        for v in coco.imgs.values()
    }
    # si path dic
    images_id_which_has_si = set([v["image_id"] for v in coco.anns.values()])
    images_path_si = [images_id_path_dic[i] for i in images_id_which_has_si]
    # nosi path dic
    images_id_which_has_no_si = set(images_id_path_dic.keys()) - images_id_which_has_si
    images_path_nosi = [images_id_path_dic[i] for i in images_id_which_has_no_si]
    # Make dirtories to save them
    os.makedirs(output_si_dir)
    os.makedirs(output_nosi_dir)
    os.makedirs(join(output_nosi_dir, "train"))
    os.makedirs(join(output_nosi_dir, "val"))
    # Do copy & paste
    _ = [shutil.copy(i, Path(output_si_dir) / i.name) for i in images_path_si]
    train_num = int(len(images_path_nosi) * split_ratio)
    for i in ["train", "val"]:
        paths = (
            images_path_nosi[:train_num]
            if i == "train"
            else images_path_nosi[train_num:]
        )
        for j in paths:
            shutil.copy(j, Path(output_nosi_dir) / i / j.name)


if __name__ == "__main__":
    images_dir = "../data/Chinese-Painting-n240-s800-labeled/images"
    coco_json_path = "../data/Chinese-Painting-n240-s800-labeled/result.json"
    output_si_dir = "../data/Chinese-Painting-s800-si"
    output_nosi_dir = "../data/Chinese-Painting-s800-nosi"
    split_si_against_nosi(
        images_dir, coco_json_path, output_si_dir, output_nosi_dir, split_ratio=0.7
    )
