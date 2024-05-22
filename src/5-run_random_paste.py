import os
import random
from os.path import join
from utils.io import read_image, write_png
from algorithms.random_paste import PaintingObjMulti

SEAL_MAX_NUM = 8
INS_MAX_NUM = 4

def run_random_paste(img_path, obj_dir, img_pasted_path, mask_path, bboxes_path):
    img = read_image(img_path)

    obj_names = os.listdir(obj_dir)
    seal_names = [i for i in obj_names if i.split('_')[1] == "seals"]
    ins_names = [i for i in obj_names if i.split('_')[1] == "inscriptions"]
    seal_names_sel = random.sample(seal_names, random.randint(1, SEAL_MAX_NUM))
    ins_names_sel = random.sample(ins_names, random.randint(1, INS_MAX_NUM))
    obj_names_sel = seal_names_sel + ins_names_sel
    obj_paths = [os.path.join(obj_dir, i) for i in obj_names_sel]
    objs = [read_image(i) for i in obj_paths]

    painting = PaintingObjMulti(img, objs, by_conflict_ratio=0.2)
    painting.random_paste()

    write_png(img_pasted_path, painting.img_pasted)
    write_png(mask_path, painting.mask)
    with open(bboxes_path, "w") as f:
        f.write("\n".join([" ".join([str(j) for j in i]) for i in painting.bbox_multi]))
    return None


def run_random_paste_multi(img_dir, obj_dir, img_pasted_dir, mask_dir, bboxes_dir):
    _ = [os.makedirs(i) for i in [img_pasted_dir, mask_dir, bboxes_dir]]
    img_names = os.listdir(img_dir)
    img_paths = [os.path.join(img_dir, i) for i in img_names]
    for img_name, img_path in zip(img_names, img_paths):
        name = img_name.split('.')[0]
        img_pasted_path = os.path.join(img_pasted_dir, f"{name}.png")
        mask_path = os.path.join(mask_dir, f"{name}_mask.png")
        bboxes_path = os.path.join(bboxes_dir, f"{name}_bboxes.txt")
        run_random_paste(img_path, obj_dir, img_pasted_path, mask_path, bboxes_path)


if __name__ == "__main__":
    data_root = "../data"
    output_dir = "Chinese-Painting-s800-pasted"
    img_dir = join(data_root, "Chinese-Painting-s800-nosi")
    obj_dir = join(data_root, "Seal-Inscription-boxes-filtered-n389-manual")
    img_pasted_dir = join(data_root, output_dir, "imgs_pasted")
    mask_dir = join(data_root, output_dir, "masks")
    bboxes_dir = join(data_root, output_dir, "bboxes")
    run_random_paste_multi(img_dir, obj_dir, img_pasted_dir, mask_dir, bboxes_dir)
