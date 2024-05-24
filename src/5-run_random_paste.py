import os
import random
from os.path import join
from utils.io import read_image, write_png
from algorithms.random_paste import PaintingObjMulti

SEAL_MAX_NUM = 8
INS_MAX_NUM = 4

def run_random_paste(img_path, obj_dir, img_pasted_path, mask_path, mask_multi_dir, bboxes_path):
    img = read_image(img_path)

    obj_names = os.listdir(obj_dir)
    seal_names = [i for i in obj_names if i.split('_')[1] == "seals"]
    ins_names = [i for i in obj_names if i.split('_')[1] == "inscriptions"]
    seal_names_sel = random.sample(seal_names, random.randint(1, SEAL_MAX_NUM))
    ins_names_sel = random.sample(ins_names, random.randint(1, INS_MAX_NUM))
    obj_names_sel = seal_names_sel + ins_names_sel
    obj_paths = [os.path.join(obj_dir, i) for i in obj_names_sel]
    objs = [read_image(i) for i in obj_paths]
    anns = [i.split('_')[1][:-1] for i in obj_paths]

    painting = PaintingObjMulti(img, objs, anns, by_conflict_ratio=0.2)
    painting.random_paste()

    write_png(img_pasted_path, painting.img_pasted)
    write_png(mask_path, painting.mask)
    os.makedirs(mask_multi_dir)
    mask_multi_anns = ["inscription" if i[0] == 0 else "seal" for i in painting.bbox_multi]
    mask_id = 1
    for ann, mask in zip(mask_multi_anns, painting.mask_multi):
        write_png(os.path.join(mask_multi_dir, f"{mask_id}_{ann}.png"), mask)
        mask_id += 1
    with open(bboxes_path, "w") as f:
        f.write("\n".join([" ".join([str(j) for j in i]) for i in painting.bbox_multi]))
    return None


def run_random_paste_multi(img_dir, obj_dir, img_pasted_dir, mask_dir, mask_multi_root, bboxes_dir):
    _ = [os.makedirs(i) for i in [img_pasted_dir, mask_dir, mask_multi_root, bboxes_dir]]
    img_names = os.listdir(img_dir)
    img_paths = [os.path.join(img_dir, i) for i in img_names]
    for img_name, img_path in zip(img_names, img_paths):
        name = img_name.split('.')[0]
        img_pasted_path = os.path.join(img_pasted_dir, f"{name}.png")
        mask_path = os.path.join(mask_dir, f"{name}_mask.png")
        mask_multi_dir = os.path.join(mask_multi_root, f"{name}-mask-multi")
        bboxes_path = os.path.join(bboxes_dir, f"{name}_bboxes.txt")
        run_random_paste(img_path, obj_dir, img_pasted_path, mask_path, mask_multi_dir, bboxes_path)


if __name__ == "__main__":
    data_root = "../data"
    output_root = "Chinese-Painting-s800-pasted"

    for i in ["train", "val"]:
        os.makedirs(join(output_root, i))
        img_dir = join(data_root, "Chinese-Painting-s800-nosi", i)
        obj_dir = join(data_root, "Seal-Inscription-boxes-filtered-manual")
        img_pasted_dir = join(data_root, output_root, i, "imgs_pasted")
        mask_dir = join(data_root, output_root, i, "masks")
        mask_multi_root = join(data_root, output_root, i, "masks_multi")
        bboxes_dir = join(data_root, output_root, i, "bboxes")
        run_random_paste_multi(img_dir, obj_dir, img_pasted_dir, mask_dir, mask_multi_root, bboxes_dir)
