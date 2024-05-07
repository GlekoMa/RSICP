import os
from os.path import basename
import random
from pathlib import Path
import matplotlib.pyplot as plt
from random_paste import random_paste_multi
from color_filter import savefig


def generate_img_pasted_mask(img_path, obj_dir, img_pasted_path, mask_path):
    img = plt.imread(img_path)
    obj_names = os.listdir(obj_dir)
    obj_paths = [os.path.join(obj_dir, i) for i in obj_names]
    obj_to_pasted_paths = random.sample(obj_paths, random.randint(1, 6))
    objs_to_pasted = [plt.imread(i) for i in obj_to_pasted_paths]
    img_pasted, mask = random_paste_multi(img, objs_to_pasted)
    if mask.sum() == 0:
        print("\033[93m[WARN] no loc to paste obj, generate empty mask\033[0m")
        print(f"\033[93m[WARN]     img_name: {basename(img_path).split('.')[0]}\033[0m")
        print(f"\033[93m[WARN]     img_id: {basename(mask_path).split('_')[0]}\033[0m")
        return None
    else:
        savefig(img_pasted, img_pasted_path)
        savefig(mask[:, :, None] / 255, mask_path)
        return None


def generate_img_pasted_mask_multi(img_dir, obj_dir, img_pasted_dir, mask_dir):
    img_names = os.listdir(img_dir)
    img_paths = [os.path.join(img_dir, i) for i in img_names]
    for i in range(len(img_paths)):
        img_path = img_paths[i]
        img_pasted_path = os.path.join(img_pasted_dir, f"{i+1}.png")
        mask_path = os.path.join(mask_dir, f"{i+1}_mask.png")
        generate_img_pasted_mask(img_path, obj_dir, img_pasted_path, mask_path)


if __name__ == "__main__":
    data_root = Path("../../data")
    obj_dir = str(data_root / "seal-inscription-boxes-filtered-manual")
    img_dir = str(data_root / "Chinese-Painting-s800-n240-nosi-manual")
    img_pasted_dir = str(data_root / "imgs_pasted")
    mask_dir = str(data_root / "masks")
    generate_img_pasted_mask_multi(img_dir, obj_dir, img_pasted_dir, mask_dir)


