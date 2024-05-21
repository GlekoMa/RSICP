import os
from os.path import basename
import random
from pathlib import Path
import matplotlib.pyplot as plt
from random_paste import random_paste_multi
from color_filter import savefig


def generate_img_random_pasted(img_path, obj_dir, img_pasted_path, mask_path, boxes_path):
    img = plt.imread(img_path)
    obj_names = os.listdir(obj_dir)
    obj_paths = [os.path.join(obj_dir, i) for i in obj_names]
    obj_to_pasted_paths = random.sample(obj_paths, random.randint(1, 6))
    objs_to_pasted = [plt.imread(i) for i in obj_to_pasted_paths]
    img_pasted, mask, boxes = random_paste_multi(img, objs_to_pasted)
    if mask.sum() == 0:
        print("\033[93m[WARN] no loc to paste obj, generate empty mask\033[0m")
        print(f"\033[93m[WARN]     img_name: {basename(img_path).split('.')[0]}\033[0m")
        print(f"\033[93m[WARN]     img_id: {basename(mask_path).split('_')[0]}\033[0m")
        return None
    else:
        savefig(img_pasted, img_pasted_path)
        savefig(mask[:, :, None] / 255, mask_path)
        with open(boxes_path, 'w') as f:
            f.write('\n'.join([' '.join([str(j) for j in i]) for i in boxes]))
        return None



def generate_img_random_pasted_multi(img_dir, obj_dir, img_pasted_dir, mask_dir, boxes_dir):
    img_names = os.listdir(img_dir)
    img_paths = [os.path.join(img_dir, i) for i in img_names]
    for i in range(len(img_paths)):
        img_path = img_paths[i]
        img_pasted_path = os.path.join(img_pasted_dir, f"{i+1}.png")
        mask_path = os.path.join(mask_dir, f"{i+1}_mask.png")
        boxes_path = os.path.join(boxes_dir, f"{i+1}_boxes.txt")
        generate_img_random_pasted(img_path, obj_dir, img_pasted_path, mask_path, boxes_path)


if __name__ == "__main__":
    data_root = Path("../../data")
    obj_dir = str(data_root / "Seal-Inscription-n389-manual")
    img_dir = str(data_root / "Chinese-Painting-s800-n81-nosi-manual-white")
    output_dir = "Chinese-Painting-s800-n81-pasted-mask"
    img_pasted_dir = str(data_root / output_dir / "imgs_pasted")
    mask_dir = str(data_root / output_dir / "masks")
    boxes_dir = str(data_root / output_dir / "boxes")
    generate_img_random_pasted_multi(img_dir, obj_dir, img_pasted_dir, mask_dir, boxes_dir)


