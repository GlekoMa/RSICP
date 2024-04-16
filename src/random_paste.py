# Date: 2024/4/17
from segment_fore_back import segment_fore_back
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import random


def random_obj_idx(img, obj):
    img_h, img_w, _ = img.shape
    # calculate object width & height
    obj_h, obj_w, _ = obj.shape
    obj_h_half, obj_w_half = (int(i / 2) for i in (obj_h, obj_w))
    # randomly generate object center index based image shape
    obj_center_x = random.randint(0 + obj_w_half + 1, img_w - obj_w_half - 2)
    obj_center_y = random.randint(0 + obj_h_half + 1, img_h - obj_h_half - 2)
    # get object's location of the image
    obj_idx_x = list(range(obj_center_x - obj_w_half, obj_center_x + obj_w_half + 1))
    obj_idx_y = list(range(obj_center_y - obj_h_half, obj_center_y + obj_h_half + 1))
    return np.ix_(obj_idx_y, obj_idx_x)


def random_paste(img, obj, return_process_imgs=False):
    img_binary = segment_fore_back(img)
    img_temp = img_binary.astype(np.float32)
    img_rgb_binary = np.stack([img_temp, img_temp, img_temp], axis=-1)
    # judge whether the object box has conflict with the image's foreground
    # if has conflict, regenerate.
    obj_rectangle_idx = random_obj_idx(img_rgb_binary, obj)
    while (~img_binary[obj_rectangle_idx]).sum() != 0:
        obj_rectangle_idx = random_obj_idx(img_rgb_binary, obj)
    # TODO: get real colorful pixels index of objects
    if not return_process_imgs:
        img[obj_rectangle_idx] = obj
        return img
    else:
        img_rgb_binary_obj = np.copy(img_rgb_binary)
        img_rgb_binary_obj[obj_rectangle_idx] = obj
        img_obj = np.copy(img)
        img_obj[obj_rectangle_idx] = obj
        return [img, img_rgb_binary, img_rgb_binary_obj, img_obj]


if __name__ == "__main__":
    img = plt.imread("../assets/image_nosi.png")
    obj = plt.imread("../assets/seal.png")
    process_imgs = random_paste(img, obj, return_process_imgs=True)
    fig, axes = plt.subplots(1, 4, constrained_layout=True, figsize=(12, 3))
    for i, ax in enumerate(axes):
        ax.imshow(process_imgs[i])
        ax.axis("off")
    plt.show()
    
