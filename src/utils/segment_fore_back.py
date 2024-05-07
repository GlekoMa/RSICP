# Date: 2024/4/16
# Note: Need run 'split_si_against_nosi.py' first
import os
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
import random
from skimage import color, filters, morphology
import matplotlib.pyplot as plt


def invert_color_by_assumption(binary_img: NDArray[bool]) -> NDArray[bool]:
    """
    The binary_img contains True and False values, where True typically
    represents white and False represents black.

    We have the following assumption: among two colors, the one that appears
    more frequently will be the background, and the one that appears less
    frequently will be the foreground. And we want white to represent the
    background, while black represents the foreground.

    To achieve this, when True is more common than False (when white is more
    common than black), we need to invert their colors.
    """
    x, y = binary_img.shape
    true_num = binary_img.sum()
    false_num = x * y - true_num
    if false_num > true_num:
        binary_img_inverted = ~np.zeros_like(binary_img)
        binary_img_inverted[binary_img] = False
    else:
        binary_img_inverted = binary_img
    return binary_img_inverted


def segment_fore_back(img: NDArray[np.float32], return_process_imgs=False):
    """
    Segments a colorful image to distinguish foreground (represented by False)
    and background (represented by True).
    ---
    img:
        Original image.
    return_process_imgs:
        If False, return closed image. If True, return [origin, gray, smooth,
        binary, closed] images list.
    """
    gray_img = color.rgb2gray(img)
    smooth_img = filters.gaussian(gray_img, sigma=1)
    threshold_value = filters.threshold_otsu(smooth_img)
    binary_img = invert_color_by_assumption(smooth_img > threshold_value)
    closed_img = morphology.closing(binary_img, morphology.square(3))
    if not return_process_imgs:
        return closed_img
    else:
        return [img, gray_img, smooth_img, binary_img, closed_img]


if __name__ == "__main__":
    img_nosi_root = "../../data/temp/Chinese-Painting-s800-n240-nosi"
    imgs = [Path(img_nosi_root) / i for i in os.listdir(img_nosi_root)]
    random.shuffle(imgs)
    
    fig, axes = plt.subplots(5, 5, constrained_layout=True, figsize=(10, 10))
    for i in range(5):
        img = plt.imread(imgs[i])
        for j, v in enumerate(segment_fore_back(img, return_process_imgs=True)):
            axes[i, j].imshow(v, cmap=plt.cm.gray)
            axes[i, j].axis("off")
    plt.show()
