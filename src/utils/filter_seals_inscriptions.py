# Date: 2024/4/15
import sys
import os
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
import matplotlib.pyplot as plt
from color_filter import savefig, filter_red_by_lab, filter_black_by_hsv


def auto_filter_si(img: NDArray[np.float32], obj: str) -> NDArray[np.float32]:
    """
    Auto processes filter function based on the object's kind.
    """
    if obj == "seal":
        img_filtered = filter_red_by_lab(img)
    elif obj == "inscription":
        img_filtered = filter_black_by_hsv(img)
    try:
        return img_filtered
    except Exception:
        print("Argument 'obj' must be 'seal' or 'inscription'!")
        sys.exit(1)


def auto_filter_si_save(img: NDArray[np.float32], obj: str, output_path: str) -> None:
    """
    Auto processes filter function based on the object's kind and save the output image.
    """
    img_filtered = auto_filter_si(img, obj)
    savefig(img_filtered, output_path)


def auto_filter_si_save_based_dir(input_dir, output_dir):
    """
    Loads all the images from the input directory and auto processes filter
    function based on the object's type. Then saves these newly generated images
    to the output directory.
    """
    imgs_path = [Path(input_dir) / i for i in os.listdir(input_dir)]
    for img_path in imgs_path:
        output_path = Path(output_dir) / (img_path.name.split(".")[0] + "_filtered.png")
        img = plt.imread(img_path)
        obj = img_path.name.split("_")[1][:-1]
        auto_filter_si_save(img, obj, str(output_path))


if __name__ == "__main__":
    root = Path("../../data/temp")
    input_dir = str(root / "seal_inscription_boxes")
    output_dir = str(root / "seal_inscription_boxes_filtered")
    os.makedirs(output_dir)
    auto_filter_si_save_based_dir(input_dir, output_dir)
