import os
from pathlib import Path
from algorithms.filter import auto_filter_seal_ins
from utils.io import read_image, write_png


def filter_boxes(boxes_dir, output_dir):
    """
    Loads all the images from the input boxes directory and auto processes filter
    function based on the object's type. Then saves these newly generated images to the
    output directory.
    """
    os.makedirs(output_dir)
    img_paths = [Path(boxes_dir) / i for i in os.listdir(boxes_dir)]
    for img_path in img_paths:
        output_path = Path(output_dir) / (img_path.name.split(".")[0] + "_filtered.png")
        img = read_image(img_path)
        obj = img_path.name.split("_")[1][:-1]
        img_filtered = auto_filter_seal_ins(img, obj)
        write_png(output_path, img_filtered)


if __name__ == "__main__":
    boxes_dir = "../data/Seal-Inscription-boxes"
    output_dir = "../data/Seal-Inscription-boxes-filtered"
    filter_boxes(boxes_dir, output_dir)
