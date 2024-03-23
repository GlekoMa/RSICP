import os
import io
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.patches import Rectangle
from pycocotools.coco import COCO


def save_boxed_img(
    img_path, coco_json_path, output_path, ann="both", only_show_anns=False
):
    """
    ann: str, 'seal', 'insciption', 'both'
    """
    img = plt.imread(img_path)
    coco = COCO(coco_json_path)
    ids = list(coco.imgs.keys())
    ann_ids = coco.getAnnIds(imgIds=ids[0])
    anns = [coco.loadAnns(ann_id)[0] for ann_id in ann_ids]
    if ann == "insciption":
        anns = [i for i in anns if coco.cats[i["category_id"]]["name"] == "Insciption"]
    elif ann == "seal":
        anns = [i for i in anns if coco.cats[i["category_id"]]["name"] == "Seal"]
    else:
        pass
    ann_rectangles = list()
    ann_pixels = list()  # for 'only_show_anns'
    for ann in anns:
        x, y, width, height = [int(i) for i in ann["bbox"]]
        ann_pixels += [
            (px, py)
            for px in range(x, x + width + 1)
            for py in range(y, y + height + 1)
        ]  # for 'only_show_anns'
        ann_rectangle = Rectangle(
            xy=[x, y], width=width, height=height, fill=False # linewidth=
        )
        ann_rectangles.append(ann_rectangle)
    if only_show_anns:  # for 'only_show_anns'
        img_not_ann_indexes = [
            (x, y) for x in range(img.shape[0]) for y in range(img.shape[1])
        ]
        img_not_ann_indexes = set(img_not_ann_indexes) - set(ann_pixels)
        tmp = list(zip(*iter(img_not_ann_indexes)))
        img[tmp[1], tmp[0], :] = 1
    # fig config
    fig, ax = plt.subplots()
    dpi = 300
    w, h, _ = [i / 300 for i in img.shape]
    fig.set_size_inches(w, h)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_axis_off()
    ax.imshow(img)
    # plot rectangles
    for ann_rectangle in ann_rectangles:
        ax.add_patch(ann_rectangle)
    # save fig
    fig.savefig(output_path, dpi=300)
    plt.clf()


if __name__ == "__main__":
    img_path = "src/assets/image.png"
    coco_json_path = "src/assets/coco_labels.json"
    output_path = "src/assets/image_box.png"
    ann = "seal"
    only_show_anns = False
    save_boxed_img(img_path, coco_json_path, output_path, ann, only_show_anns)
