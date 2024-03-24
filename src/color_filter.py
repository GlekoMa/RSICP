# Date: 2024/3/21
#
# Note:
#   A filter maybe useful to filter noise
#   """
#   from skimage import filters
#   img = filters.median(img_noise)
#   """

import numpy as np
from skimage import io, color
import matplotlib.pyplot as plt


def savefig(img, output_path):
    fig, ax = plt.subplots()
    dpi = 300
    w, h, _ = [i / 300 for i in img.shape]
    fig.set_size_inches(w, h)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_axis_off()
    ax.imshow(img)
    fig.savefig(output_path, dpi=300)


def filter_red_by_lab(rgba_img_array, a_threshold=0, b_threshold=0, only_mask=False):
    rgb = color.rgba2rgb(rgba_img_array)
    lab = color.rgb2lab(rgb, illuminant="D65")
    # the threshold of red
    a_cond = lab[:, :, 1] > a_threshold
    b_cond = lab[:, :, 2] < b_threshold
    red_cond = a_cond | b_cond
    if only_mask:
        # if red, let be white
        lab[red_cond] = [100, 0, 0]
    else:
        # if not red, let be white
        lab[~red_cond] = [100, 0, 0]
    rgb_filtered = color.lab2rgb(lab, illuminant="D65")
    return rgb_filtered

def filter_black_by_hsv(rgba_img_array, only_mask=False):
    rgba_img_array = img
    rgb = color.rgba2rgb(rgba_img_array)
    hsv = color.rgb2hsv(rgb)
    # the threshold of red
    s_cond = hsv[:, :, 1] < 0.17
    v_cond = (hsv[:, :, 2] > 0.18) & (hsv[:, :, 2] < 0.86)
    black_cond = s_cond & v_cond
    if only_mask:
        # if black, let be white
        hsv[black_cond] = [0, 0, 1]
    else:
        # if not black, let be white
        hsv[~black_cond] = [0, 0, 1]
    rgb_filtered = color.hsv2rgb(hsv)
    return rgb_filtered


if __name__ == "__main__":
    # seal red filtered
    img = io.imread("src/assets/image_box_seal.png")
    output_path = "src/assets/image_box_only_filtered_seal.png"
    img_filtered_red = filter_red_by_lab(img, a_threshold=5, b_threshold=-5, only_mask=False)
    savefig(img_filtered_red, output_path)
    # only seal red filtered
    img = io.imread("src/assets/image_only_box_seal.png")
    output_path = "src/assets/image_box_only_filtered_seal.png"
    img_filtered_red = filter_red_by_lab(img, a_threshold=5, b_threshold=-5, only_mask=False)
    savefig(img_filtered_red, output_path)
    # inscription black filtered
    img = io.imread("src/assets/image_box_inscription.png")
    output_path = "src/assets/image_box_only_filtered_insection.png"
    img_filtered_black = filter_black_by_hsv(img, only_mask=False)
    savefig(img_filtered_black, output_path)
    # only inscription black filtered
    img = io.imread("src/assets/image_only_box_inscription.png")
    output_path = "src/assets/image_box_only_filtered_insection.png"
    img_filtered_black = filter_black_by_hsv(img, only_mask=False)
    savefig(img_filtered_black, output_path)
