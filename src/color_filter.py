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


def filter_red_by_lab(rgb_img_array, a_threshold=0, b_threshold=0, only_mask=True):
    lab = color.rgb2lab(rgb_img_array, illuminant="D65")
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


if __name__ == "__main__":
    img = color.rgba2rgb(io.imread("src/assets/image.png"))
    img_filtered_red = filter_red_by_lab(img)
    plt.imshow(img_filtered_red)
    plt.show()
