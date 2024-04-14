# Date: 2024/3/21
#
# TODO:
#   A filter maybe useful to filter noise
#   """
#   from skimage import filters
#   img = filters.median(img_noise)
#   """

import numpy as np
from numpy.typing import NDArray
from skimage import io, color
import matplotlib.pyplot as plt


def savefig(img: NDArray[np.uint8], output_path: str) -> None:
    """
    Save numpy array to image format.
    """
    fig, ax = plt.subplots()
    dpi = 300
    w, h, _ = [i / 300 for i in img.shape]
    fig.set_size_inches(w, h)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_axis_off()
    ax.imshow(img)
    fig.savefig(output_path, dpi=300)


def filter_red_by_lab(
    rgba_img_array, a_threshold=5, b_threshold=-5, only_keep_mask=False
):
    """
    Filter the other colors using lab color space to only keep red pixels.
    ---
    rgba_img_array:
        Usually loaded from a `png` format image.
    a_threshold:
        The threshold value for the 'a' channel in the Lab color space.
        Pixels with 'a' channel values greater this threshold will be considered as red.
    b_threshold:
        The threshold value for the 'b' channel in the Lab color space.
        Pixels with 'b' channel values below this threshold will be considered as red.
    only_keep_mask:
        If True, the other pixels (exclude the red pixels) of the image will be changed
        to white. Otherwise, only the red pixels will be changed to white.
    """
    rgb = color.rgba2rgb(rgba_img_array)
    lab = color.rgb2lab(rgb, illuminant="D65")
    a_cond = lab[:, :, 1] > a_threshold
    b_cond = lab[:, :, 2] < b_threshold
    red_cond = a_cond | b_cond
    if only_keep_mask:
        lab[red_cond] = [100, 0, 0]
    else:
        lab[~red_cond] = [100, 0, 0]
    rgb_filtered = color.lab2rgb(lab, illuminant="D65")
    return rgb_filtered


# The default value of s_threshold and v_threshold is according to
# DOI: 10.27307/d.cnki.gsjtu.2017.000228 (p63, 64, 81).
def filter_black_by_hsv(
    rgba_img_array, s_threshold=0.17, v_threshold=(0.18, 0.86), only_keep_mask=False
):
    """
    Filter the other colors using hsv color space to only keep black pixels.
    ---
    rgba_img_array:
        Usually loaded from a `png` format image.
    s_threshold:
        The threshold value for the 'saturation' channel in the hsv color space.
        Pixels with 'saturation' channel values below this threshold will be
        considered as black.
    v_threshold: tuple, e.g. (0.18, 0.86)
        The threshold value for the 'value' channel in the hsv color space.
        Pixels with 'value' channel values between this threshold will be
        considered as black.
    only_keep_mask:
        If True, the other pixels (exclude the black pixels) of the image will be changed
        to white. Otherwise, only the red pixels will be changed to white.
    """
    rgba_img_array = img
    rgb = color.rgba2rgb(rgba_img_array)
    hsv = color.rgb2hsv(rgb)
    # the threshold of black
    s_cond = hsv[:, :, 1] < s_threshold
    v_cond = (hsv[:, :, 2] > v_threshold[0]) & (hsv[:, :, 2] < v_threshold[1])
    black_cond = s_cond & v_cond
    if only_keep_mask:
        hsv[black_cond] = [0, 0, 1]
    else:
        hsv[~black_cond] = [0, 0, 1]
    rgb_filtered = color.hsv2rgb(hsv)
    return rgb_filtered


if __name__ == "__main__":
    img = io.imread("../assets/image.png")
    # filter others to keep red
    img_red_kept = filter_red_by_lab(img)
    savefig(img_red_kept, output_path="test_red.png")
    # filter others to keep black
    img_black_kept = filter_black_by_hsv(img)
    savefig(img_black_kept, output_path="test_black.png")
