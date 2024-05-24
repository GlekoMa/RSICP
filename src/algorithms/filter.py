# Date: 2024/3/21
import numpy as np
from typing import Tuple
from numpy.typing import NDArray
from skimage import color

A_THRESHOLD = 5
B_THRESHOLD = -5

# see DOI: 10.27307/d.cnki.gsjtu.2017.000228 (p63, 64, 81).
S_THRESHOLD = 0.17
V_THRESHOLD = (0.18, 0.86)


def filter_red_by_lab(
    rgb_img: NDArray[np.uint8], a_threshold: int, b_threshold: int
) -> NDArray[np.uint8]:
    """
    Filter the other colors using lab color space to only keep red pixels.
    Pixels greater than a_threshold or smaller than b_threshold would be considered as red.
    """
    lab = color.rgb2lab(rgb_img, illuminant="D65")
    a_cond = lab[:, :, 1] > a_threshold
    b_cond = lab[:, :, 2] < b_threshold
    lab[~(a_cond | b_cond)] = [100, 0, 0]
    rgb_filtered = color.lab2rgb(lab, illuminant="D65")
    rgb_filtered[np.isclose(rgb_filtered, 1, atol=1e-4)] = 1.
    return (rgb_filtered * 255).astype(np.uint8)


def filter_black_by_hsv(
    rgb_img: NDArray[np.uint8], s_threshold: int, v_threshold: Tuple[float, float]
) -> NDArray[np.uint8]:
    """
    Filter the other colors using hsv color space to only keep black pixels.
    Pixels smaller than s_threshold or between v_threshold will be considered as black.
    """
    hsv = color.rgb2hsv(rgb_img)
    s_cond = hsv[:, :, 1] < s_threshold
    v_cond = (hsv[:, :, 2] > v_threshold[0]) & (hsv[:, :, 2] < v_threshold[1])
    hsv[~(s_cond & v_cond)] = [0, 0, 1]
    rgb_filtered = color.hsv2rgb(hsv)
    rgb_filtered[np.isclose(rgb_filtered, 1, atol=1e-4)] = 1.
    return (rgb_filtered * 255).astype(np.uint8)


def auto_filter_seal_ins(img: NDArray[np.uint8], obj: str) -> NDArray[np.uint8]:
    """Auto filter based on the object's kind."""
    if obj == "seal":
        img_filtered = filter_red_by_lab(img, A_THRESHOLD, B_THRESHOLD)
    elif obj == "inscription":
        img_filtered = filter_black_by_hsv(img, S_THRESHOLD, V_THRESHOLD)
    else:
        raise ValueError("Argument 'obj' must be 'seal' or 'inscription'.")
    return img_filtered
