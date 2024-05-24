# Date: 2024/4/17
import random
import numpy as np
from functools import lru_cache
from numpy.typing import NDArray
from skimage import color, filters, morphology


def _segment_fore_back(img: NDArray[np.uint8]) -> NDArray[bool]:
    """
    Segment a colorful image to distinguish foreground (represented by False)
    and background (represented by True).
    """
    gray_img = color.rgb2gray(img)
    smooth_img = filters.gaussian(gray_img, sigma=1)
    threshold_value = filters.threshold_otsu(smooth_img)
    bin_img = smooth_img > threshold_value
    # if True is more common than False, invert color
    bin_img = bin_img if bin_img.sum() > bin_img.size / 2 else ~bin_img
    closed_img = morphology.closing(bin_img, morphology.square(3))
    segmented_img = np.stack([closed_img for _ in range(3)], axis=-1)
    return segmented_img


class PaintingObj:
    """
    Painting to random paste single object.
    """

    def __init__(
        self, img: NDArray[np.uint8], obj: NDArray[np.uint8], ann: int, by_conflict: bool
    ) -> None:
        """
        `ann` could be 0 (ins) or 1 (seal). If `by_conflict` is True, the object bbox
        would be conflict with the image's foreground, else would not be conflict with
        foreground.
        """
        self.img = img
        self.obj = obj
        self.ann = ann
        self.by_conflict = by_conflict

    @property
    @lru_cache
    def segmented_image(self) -> NDArray[bool]:
        return _segment_fore_back(self.img)

    def _random_location_dic(self) -> dict[tuple[int], NDArray[np.uint8]]:
        """
        Randomly select a location for the object within the image. The condition is
        that the entire object must fit within the image (no cropping).
        """
        img_h, img_w, _ = self.img.shape
        # calculate object width & height
        obj_h, obj_w, _ = self.obj.shape
        obj_h_half, obj_w_half = obj_h // 2, obj_w // 2
        # randomly generate object center index based image shape
        obj_center_x = random.randint(0 + obj_w_half + 1, img_w - obj_w_half - 2)
        obj_center_y = random.randint(0 + obj_h_half + 1, img_h - obj_h_half - 2)
        # get object's location of the image
        obj_idx_x_start = obj_center_x - obj_w_half
        obj_idx_x = list(range(obj_idx_x_start, obj_idx_x_start + obj_w))
        obj_idx_y_start = obj_center_y - obj_h_half
        obj_idx_y = list(range(obj_idx_y_start, obj_idx_y_start + obj_h))
        # get the object location of image and its idx:val dict
        obj_loc = [(j, i) for i in obj_idx_x for j in obj_idx_y]
        obj_val_flat = self.obj.transpose(1, 0, 2).reshape(-1, 3)
        return {k: v for k, v in zip(obj_loc, obj_val_flat)}

    @property
    @lru_cache
    def obj_loc_dic(self) -> dict[tuple[int], NDArray[np.uint8]]:
        """
        The object bbox location dic of the image.

        Note: If can't find a good location (50 attempts), then return an empty dict.
        """
        for _ in range(50):
            obj_loc_dic = self._random_location_dic()
            obj_loc = obj_loc_dic.keys()
            is_conflict = sum([not all(self.segmented_image[i]) for i in obj_loc]) != 0
            cond = not is_conflict if self.by_conflict else is_conflict
            if cond:
                continue
            else:
                return obj_loc_dic
        else:
            return {}

    @property
    @lru_cache
    def bbox(self) -> list:
        """The bounding box with format `XYXY`"""
        if self.obj_loc_dic == {}:
            return []
        else:
            loc = self.obj_loc_dic.keys()
            ys, xs = [i[0] for i in loc], [i[1] for i in loc]
            x_min, y_min, x_max, y_max = min(xs), min(ys), max(xs), max(ys)
            ann_label = 0 if self.ann == "inscription" else 1
            return [ann_label, x_min, y_min, x_max, y_max]

    @property
    @lru_cache
    def obj_mask_dic(self) -> dict[tuple[int], NDArray[np.uint8]]:
        """The object mask location dic of the image."""

        def is_white(pixel: NDArray[np.uint8]) -> bool:
            """The shape of pixel must be (3,)"""
            return all(pixel == 255)

        return {k: v for k, v in self.obj_loc_dic.items() if not is_white(v)}

    @staticmethod
    def _assign_val_based_dic(
        img, obj_dic: dict[tuple[int], NDArray[np.uint8]]
    ) -> None:
        for k, v in obj_dic.items():
            img[k] = v
        return None

    @property
    @lru_cache
    def mask(self) -> NDArray[np.uint8]:
        """Mask with the object random pasted"""
        mask = np.zeros(self.img.shape, dtype=np.uint8)
        obj_mask_dic_white = {k: [255, 255, 255] for k in self.obj_mask_dic.keys()}
        self._assign_val_based_dic(mask, obj_mask_dic_white)
        return mask

    @property
    @lru_cache
    def img_pasted(self) -> NDArray[np.int8]:
        """Image with the object random pasted."""
        img_pasted = self.img.copy()
        self._assign_val_based_dic(img_pasted, self.obj_mask_dic)
        return img_pasted


class PaintingObjMulti:
    """
    Painting to random paste multiple objects.
    """

    def __init__(
        self,
        img: NDArray[np.uint8],
        obj_multi: list[NDArray[np.uint8]],
        ann_multi: list[int],
        by_conflict_ratio: float,
    ) -> None:
        """
        `ann_multi` could be list consist of 0 (ins) or 1 (seal). 
        The `by_conflict_ratio` parameter specifies the proportion of objects to be
        pasted such that they conflict with the foreground of the image.
        """
        self.img = img
        self.obj_multi = obj_multi
        self.ann_multi = ann_multi
        self.by_conflict_ratio = by_conflict_ratio

    def random_paste(self) -> None:
        # random generate `by_conflict_multi`
        conflict_n = int(len(self.obj_multi) * self.by_conflict_ratio)
        not_conflict_n = len(self.obj_multi) - conflict_n
        by_conflict_multi = [True] * conflict_n + [False] * not_conflict_n
        random.shuffle(by_conflict_multi)

        self.img_pasted = self.img.copy()
        self.bbox_multi = []
        self.mask_multi = []

        for i, obj in enumerate(self.obj_multi):
            painting_obj = PaintingObj(self.img_pasted, obj, self.ann_multi[i],
                                       by_conflict_multi[i])
            if painting_obj.bbox == []:
                print("[INFO] Dropped one object as it couldn't fit in the image")
                continue
            self.img_pasted = painting_obj.img_pasted
            self.bbox_multi.append(painting_obj.bbox)
            self.mask_multi.append(painting_obj.mask)

        self.mask = sum(self.mask_multi)
        if self.mask_multi != []:
            self.mask[self.mask != 0] = 255
        return None
