from PIL import Image
import numpy as np
from numpy.typing import NDArray


def read_image(path: str) -> NDArray[np.uint8]:
    img = Image.open(path)
    return np.asarray(img).copy()


def write_png(path: str, img: NDArray[np.uint8]) -> None:
    img = Image.fromarray(img)
    img.save(path)
    return None
