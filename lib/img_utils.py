from typing import Tuple

from numpy import ndarray


def rel2abs(img: ndarray, coord: Tuple[float, float]) -> Tuple[int, int]:
    """
    change relative coordinate into absolute coordinate inside the image
    """
    h, w, _ = img.shape
    cx, cy = int(coord[0] * w), int(coord[1] * h)
    return cx, cy
