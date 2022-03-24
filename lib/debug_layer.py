from typing import Union, Tuple

import cv2
from numpy import ndarray


def bbox(img: ndarray, coord: Tuple[int, int, int, int], color: Tuple[int, int, int]):
    """
    display the boundary box arround an object

    :param img:
    :param coord: coordinate of the boundary box (top, left, bottom, right)
    :param color:
    :return:
    """
    cv2.rectangle(img=img, pt1=(coord[1], coord[0]), pt2=(coord[3], coord[2]), color=color, thickness=2)


def pot(img: ndarray, coord: Tuple[int, int], color: Tuple[int, int, int]):
    cv2.circle(img=img, center=coord, radius=15, color=color, thickness=cv2.FILLED)


def text(img: ndarray, text: Union[int, str, float], line: int = 0):
    """
    display a debug message on a specific line

    >>> debug_layer.process(img, "hello fabien", 1)
    """
    cv2.putText(img=img, text=str(text),
                org=(10, 35 + line * 35),
                fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2,
                color=(0, 0, 255, 0),
                thickness=2)


