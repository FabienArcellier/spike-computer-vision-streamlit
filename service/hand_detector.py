from typing import List, Any, Tuple

import cv2
import mediapipe
from numpy import ndarray

from lib import img_utils


class HandDetector:

    def __init__(self, *args, **kwargs):
        self.hands = mediapipe.solutions.hands.Hands(*args, **kwargs)
        self.landmark_layer = mediapipe.solutions.drawing_utils
        self.output = None
        self.img = None

    def count(self) -> int:
        if self.output is None or self.output.multi_hand_landmarks is None:
            return 0

        return len(self.output.multi_hand_landmarks)

    def hand(self, hand: int = 0) -> List[Tuple[int, int]]:
        hand_landmarks = self.output.multi_hand_landmarks[hand]
        return [img_utils.rel2abs(self.img, (hand_landmark.x, hand_landmark.y)) for hand_landmark in hand_landmarks.landmark]

    def hand_bbox(self, hand: int = 0) -> Tuple[int, int, int, int]:
        hand_landmarks = self.output.multi_hand_landmarks[hand]
        first_landmark = hand_landmarks.landmark[0]
        cx, cy = img_utils.rel2abs(self.img, (first_landmark.x, first_landmark.y))
        top, bottom, left, right = cy, cy, cx, cx
        for hand_landmark in hand_landmarks.landmark:
            cx, cy = img_utils.rel2abs(self.img, (hand_landmark.x, hand_landmark.y))
            top = cy if cy > top else top
            bottom = cy if cy < bottom else bottom
            left = cx if cx > left else left
            right = cx if cx < right else right

        return (top, left, bottom, right)

    def process(self, img: ndarray, display_landmark: bool = False) -> None:
        """
        >>> hand_detector = HandDetector()
        >>> hand_detector.process(img)
        >>> count_hands = hand_detector.count()
        >>> hands_landmark = hand_detector.hand(0)
        """
        adapted_img: ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.img = img
        self.output = self.hands.process(adapted_img)
        if display_landmark and self.output.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(self.output.multi_hand_landmarks):
                self.landmark_layer.draw_landmarks(img, hand_landmarks, mediapipe.solutions.hands.HAND_CONNECTIONS)
