import av
import mediapipe
from streamlit_webrtc import webrtc_streamer

import cv2

hands = mediapipe.solutions.hands.Hands()
landmark_layer = mediapipe.solutions.drawing_utils


class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        #libimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        output = hands.process(img)
        if output.multi_hand_landmarks:
            for i, hand_landmark in enumerate(output.multi_hand_landmarks):
                landmark_layer.draw_landmarks(img, hand_landmark, mediapipe.solutions.hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(key="Image", video_processor_factory=VideoProcessor)
