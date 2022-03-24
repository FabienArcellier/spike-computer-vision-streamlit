import av
import streamlit
from streamlit_webrtc import webrtc_streamer

from lib import debug_layer
from lib.framerate import Framerate
from service.hand_detector import HandDetector

hand_detector = HandDetector()
framerate = Framerate()

class VideoProcessor:

    def __init__(self):
        self.show_hand_landmarks = False
        self.show_hand_bbox = False
        self.show_thumb = False
        self.show_frame_rate = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        hand_detector.process(img, self.show_hand_landmarks)
        if hand_detector.count() > 0:
            if self.show_hand_bbox:
                bbox = hand_detector.hand_bbox(0)
                debug_layer.bbox(img, bbox, (0, 255, 0))

            if self.show_thumb:
                hand = hand_detector.hand(0)
                debug_layer.pot(img, hand[4], (255, 0, 0))

        if self.show_frame_rate:
            debug_layer.text(img, framerate.next(), 0)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

streamlit.title('Hand detection')
show_hand_landmarks = streamlit.checkbox('Show the hand landmark', True)
show_hand_bbox = streamlit.checkbox('Show the hand bounding box')
show_thumb = streamlit.checkbox('Show the thumb')
show_frame_rate = streamlit.checkbox('Show frame rate')

ctx = webrtc_streamer(key="Image", video_processor_factory=VideoProcessor)
if ctx.video_processor:
    ctx.video_processor.show_hand_landmarks = show_hand_landmarks
    ctx.video_processor.show_hand_bbox = show_hand_bbox
    ctx.video_processor.show_thumb = show_thumb
    ctx.video_processor.show_frame_rate = show_frame_rate
