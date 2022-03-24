import attr
import av
import cv2
import streamlit
from streamlit_webrtc import webrtc_streamer


@attr.s
class VideoProcessor:
    my_name = attr.ib(default='')

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        cv2.putText(img=img, text=str(self.my_name),
                    org=(10, 35),
                    fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2,
                    color=(0, 0, 255, 0),
                    thickness=2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


my_name = streamlit.text_input('Show my name', '')
ctx = webrtc_streamer(key="Image", video_processor_factory=VideoProcessor)

if ctx.video_processor:
    ctx.video_processor.my_name = my_name
