import time


class Framerate():
    """
    calculate the fps to embed it in video

    >>> framerate = Framerate()
    >>> while True:
    >>>     success, img = cap.read()
    >>>     fps = framerate.next()
    >>>     debug_layer.process(img, framerate.next(), 0)
    >>>
    >>>     cv2.imshow("Image", img)
    >>>     cv2.waitKey(1)
    """

    def __init__(self):
        self.current_time = time.time()
        self.previous_time = time.time()

    def next(self) -> int:
        self.current_time = time.time()
        fps = 1 / (self.current_time - self.previous_time)
        self.previous_time = self.current_time

        return int(fps)
