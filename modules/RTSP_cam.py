import cv2
import threading


class RTSPCapture(cv2.VideoCapture):
    _cur_frame = None  # 当前帧
    _reading = False  # 是否正在读取
    schemes = ["rtsp://"]  # 支持的协议

    def __init__(self, url, *schemes):
        super().__init__(url)
        self.capture = None
        self.frame_receiver = threading.Thread(target=self._recv_frame, daemon=True)  # 创建线程用于接收帧
        self.schemes.extend(schemes)
        if isinstance(url, str) and url.startswith(tuple(self.schemes)):
            self._reading = True
        elif isinstance(url, int):
            pass

    @classmethod
    def create(cls, url, *schemes):
        return cls(url, *schemes)

    def is_started(self):
        ok = self.isOpened()
        if ok and self._reading:
            ok = self.frame_receiver.is_alive()
        return ok

    def _recv_frame(self):
        while self._reading and self.isOpened():
            ok, frame = self.read()
            if not ok:
                break
            self._cur_frame = frame
        self._reading = False

    def refresh(self):
        frame = self._cur_frame
        self._cur_frame = None
        return frame is not None, frame

    def start_read(self):
        self.frame_receiver.start()
        self.capture = self.refresh if self._reading else self.read

    def stop_read(self):
        self._reading = False
        if self.frame_receiver.is_alive():
            self.frame_receiver.join()
