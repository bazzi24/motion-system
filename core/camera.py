import cv2

class Camera:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()
