import cv2
import numpy as np
from datetime import datetime


class Webcam(object):

    # read image from webcam
    def read_image(self):
        return cv2.VideoCapture(0).read()[1]