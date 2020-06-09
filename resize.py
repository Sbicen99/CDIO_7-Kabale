import os

from PIL import Image
from resizeimage import resizeimage
import numpy as np
import cv2

path = os.path.dirname(os.path.abspath(__file__))

def resize(pathInput):
    with open(pathInput, 'r+b') as f:
        with Image.open(f) as image:
            width, height = image.size
            cover = resizeimage.resize_cover(image, [3022 / 2, 2691/ 2])
            cover.save(path + '/training_imgs/temp-test.jpg', image.format)