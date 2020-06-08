from PIL import Image
from resizeimage import resizeimage
import os

path = os.path.dirname(os.path.abspath(__file__))

def resize(pathInput):
    with open(pathInput, 'r+b') as f:
        with Image.open(f) as image:
            cover = resizeimage.resize_cover(image, [2048 / 2, 1536/ 2])
            cover.save(path + '/training_imgs/temp-test.jpg', image.format)