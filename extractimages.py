from PIL import Image


im = Image.open('training_imgs/test_kabale.jpg')
width, height = im.size
print(width)
print(height)
# The box is a 4-tuple defining the left, upper, right, and lower pixel coordinate.
crop_rectangle = (0, 0, 1000, 1000)
cropped_im = im.crop(crop_rectangle)


picturelist = [cropped_im]

# første crop foregår i øverste hjørne, og vi skal bruge syv rækker
# De her crops er til de 4 bygge stabler.
crop1 = (0*width/7, 0, 1*width/7, height/4)
crop2 = (1*width/7, 0, 2*width/7, height/4)
crop3 = (2*width/7, 0, 3*width/7, height/4)
crop4 = (3*width/7, 0, 4*width/7, height/4)


cropped_im = im.crop(crop1)
cropped_im.show()

cropped_im = im.crop(crop2)
cropped_im.show()

cropped_im = im.crop(crop3)
cropped_im.show()

cropped_im = im.crop(crop4)
cropped_im.show()

# Det her crop finder det kort der er nyt fra bunken.

crop_newcard = (6*width/7, 0, 7*width/7, height/4)
cropped_im = im.crop(crop_newcard)
cropped_im.show()

# Sidste crops finder de 7 stabler på bordet.

crop_pile1 = (0*width/7, height/4, 1*width/7, height)
crop_pile2 = (1*width/7, height/4, 2*width/7, height)
crop_pile3 = (2*width/7, height/4, 3*width/7, height)
crop_pile4 = (3*width/7, height/4, 4*width/7, height)
crop_pile5 = (4*width/7, height/4, 5*width/7, height)
crop_pile6 = (5*width/7, height/4, 6*width/7, height)
crop_pile7 = (6*width/7, height/4, 7*width/7, height)


cropped_im = im.crop(crop_pile1)
cropped_im.show()

cropped_im = im.crop(crop_pile2)
cropped_im.show()

cropped_im = im.crop(crop_pile3)
cropped_im.show()

cropped_im = im.crop(crop_pile4)
cropped_im.show()

cropped_im = im.crop(crop_pile5)
cropped_im.show()

cropped_im = im.crop(crop_pile6)
cropped_im.show()

cropped_im = im.crop(crop_pile7)
cropped_im.show()