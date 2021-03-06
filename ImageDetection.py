import cv2
import imutils
import numpy

import Cards


# Used for checking image processing and find countours works on different card setups pr 17/04 everything works as
# intented
def main():
    card_location = 'training_imgs/'

    test_images = [card_location + "1_image.png", card_location + "2_image.png", card_location + "3_image.png",
                   card_location + "4_image.png", card_location + "5_image.png", card_location + "6_image.png",
                   card_location + "7_image.png"]

    for imagepath in test_images:

        print_img = cv2.imread(imagepath)
        print_frame = imutils.resize(print_img, 640, 640)

        image = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
        frame = imutils.resize(image, 640, 640)

        # Standard prerpoccesing of input
        dilate = Cards.preprocces_image(frame)

        contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        corners = cv2.goodFeaturesToTrack(image, 1000, 0.1, 20, 4)
        if corners is not None:
            corners = numpy.int0(corners)
            for corner in corners:
                set = corner.ravel()

                x, y = set
                cv2.circle(print_frame, (x, y), 5, (0, 0, 255), -1)

        temp_contours = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 30000 >= area >= 1200:
                temp_contours.append(cnt)

        cv2.drawContours(print_frame, temp_contours, -1, (100, 200, 300), 3)
        cv2.imshow(imagepath, print_frame)

        print(imagepath + " number of contours %d -> " % len(temp_contours))

    cv2.waitKey(0)
    cv2.destroyAllWindows()


main()
