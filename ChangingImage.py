import cv2
import imutils
import numpy as np
import os

import Cards

""""
Printer alle slags kort ud; printer hvert kort der detekteres, ændrer baggrunden på billederne (edge detection) og farver billederne grå
"""


def main():
    cardPath = 'training_imgs/1_image.png'

    print_img = cv2.imread(cardPath)
    print_frame = imutils.resize(print_img, 640, 640)

    image = cv2.imread(cardPath, cv2.IMREAD_GRAYSCALE)
    frame = imutils.resize(image, 640, 640)

    # Standard prerpoccesing of input
    dilate = Cards.preprocces_image(frame)

    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.imshow('Dialated', dilate)

    temp_contours = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(cv2.contourArea(cnt))
        if area <= 30000 and area >= 1200:
            temp_contours.append(cnt)

            card = cnt

            # Approximate the corner points of the card
            peri = cv2.arcLength(card, True)
            approx = cv2.approxPolyDP(card, 0.01 * peri, True)
            pts = np.float32(approx)

            x, y, w, h = cv2.boundingRect(card)

            # Flatten the card and convert it to 200x300
            warp = Cards.flattener(frame, pts)

            cv2.imshow(str(area), warp)

    cv2.drawContours(print_frame, temp_contours, -1, (0, 255, 0), 3)

    cv2.imshow('Contours', print_frame)

    print("number of contours %d -> " % len(temp_contours))

    cv2.waitKey(0)
    cv2.destroyAllWindows()


main()

main()
