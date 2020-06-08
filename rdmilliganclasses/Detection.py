import cv2
import numpy as np
from datetime import datetime

class Detection(object):
    WINDOW_NAME = "Playing Card Detection System"

    # is card detected in image
    def is_card_detected_in_image(self, card, image):

        if (card.card_cascade is None) or (card.card_cascade_minneighbors is None):
            return False

        # do detection
        is_detected = False
        is_detected = self._detect_card_in_image(card, image)

        if is_detected == False:
            image = self._rotate_image(image)
            is_detected = self._detect_card_in_image(card, image)

        # save image to disk
        self._save_image(image)

        # show image in window
        cv2.imshow(self.WINDOW_NAME, image)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()

        # indicate whether card detected in image
        return is_detected

    # detect card in image
    def _detect_card_in_image(self, card, colour_image):

        # detect cards
        gray_image = cv2.cvtColor(colour_image, cv2.COLOR_RGB2GRAY)
        cards = card.card_cascade.detectMultiScale(gray_image, scaleFactor=1.1,
                                                   minNeighbors=card.card_cascade_minneighbors)

        for (x, y, w, h) in cards:
            rio_colour = colour_image[y:y + h, x:x + w]

            # detect colour
            if card.card_is_red is not None:
                has_red_colour = self._has_red_colour(rio_colour)
                if (card.card_is_red and not has_red_colour) or (not card.card_is_red and has_red_colour):
                    continue

            # detect figures
            if (card.figure_cascade is not None) and (card.figure_cascade_minneighbors is not None) and (
                    card.figure_amount is not None):
                figure_count = 0
                roi_gray = cv2.cvtColor(rio_colour, cv2.COLOR_RGB2GRAY)
                figure_count += len(card.figure_cascade.detectMultiScale(roi_gray, scaleFactor=1.1,
                                                                         minNeighbors=card.figure_cascade_minneighbors))
                roi_gray = self._rotate_image(roi_gray)
                figure_count += len(card.figure_cascade.detectMultiScale(roi_gray, scaleFactor=1.1,
                                                                         minNeighbors=card.figure_cascade_minneighbors))
                print
                'Figure count: {}'.format(figure_count)  # debug only
                if card.figure_amount != figure_count:
                    continue

            # detect motifs
            if (card.motif_cascade is not None) and (card.motif_cascade_minneighbors is not None) and (
                    card.motif_amount is not None):
                motif_count = 0
                roi_gray = cv2.cvtColor(rio_colour, cv2.COLOR_RGB2GRAY)
                motif_count += len(card.motif_cascade.detectMultiScale(roi_gray, scaleFactor=1.1,
                                                                       minNeighbors=card.motif_cascade_minneighbors))
                roi_gray = self._rotate_image(roi_gray)
                motif_count += len(card.motif_cascade.detectMultiScale(roi_gray, scaleFactor=1.1,
                                                                       minNeighbors=card.motif_cascade_minneighbors))
                print
                'Motif count: {}'.format(motif_count)  # debug only
                if card.motif_amount != motif_count:
                    continue

            cv2.rectangle(colour_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            return True

        return False

    # rotate image
    def _rotate_image(self, img):
        (h, w) = img.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, 180, 1.0)
        return cv2.warpAffine(img, M, (w, h))

    # save image to disk
    def _save_image(self, img):
        filename = datetime.now().strftime('%Y%m%d_%Hh%Mm%Ss%f') + '.jpg'
        cv2.imwrite("WebCam/Detection/" + filename, img)

    # detect red colour
    def _has_red_colour(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        threshold = cv2.inRange(hsv, np.array([0, 90, 60]), np.array([10, 255, 255]))
        print
        'Red count: {}'.format(cv2.countNonZero(threshold))  # debug only
        return cv2.countNonZero(threshold) > 0