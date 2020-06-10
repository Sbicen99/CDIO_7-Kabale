
import os
import time
import numpy as np
# Import necessary packages
import cv2
from imutils.video import videostream

import Cards

STACK_MAX_AREA = 300000
STACK_MIN_AREA = 2500

class Query_card:
    """Structure to store information about query cards in the camera image."""

    def __init__(self):
        self.contour = []  # Contour of card
        self.width, self.height = 0, 0  # Width and height of card
        self.corner_pts = []  # Corner points of card
        self.center = []  # Center point of card
        self.warp = []  # 200x300, flattened, grayed, blurred image
        self.rank_img = []  # Thresholded, sized image of card's rank
        self.suit_img = []  # Thresholded, sized image of card's suit
        self.best_rank_match = "Unknown"  # Best matched rank
        self.best_suit_match = "Unknown"  # Best matched suit
        self.rank_diff = 0  # Difference between rank image and best matched train rank image
        self.suit_diff = 0  # Difference between suit image and best matched train suit image


def findstack(thresh_image):
    """Finds all card-sized contours in a thresholded camera image.
    Returns the number of cards, and a list of card contours sorted
    from largest to smallest."""

    # Find contours and sort their indices by contour size
    cnts, hier = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    index_sort = sorted(range(len(cnts)), key=lambda i: cv2.contourArea(cnts[i]), reverse=True)

    # If there are no contours, do nothing
    if len(cnts) == 0:
        return [], []

    # Otherwise, initialize empty sorted contour and hierarchy lists
    cnts_sort = []
    hier_sort = []
    cnt_is_card = np.zeros(len(cnts), dtype=int)

    # Fill empty lists with sorted contour and sorted hierarchy. Now,
    # the indices of the contour list still correspond with those of
    # the hierarchy list. The hierarchy array can be used to check if
    # the contours have parents or not.
    for i in index_sort:
        cnts_sort.append(cnts[i])
        hier_sort.append(hier[0][i])

    # Determine which of the contours are cards by applying the
    # following criteria: 1) Smaller area than the maximum card size,
    # 2), bigger area than the minimum card size, 3) have no parents,
    # and 4) have four corners

    for i in range(len(cnts_sort)):
        size = cv2.contourArea(cnts_sort[i])
        peri = cv2.arcLength(cnts_sort[i], True)
        approx = cv2.approxPolyDP(cnts_sort[i], 0.01 * peri, True)

        if ((size < STACK_MAX_AREA) and (size > STACK_MIN_AREA)
                and (hier_sort[i][3] == -1) and (len(approx) == 4)):
            cnt_is_card[i] = 1

    return cnts_sort, cnt_is_card


def findcornerpts(contour):
    """Uses contour to find information about the query card. Isolates rank
    and suit images from the card."""

    # Initialize new Query_card object
    qCard = Query_card()

    qCard.contour = contour

    # Find perimeter of card and use it to approximate corner points
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.01 * peri, True)
    pts = np.float32(approx)
    qCard.corner_pts = pts

    for point in pts:
        print (point)

############## Python-OpenCV Playing Card Detector ###############
#
# Author: Evan Juras
# Date: 9/5/17
# Description: Python script to detect and identify playing cards
# from a PiCamera video feed.
#


## Camera settings

### ---- INITIALIZATION ---- ###
# Define constants and initialize variables

## Camera settings

IM_WIDTH = 1280
IM_HEIGHT = 720
FRAME_RATE = 2
BLUE_COLOR = (255, 0, 0)
RED_COLOR = (0, 0, 255)

## Initialize calculated frame rate because it's calculated AFTER the first time it's displayed
frame_rate_calc = 1
freq = cv2.getTickFrequency()
## Define font to use
font = cv2.FONT_HERSHEY_SIMPLEX
# Initialize camera object and video feed from the camera. The video stream is set up
# as a seperate thread that constantly grabs frames from the camera feed.
# See VideoStream.py for VideoStream class definition
## IF USING USB CAMERA INSTEAD OF PICAMERA,
## CHANGE THE THIRD ARGUMENT FROM 1 TO 2 IN THE FOLLOWING LINE:
# videostream = VideoStream.VideoStream((IM_WIDTH, IM_HEIGHT), FRAME_RATE, 2, 0).start()
time.sleep(1)  # Give the camera time to warm up

# Load the train rank and suit images
path = os.path.dirname(os.path.abspath(__file__))
train_ranks = Cards.load_ranks(path + '/card_Imgs/ranks/')
train_suits = Cards.load_suits(path + '/card_Imgs/suits/')

### ---- MAIN LOOP ---- ###
# The main loop repeatedly grabs frames from the video stream
# and processes them to find and identify playing cards.

cam_quit = 0  # Loop control variable

input_from_user = input("If you want to use computer webcam press 1, "
                        "for IP Cam Server press ENTER ")
if input_from_user == '1':
    cap = cv2.VideoCapture(1)
else:
    pasted_URL = input("Paste the IP Camera Server URL ")
    cap = cv2.VideoCapture(
        f'{pasted_URL}/video')  # Ændres, hvis der skal testes. Skrives der '1' i stedet, vil webcam kunne anvendes

hej = True
# Begin capturing frames
while cam_quit == 0:
    ##### resize.resize(path + '/training_imgs/IMG_0817.jpg')
    # Grab frame from video stream
    ###### image = videostream.read()
    ###### image = cv2.imread(path + '/training_imgs/temp-test.jpg')

    ret, frame = cap.read()

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()
    # Pre-process camera image (gray, blur, and threshold it)
    pre_proc = Cards.preprocces_image(frame)

    # Find and sort the contours of all cards in the image (query cards)
    cnts_sort, cnt_is_card = findstack(pre_proc)

    if hej == True:
        findcornerpts(cnts_sort[0])

    hej = False

    # If there are no contours, do nothing
    if len(cnts_sort) != 0:

        # Initialize a new "cards" list to assign the card objects.
        # k indexes the newly made array of cards.
        cards = []
        k = 0

        # For each contour detected:
        for i in range(len(cnts_sort)):
            # print(cnt_is_card[i])
            if (cnt_is_card[i] == 1):
                # Create a card object from the contour and append it to the list of cards.
                # preprocess_card function takes the card contour and contour and
                # determines the cards properties (corner points, etc). It generates a
                # flattened 200x300 image of the card, and isolates the card's
                # suit and rank from the image.

                cards.append(Cards.preprocess_card(cnts_sort[i], frame))

                # Find the best rank and suit match for the card.
                cards[k].best_rank_match, cards[k].best_suit_match, cards[k].rank_diff, cards[
                    k].suit_diff = Cards.match_card(cards[k], train_ranks, train_suits)

                # Draw center point and match result on the image.
                frame = Cards.draw_results(frame, cards[k])
                k = k + 1

        # Draw card contours on image (have to do contours all at once or
        # they do not show up properly for some reason)
        if (len(cards) != 0):
            temp_cnts = []
            for i in range(len(cards)):
                temp_cnts.append(cards[i].contour)
            cv2.drawContours(frame, temp_cnts, -1, (255, 0, 0), 2)

    # Draw framerate in the corner of the image. Framerate is calculated at the end of the main loop,
    # so the first time this runs, framerate will be shown as 0.
    cv2.putText(frame, "FPS: " + str(int(frame_rate_calc)), (10, 26), font, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "KORTBUNKE ", (10, 50), font, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "GRUNDBUNKER ", (720, 26), font, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "BYGGESTABLER ", (10, 500), font, 0.7, (255, 0, 255), 2, cv2.LINE_AA)

    # Draw the lines into the frame for splitting the card piles. This may make it easier to identify cards.
    cv2.line(frame, (0, 450), (frame.size, 450), BLUE_COLOR, 5)
    cv2.line(frame, (700, 0), (700, 450), RED_COLOR, 5)

    # Resize the frame.
    scale_percent = 60  # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    # Finally, display the image with the identified cards!
    cv2.imshow("Card Detector", frame)

    # cv2.imshow("Preprossed image", Cards.preprocces_image(image))

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1 / time1

    # Poll the keyboard. If 'q' is pressed, exit the main loop.
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        cam_quit = 1

# Close all windows and close the IP Camera video stream.
cap.release()
cv2.destroyAllWindows()

