############## Python-OpenCV Playing Card Detector ###############
#
# Author: Evan Juras
# Date: 9/5/17
# Description: Python script to detect and identify playing cards
# from a PiCamera video feed.
#

import os
import time

# Import necessary packages
import cv2
from imutils.video import videostream

import Cards

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
train_ranks = Cards.load_ranks(path + '/card_Imgs/')
train_suits = Cards.load_suits(path + '/card_Imgs/')

### ---- MAIN LOOP ---- ###
# The main loop repeatedly grabs frames from the video stream
# and processes them to find and identify playing cards.

cam_quit = 0  # Loop control variable

input_from_user = input("If you want to use computer webcam press 1, "
                        "for IP Cam Server press ENTER ")
if input_from_user == '1':
    cap = cv2.VideoCapture(0)
else:
    pasted_URL = input("Paste the IP Camera Server URL ")
    cap = cv2.VideoCapture(
        f'{pasted_URL}/video')  # Ændres, hvis der skal testes. Skrives der '1' i stedet, vil webcam kunne anvendes

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
    cnts_sort, cnt_is_card, crns = Cards.find_cards(pre_proc)

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

            if crns is not None:
                for corn in crns:
                    print(corn)
                    x = corn[0][0]
                    y = corn[0][1]
                    cv2.circle(frame, (x,y), 5, (0,0,255), -1)

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

