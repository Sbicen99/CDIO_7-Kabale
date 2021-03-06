############## Python-OpenCV Playing Card Detector ###############
#
# Author: Evan Juras
# Date: 9/5/17
# Description: Python script to detect and identify playing cards
# from a PiCamera video feed.
#

# Import necessary packages
import cv2
import json
import numpy as np
import os
import os
import sched
import time
import time
import time

import Cards
import extractimages


## Camera settings

def writeCardJson(cards, i: int):
    if cards[i] is None:
        return None
    else:
        return {
            "suit": cards[i].best_suit_match[0],
            "rank": Cards.rank_converter(cards[i].best_rank_match.upper())
        }


## Camera settings

def writeJson(cards):
    cardsJson = {
        "waste": writeCardJson(cards, 7),
        "tableau1": writeCardJson(cards, 0),
        "tableau2": writeCardJson(cards, 1),
        "tableau3": writeCardJson(cards, 2),
        "tableau4": writeCardJson(cards, 3),
        "tableau5": writeCardJson(cards, 4),
        "tableau6": writeCardJson(cards, 5),
        "tableau7": writeCardJson(cards, 6),
    }
    return json.dumps(cardsJson)


## Define font to use
font = cv2.FONT_HERSHEY_SIMPLEX
# Initialize camera object and video feed from the camera. The video stream is set up
# as a seperate thread that constantly grabs frames from the camera feed.
# See VideoStream.py for VideoStream class definition
## IF USING USB CAMERA INSTEAD OF PICAMERA,
## CHANGE THE THIRD ARGUMENT FROM 1 TO 2 IN THE FOLLOWING LINE:

# Load the train rank and suit images
path = os.path.dirname(os.path.abspath(__file__))
train_ranks = Cards.load_ranks(path + '/Card_Imgs/ranks/')
train_suits = Cards.load_suits(path + '/Card_Imgs/suits/')

### ---- MAIN LOOP ---- ###
# The main loop repeatedly grabs frames from the video stream
# and processes them to find and identify playing cards.

cam_quit = 0  # Loop control variable
# Henter kamera kallibrerings variable.
mtx = np.load('Callibration_files/mtx_gustav.npy')
dist = np.load('Callibration_files/dist_gustav.npy')

input_from_user = input("If you want to use computer webcam press 1, "
                        "for IP Cam Server press ENTER ")
if input_from_user == '1':
    cap = cv2.VideoCapture(0)
    time.sleep(1)
else:
    pasted_URL = input("Paste the IP Camera Server URL ")
    cap = cv2.VideoCapture(
        f'{pasted_URL}/video')  # Ændres, hvis der skal testes. Skrives der '1' i stedet, vil webcam kunne anvendes

# Begin capturing frames
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
framecounter = 0
qCard1 = Cards.Query_card()
qCard2 = Cards.Query_card()
qCard3 = Cards.Query_card()
qCard4 = Cards.Query_card()
qCard5 = Cards.Query_card()
qCard6 = Cards.Query_card()
qCard7 = Cards.Query_card()
qCard8 = Cards.Query_card()
cards = [qCard1, qCard2, qCard3, qCard4, qCard5, qCard6, qCard7, qCard8]
cam_quit = 0  # Loop control variable
oldintersections = [[[], []], [[], []], [[], []], [[], []], [[], []], [[], []], [[], []], [[], []]]

while cam_quit == 0:
    framecounter += 1
    # Grab frame from video stream
    ###### image = videostream.read()
    ###### image = cv2.imread(path + '/training_imgs/temp-test.jpg')
    ret, frame = cap.read()
    # frame = cv2.flip(frame, 1)
    frame = cv2.flip(frame, -1)
    # frame = cv2.flip(frame, 0)
    ##frame = cv2.imread(path + '/training_imgs/test_kabale.jpg')
    # Her bruges kamera perspektivet, den her linje og ned til frame = dst[y:y + h, x:x + w] skal udkommenteres
    # Hvis du ikke bruger din egen kamera kallibration.
    h, w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # undistort
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
    dst = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

    # crop the image
    x, y, w, h = roi
    # Framen bliver nu ændret med vores variable.
    frame = dst[y:y + h, x:x + w]

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Pre-process camera image (gray, blur, and threshold it)
    # pre_proc = Cards.preprocces_image(frame)

    subimagelist = extractimages.getimages(frame)

    k = 0
    width, height, channel = frame.shape
    for i in range(len(subimagelist)):
        # Find contour for kort stakken i billedet.
        # subimage = getattr(image, 'img')
        # cv2.imshow('hej', subimagelist[i])
        pre_proc = Cards.preprocces_image(subimagelist[i], i)
        cnts_sort, cnt_is_card, crns = Cards.find_cards(pre_proc)

        if len(crns) != 0:

            w, h, top1, top2, bot1, bot2 = Cards.CalculateCardPosition(crns, subimagelist[i], oldintersections[i])
            oldintersections[i][0] = bot1
            oldintersections[i][1] = bot2
            crns = [bot1, bot2, top1, top2]
            # cv2.circle(frame, (int(top1[0]), int(top1[1])), 6, (0, 255, 255), -1)
            # cv2.circle(frame, (int(top2[0]), int(top2[1])), 6, (0, 255, 255), -1)
            # cv2.circle(frame, (int(bot1[0]), int(bot1[1])), 6, (0, 0, 255), -1)
            # cv2.circle(frame, (int(bot2[0]), int(bot2[1])), 6, (0, 0, 255), -1)
            cards[k] = Cards.preprocess_card(subimagelist[i], crns, w, h, cards[k])

            cards[k].best_rank_match, cards[k].best_suit_match, cards[k].rank_diff, \
            cards[k].suit_diff = Cards.match_card(cards[k], train_ranks, train_suits)

            # Draw center point and match result on the image.
            # Vi bliver nødt til at shifte vores koordinater. De koordinater der kommer ud af vores cropped billeder
            # Kan vi ikke bruge på vores rigtige frame, derfor udregner vi hvad deres position burde være på det nye
            # billede.

            # Det her er for vores pilecard der bliver vendt
            if k != len(subimagelist) - 1:
                cards[k].center[0] = cards[k].center[0] + k * int(height / 7)
                cards[k].center[1] = cards[k].center[1] + int(width / 4)
                # Det her er for resten af kortene.
            else:
                cards[k].center[0] = cards[k].center[0] + 1 * int(height / 7)

            frame = Cards.draw_results(frame, cards[k])
            # Vi tilføjer lige et tomt kort hvis vores subbillede ikke indeholder et kort, det bruges når information
            # Sendes til java processen senere.
            # card = None
            # cards[k] = card
        # Counter til at holde styr på hvilket kort vi er på.
        k = k + 1

    # Draw framerate in the corner of the image. Framerate is calculated at the end of the main loop,
    # so the first time this runs, framerate will be shown as 0.

    cv2.putText(frame, "FPS: " + str(int(frame_rate_calc)), (10, 26), font, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "KORTBUNKE ", (10, 50), font, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "GRUNDBUNKER ", (2 * int(height / 7) + 20, 26), font, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "BYGGESTABLER ", (10, int(width / 4) + 35), font, 0.7, (255, 0, 255), 2, cv2.LINE_AA)

    # Draw the lines into the frame for splitting the card piles. This may make it easier to identify cards.
    cv2.line(frame, (0, int(width / 4)), (frame.size, int(width / 4)), BLUE_COLOR, 5)
    cv2.line(frame, (2 * int(height / 7), 0), (2 * int(height / 7), int(height / 7)), RED_COLOR, 5)
    cv2.line(frame, (int(height / 7), 0), (int(height / 7), int(height / 7)), RED_COLOR, 5)

    for i in range(7):
        if i == 0:
            pass
        else:
            cv2.line(frame, (i * int(height / 7), int(width / 4)), (int(width / 4), frame.size), RED_COLOR, 5)

    # Resize the frame.
    scale_percent = 80  # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    # Finally, display the image with the identified cards!
    cv2.imshow("Card Detector", frame)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1 / time1

    # Poll the keyboard. If 'q' is pressed, exit the main loop.
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        cam_quit = 1

    if framecounter >= int(frame_rate_calc):
        framecounter = 0
        # print('Updated json')

        with open('kabalen2.json', 'w') as d:
            data = writeJson(cards)
            d.write(data)

        with open('message.txt', 'r+') as f:
            if f.readline() == "clear":
                f.truncate(0)
                qCard1 = Cards.Query_card()
                qCard2 = Cards.Query_card()
                qCard3 = Cards.Query_card()
                qCard4 = Cards.Query_card()
                qCard5 = Cards.Query_card()
                qCard6 = Cards.Query_card()
                qCard7 = Cards.Query_card()
                qCard8 = Cards.Query_card()
                cards = [qCard1, qCard2, qCard3, qCard4, qCard5, qCard6, qCard7, qCard8]
                print("yea!")

    # This saves the cards names in a file and also cutting it down to its initials.
