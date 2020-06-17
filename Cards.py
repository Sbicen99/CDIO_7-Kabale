# Playing Card Detector Functions ###############
#
# Author: Evan Juras
# Date: 9/5/17
# Description: Functions and classes for CardDetector.py that perform 
# various steps of the card detection algorithm


import cv2
# Import necessary packages
import numpy as np
import os
import math
import os
from PIL import Image

# Constants #

# Adaptive threshold levels

BKG_THRESH = 60
CARD_THRESH = 30

# Width and height of card corner, where rank and suit are
CORNER_WIDTH = 32
CORNER_HEIGHT = 84

# Dimensions of rank train images
RANK_WIDTH = 70
RANK_HEIGHT = 125

# Dimensions of suit train images
SUIT_WIDTH = 70
SUIT_HEIGHT = 100

RANK_DIFF_MAX = 2000
SUIT_DIFF_MAX = 700

CARD_MAX_AREA = 2000000
CARD_MIN_AREA = 25000

font = cv2.FONT_HERSHEY_SIMPLEX


def preprocces_image(image):
    # cv2.imshow('Card class recieved image', image)

    blur = cv2.GaussianBlur(image, (9, 9), 0)

    edges = cv2.Canny(blur, 50, 150, True)

    # cv2.imshow('edges', edges)

    kernel = np.ones((3, 3), np.uint8)
    dilate = cv2.dilate(edges, kernel, iterations=1)

    return dilate


# Structures to hold query card and train card information #

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


class Train_ranks:
    """Structure to store information about train rank images."""

    def __init__(self):
        self.img = []  # Thresholded, sized rank image loaded from hard drive
        self.name = "Placeholder"


class Train_suits:
    """Structure to store information about train suit images."""

    def __init__(self):
        self.img = []  # Thresholded, sized suit image loaded from hard drive
        self.name = "Placeholder"


# Functions ###
def load_ranks(filepath):
    """Loads rank images from directory specified by filepath. Stores
    them in a list of Train_ranks objects."""

    train_ranks = []
    i = 0

    for Rank in os.listdir(filepath):
        if Rank.endswith(".jpg"):
            train_ranks.append(Train_ranks())
            train_ranks[i].name = os.path.splitext(Rank)[0]
            filename = Rank
            train_ranks[i].img = cv2.imread(filepath + filename, cv2.IMREAD_GRAYSCALE)
            i = i + 1

    return train_ranks


def load_suits(filepath):
    """Loads suit images from directory specified by filepath. Stores
    them in a list of Train_suits objects."""

    train_suits = []
    i = 0

    for Suit in os.listdir(filepath):
        if Suit.endswith(".jpg"):
            train_suits.append(Train_suits())
            train_suits[i].name = os.path.splitext(Suit)[0]
            filename = Suit
            train_suits[i].img = cv2.imread(filepath + filename, cv2.IMREAD_GRAYSCALE)
            i = i + 1

    return train_suits


def preprocess_imageOLD(image):
    """Returns a grayed, blurred, and adaptively thresholded camera image."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)

    # The best threshold level depends on the ambient lighting conditions.
    # For bright lighting, a high threshold must be used to isolate the cards
    # from the background. For dim lighting, a low threshold must be used.
    # To make the card detector independent of lighting conditions, the
    # following adaptive threshold method is used.
    #
    # A background pixel in the center top of the image is sampled to determine
    # its intensity. The adaptive threshold is set at 50 (THRESH_ADDER) higher
    # than that. This allows the threshold to adapt to the lighting conditions.
    img_w, img_h = np.shape(image)[:2]
    bkg_level = gray[int(img_h / 100)][int(img_w / 2)]
    thresh_level = bkg_level + BKG_THRESH

    retval, thresh = cv2.threshold(blur, thresh_level, 255, cv2.THRESH_BINARY)
    return thresh


def find_cards(thresh_image):
    """Finds all card-sized contours in a thresholded camera image.
    Returns the number of cards, and a list of card contours sorted
    from largest to smallest."""

    # Find contours and sort their indices by contour size
    cnts, hier = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    index_sort = sorted(range(len(cnts)), key=lambda i: cv2.contourArea(cnts[i]), reverse=True)

    # If there are no contours, do nothing
    if len(cnts) == 0:
        return [], [], []

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
    crns = []
    box = []
    for i in range(len(cnts_sort)):
        size = cv2.contourArea(cnts_sort[i])
        peri = cv2.arcLength(cnts_sort[i], True)
        #approx = cv2.approxPolyDP(cnts_sort[i], 0.01 * peri, True)
        rect = cv2.minAreaRect(cnts_sort[i])
        box = cv2.boxPoints(rect)

        if ((size < CARD_MAX_AREA) and (size > CARD_MIN_AREA)
                and (hier_sort[i][3] == -1) and (len(box) == 4)):
            cnt_is_card[i] = 1
            # print(approx)
            crns = box

    return cnts_sort, cnt_is_card, crns


def preprocess_card(image, pts, w, h):
    """Uses contour to find information about the query card. Isolates rank
    and suit images from the card."""

    # Initialize new Query_card object
    qCard = Query_card()

    # qCard.contour = contour

    # Find perimeter of card and use it to approximate corner points
    # peri = cv2.arcLength(contour, True)
    # approx = cv2.approxPolyDP(contour, 0.01 * peri, True)
    # pts = np.float32(approx)
    qCard.corner_pts = pts

    # Find width and height of card's bounding rectangle
    qCard.width, qCard.height = w, h

    # Find center point of card by taking x and y average of the four corners.
    average = np.sum(pts, axis=0) / len(pts)
    cent_x = int(average[0])
    cent_y = int(average[1])
    qCard.center = [cent_x, cent_y]

    # Warp card into 200x300 flattened image using perspective transform
    qCard.warp = flattener(image, pts)


    cv2.imshow("200x300 card", qCard.warp)


    # Tager fat i nederste venstre hjørner og zoomer x4
    Qcorner = qCard.warp[300-CORNER_HEIGHT:295, 5:CORNER_WIDTH+2]
    Qcorner_zoom = cv2.resize(Qcorner, (0, 0), fx=4, fy=4)

    # Sample known white pixel intensity to determine good threshold level
    white_level = Qcorner_zoom[15, int((CORNER_WIDTH * 4) / 2)]
    thresh_level = white_level - CARD_THRESH

    # Flipper det så det vender ordenligt
    Qcorner_zoom = cv2.flip(Qcorner_zoom, -1)
    cv2.imshow('Qcorner', Qcorner_zoom)

    # Laver det om så vi kan bruge cv2.threshold.
    gray_Qcorner = cv2.cvtColor(Qcorner_zoom, cv2.COLOR_BGR2GRAY)

    # Retval bruges ikke - OTSU giver den rigtige thresh baseret på hvilke farver der eksisterer.
    (thresh, im_bw) = cv2.threshold(gray_Qcorner, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)


    # Split in to top and bottom half (top shows rank, bottom shows suit)
    Qrank = im_bw[20:190, 0:135]
    Qsuit = im_bw[150:336, 0:135]

    # cv2.imshow('Qrank thresh', Qrank)
    # cv2.imshow('Qsuit thresh', Qsuit)

    # Find rank contour and bounding rectangle, isolate and find largest contour
    Qrank_cnts, hier = cv2.findContours(Qrank, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    Qrank_cnts = sorted(Qrank_cnts, key=cv2.contourArea, reverse=True)

    # Find bounding rectangle for largest contour, use it to resize query rank
    # image to match dimensions of the train rank image
    if len(Qrank_cnts) != 0:
        x1, y1, w1, h1 = cv2.boundingRect(Qrank_cnts[0])
        Qrank_roi = Qrank[y1:y1 + h1, x1:x1 + w1]
        Qrank_sized = cv2.resize(Qrank_roi, (RANK_WIDTH, RANK_HEIGHT), 0, 0)
        qCard.rank_img = Qrank_sized
        cv2.imshow('qCard.rank_img', qCard.rank_img)

    # Find suit contour and bounding rectangle, isolate and find largest contour
    Qsuit_cnts, hier = cv2.findContours(Qsuit, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    Qsuit_cnts = sorted(Qsuit_cnts, key=cv2.contourArea, reverse=True)

    # Find bounding rectangle for largest contour, use it to resize query suit
    # image to match dimensions of the train suit image
    if len(Qsuit_cnts) != 0:
        x2, y2, w2, h2 = cv2.boundingRect(Qsuit_cnts[0])
        Qsuit_roi = Qsuit[y2:y2 + h2, x2:x2 + w2]
        Qsuit_sized = cv2.resize(Qsuit_roi, (SUIT_WIDTH, SUIT_HEIGHT), 0, 0)
        qCard.suit_img = Qsuit_sized
        cv2.imshow('qCard.suit', qCard.suit_img)

    return qCard


def match_card(qCard, train_ranks, train_suits):
    """Finds best rank and suit matches for the query card. Differences
    the query card rank and suit images with the train rank and suit images.
    The best match is the rank or suit image that has the least difference."""

    best_rank_match_diff = 10000
    best_suit_match_diff = 10000
    best_rank_match_name = "Unknown"
    best_suit_match_name = "Unknown"
    i = 0

    # If no contours were found in query card in preprocess_card function,
    # the img size is zero, so skip the differencing process
    # (card will be left as Unknown)
    if (len(qCard.rank_img) != 0) and (len(qCard.suit_img) != 0):

        # Difference the query card rank image from each of the train rank images,
        # and store the result with the least difference
        for Trank in train_ranks:

            diff_img = cv2.absdiff(qCard.rank_img, Trank.img)
            rank_diff = int(np.sum(diff_img) / 255)

            if rank_diff < best_rank_match_diff:
                best_rank_diff_img = diff_img
                best_rank_match_diff = rank_diff
                best_rank_name = Trank.name

        # Same process with suit images
        for Tsuit in train_suits:

            diff_img = cv2.absdiff(qCard.suit_img, Tsuit.img)
            suit_diff = int(np.sum(diff_img) / 255)

            if suit_diff < best_suit_match_diff:
                best_suit_diff_img = diff_img
                best_suit_match_diff = suit_diff
                best_suit_name = Tsuit.name

    # Combine best rank match and best suit match to get query card's identity.
    # If the best matches have too high of a difference value, card identity
    # is still Unknown
    if best_rank_match_diff < RANK_DIFF_MAX:
        best_rank_match_name = best_rank_name

    if best_suit_match_diff < SUIT_DIFF_MAX:
        best_suit_match_name = best_suit_name

    best_suit_match_list = best_suit_match_name.split('_')
    best_suit_match_name = best_suit_match_list[0]

    best_rank_match_list = best_rank_match_name.split('_')
    best_rank_match_name = best_rank_match_list[0]

    # Return the identiy of the card and the quality of the suit and rank match
    return best_rank_match_name, best_suit_match_name, best_rank_match_diff, best_suit_match_diff


def draw_results(image, qCard):
    """Draw the card name, center point, and contour on the camera image."""

    x = qCard.center[0]
    y = qCard.center[1]
    cv2.circle(image, (x, y), 5, (255, 0, 0), -1)

    rank_name = qCard.best_rank_match
    suit_name = qCard.best_suit_match


    # Draw card name twice, so letters have black outline
    cv2.putText(image, (rank_name + ' of'), (x - 60, y - 10), font, 1, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, (rank_name + ' of'), (x - 60, y - 10), font, 1, (50, 200, 200), 2, cv2.LINE_AA)

    cv2.putText(image, suit_name, (x - 60, y + 25), font, 1, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, suit_name, (x - 60, y + 25), font, 1, (50, 200, 200), 2, cv2.LINE_AA)

    # Can draw difference value for troubleshooting purposes
    # (commented out during normal operation)
    # r_diff = str(qCard.rank_diff)
    # s_diff = str(qCard.suit_diff)
    # cv2.putText(image,r_diff,(x+20,y+30),font,0.5,(0,0,255),1,cv2.LINE_AA)
    # cv2.putText(image,s_diff,(x+20,y+50),font,0.5,(0,0,255),1,cv2.LINE_AA)

    return image


def flattener(image, pts):
    """Flattens an image of a card into a top-down 200x300 perspective.
    Returns the flattened, re-sized, grayed image.
    See www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/"""
    temp_rect = np.zeros((4, 2), dtype="float32")
    s = np.sum(pts, axis=1)
    # Vi starter med at udregne hvor vores 4 hjørner er i forhold til vores kort.

    topcornerlist = np.array([[1000000, 10000000], [100000000, 100000000]])

    for corn in pts:
        if corn[1] < topcornerlist[0][1]:
            topcornerlist[1] = topcornerlist[0]
            topcornerlist[0] = corn

        elif corn[1] < topcornerlist[1][1]:
            topcornerlist[1] = corn

    bottomcornerlist = np.array([[0, 0], [0, 0]])

    for corn in pts:
        if corn[1] > bottomcornerlist[0][1]:
            bottomcornerlist[1] = bottomcornerlist[0]
            bottomcornerlist[0] = corn

        elif corn[1] > bottomcornerlist[1][1]:
            bottomcornerlist[1] = corn

    # Please være søde ikke at rette i det her - Gustav

    # Hvis kortet hælder mod højre
    # top1 = venstre hjørne, top2 = højre
    # bot1 = højre hjørne bot2 = venstre hjørne
    if int(topcornerlist[0][1]) == int(topcornerlist[1][1]):
        temp_rect[0] = pts[np.argmin(s)]
        temp_rect[2] = pts[np.argmax(s)]
        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis=1)
        temp_rect[1] = pts[np.argmin(diff)]
        temp_rect[3] = pts[np.argmax(diff)]

    elif topcornerlist[0][0] < topcornerlist[1][0]:
        temp_rect[0] = topcornerlist[0]
        temp_rect[1] = topcornerlist[1]
        temp_rect[2] = bottomcornerlist[0]
        temp_rect[3] = bottomcornerlist[1]
    # Nu skifter vi. Den der har den laveste y værdi er nu modsatte hjørne.
    else:
        temp_rect[0] = topcornerlist[1]
        temp_rect[1] = topcornerlist[0]
        temp_rect[2] = bottomcornerlist[1]
        temp_rect[3] = bottomcornerlist[0]
        # if (bottomcornerlist[0][1] < bottomcornerlist[1][1]):
        #    temp_rect[0] = br
       #     temp_rect[3] = bl
       # else:
        #    temp_rect[0] = br
        #    temp_rect[3] = bl

    maxWidth = 200
    maxHeight = 300

    # Create destination array, calculate perspective transform matrix,
    # and warp card image
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], np.float32)
    M = cv2.getPerspectiveTransform(temp_rect, dst)
    warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # warp = cv2.cvtColor(warp,cv2.COLOR_BGR2GRAY)

    return warp

def CalculateCardPosition(crns):
    cornerlist = np.array([[0, 0], [0, 0]])
    # Finder de to højeste y værdier i vores array. De højeste yværdier er de nederste punkter.
    for corn in crns:
        if corn[1] >= cornerlist[0][1]:
            cornerlist[1] = cornerlist[0]
            cornerlist[0] = corn

        elif corn[1] >= cornerlist[1][1]:
            cornerlist[1] = corn

    vector = None
    if cornerlist[1][0] <= cornerlist[0][0]:
        vector = cornerlist[1] - cornerlist[0]
    else:
        vector = cornerlist[0] - cornerlist[1]

    # Den ortogonale vektor bruges til at udrenge approximationen for de to top punkter.
    orthogonal_vector = [-1.45*vector[1], 1.45*vector[0]]
    # width
    w = math.sqrt(math.pow(vector[0], 2) + math.pow(vector[1], 2))
    # height
    h = w*1.45

    topcorner1 = cornerlist[0] + orthogonal_vector
    topcorner2 = cornerlist[1] + orthogonal_vector

    return w, h, topcorner1, topcorner2, cornerlist[0], cornerlist[1]


def rank_converter(rank):
    switcher = {
        "ACE": "1",
        "TWO": "2",
        "THREE": "3",
        "FOUR": "4",
        "FIVE": "5",
        "SIX": "6",
        "SEVEN": "7",
        "EIGHT": "8",
        "NINE": "9",
        "TEN": "10",
        "JACK": "J",
        "QUEEN": "Q",
        "KING": "K"
    }
    return switcher.get(rank, " ")

