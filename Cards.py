############## Playing Card Detector Functions ###############
#
# Author: Evan Juras
# Date: 9/5/17
# Description: Functions and classes for CardDetector.py that perform 
# various steps of the card detection algorithm


import cv2
import math
# Import necessary packages
import numpy as np
import os
from scipy.spatial import distance

# Constants #

# Adaptive threshold levels

BKG_THRESH = 60
CARD_THRESH = 30

# Width and height of card corner, where rank and suit are
CORNER_WIDTH = 32
CORNER_HEIGHT = 84
WIDTH_TO_HEIGHT_RATIO = 1.45
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


def preprocces_image(image, i):
    # cv2.imshow('Card class recieved image', image)

    blur = cv2.GaussianBlur(image, (9, 9), 0)

    edges = cv2.Canny(blur, 50, 150, True)

    kernel = np.ones((3, 3), np.uint8)
    dilate = cv2.dilate(edges, kernel, iterations=1)

    return dilate


# Structures to hold query card and train card information #

def distances(xy1, xy2):
    d0 = np.subtract.outer(xy1[:, 0], xy2[:, 0])
    d1 = np.subtract.outer(xy1[:, 1], xy2[:, 1])
    return np.hypot(d0, d1)


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
        # approx = cv2.approxPolyDP(cnts_sort[i], 0.01 * peri, True)
        rect = cv2.minAreaRect(cnts_sort[i])
        box = cv2.boxPoints(rect)

        if ((size < CARD_MAX_AREA) and (size > CARD_MIN_AREA)
                and (hier_sort[i][3] == -1) and (len(box) == 4)):
            cnt_is_card[i] = 1
            # print("Hello")
            crns = box

    return cnts_sort, cnt_is_card, crns


def preprocess_card(image, pts, w, h, qCard):
    """Uses contour to find information about the query card. Isolates rank
    and suit images from the card."""

    # Initialize new Query_card object

    qCard.corner_pts = pts

    # Assign the width and height of the bounding rectangle.
    qCard.width, qCard.height = w, h

    # Find center point of card by taking x and y average of the four corners.
    average = np.sum(pts, axis=0) / len(pts)
    cent_x = int(average[0])
    cent_y = int(average[1])
    qCard.center = [cent_x, cent_y]

    # Warp card into 200x300 flattened image using perspective transform
    qCard.warp = flattener(image, pts)
    cv2.imshow("200x300 card", qCard.warp)

    # Tager fat i nederste højre hjørner og zoomer x4
    Qcorner = qCard.warp[300 - CORNER_HEIGHT:295, 200 - CORNER_WIDTH:190]
    Qcorner_zoom = cv2.resize(Qcorner, (0, 0), fx=4, fy=4)

    # Flipper det så det vender ordenligt
    Qcorner_zoom = cv2.flip(Qcorner_zoom, -1)
    # cv2.imshow('Qcorner', Qcorner_zoom)

    # Laver det om så vi kan bruge cv2.threshold.
    gray_Qcorner = cv2.cvtColor(Qcorner_zoom, cv2.COLOR_BGR2GRAY)

    # Retval bruges ikke - OTSU giver den rigtige thresh baseret på hvilke farver der eksisterer.
    (thresh, im_bw) = cv2.threshold(gray_Qcorner, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # cv2.imshow('im', im_bw)
    # Split in to top and bottom half (top shows rank, bottom shows suit)
    Qrank = im_bw[20:190, 0:135]
    Qsuit = im_bw[170:336, 0:135]

    cv2.imshow('Qrank thresh', Qrank)
    cv2.imshow('Qsuit thresh', Qsuit)

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
        # cv2.imshow('qCard.rank_img', qCard.rank_img)

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
        # cv2.imshow('qCard.suit', qCard.suit_img)

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

    # Shorten the name since we match on pics with weird names.
    best_suit_match_list = best_suit_match_name.split('_')
    best_suit_match_name = best_suit_match_list[0]

    best_rank_match_list = best_rank_match_name.split('_')
    best_rank_match_name = best_rank_match_list[0]

    # Return the identiy of the card and the quality of the suit and rank match
    if best_rank_match_name == "Unknown" and best_suit_match_name == "Unknown":
        return qCard.best_rank_match, qCard.best_suit_match, best_rank_match_diff, best_suit_match_diff

    elif best_rank_match_name == "Unknown":
        return qCard.best_rank_match, best_suit_match_name, best_rank_match_diff, best_suit_match_diff

    elif best_suit_match_name == "Unknown":
        return best_rank_match_name, qCard.best_suit_match, best_rank_match_diff, best_suit_match_diff

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
    # Bare 4 store tal
    topcornerlist = np.array([[1000000, 10000000], [100000000, 100000000]])
    # Loop over for at finde de to øverste punkter
    for corn in pts:
        if corn[1] < topcornerlist[0][1]:
            topcornerlist[1] = topcornerlist[0]
            topcornerlist[0] = corn

        elif corn[1] < topcornerlist[1][1]:
            topcornerlist[1] = corn

    bottomcornerlist = np.array([[0, 0], [0, 0]])
    # Loop til at finde de to nederste
    for corn in pts:
        if corn[1] > bottomcornerlist[0][1]:
            bottomcornerlist[1] = bottomcornerlist[0]
            bottomcornerlist[0] = corn

        elif corn[1] > bottomcornerlist[1][1]:
            bottomcornerlist[1] = corn

    # Please være søde ikke at rette i det her - Gustav

    # Denne udregning er hvis kortet står i en 90* vinkel, og hælder derfor hverken mod venstre eller højre
    if int(topcornerlist[0][1]) == int(topcornerlist[1][1]):
        temp_rect[0] = pts[np.argmin(s)]
        temp_rect[2] = pts[np.argmax(s)]
        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis=1)
        temp_rect[1] = pts[np.argmin(diff)]
        temp_rect[3] = pts[np.argmax(diff)]

    # Hvis kortet hælder mod højre
    # top1 er venstre hjørne, top2 er højre hjørne
    # bot1 er højre hjørne bot2 er venstre hjørne

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

    # de her tal definerer det 200x300 warp vi trækker ud.
    maxWidth = 200
    maxHeight = 300

    # Create destination array, calculate perspective transform matrix,
    # and warp card image
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], np.float32)
    M = cv2.getPerspectiveTransform(temp_rect, dst)
    warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # warp = cv2.cvtColor(warp,cv2.COLOR_BGR2GRAY)

    return warp


def CalculateCardPosition(crns, image, oldlines):
    runs = False
    cornerlist = np.array([[0, 0], [0, 0]])
    # Finder de to højeste y værdier i vores array. De højeste y-værdier er de nederste punkter.

    for corn in crns:
        if corn[1] >= cornerlist[0][1]:
            cornerlist[1] = cornerlist[0]
            cornerlist[0] = corn

        elif corn[1] >= cornerlist[1][1]:
            cornerlist[1] = corn

    while True:
        vector = None
        if cornerlist[1][0] <= cornerlist[0][0]:
            vector = cornerlist[1] - cornerlist[0]
        else:
            vector = cornerlist[0] - cornerlist[1]

        # Den ortogonale vektor bruges til at udrenge approximationen for de to top punkter.
        orthogonal_vector = [-1 * WIDTH_TO_HEIGHT_RATIO * vector[1], WIDTH_TO_HEIGHT_RATIO * vector[0]]
        # width
        w = math.sqrt(math.pow(vector[0], 2) + math.pow(vector[1], 2))
        # height
        h = w * WIDTH_TO_HEIGHT_RATIO

        topcorner1 = cornerlist[0] + orthogonal_vector
        topcorner2 = cornerlist[1] + orthogonal_vector

        intersections = houghLinesCorners(image, cornerlist[0], cornerlist[1], topcorner1, topcorner2, oldlines)
        if intersections == None or runs == True:
            if runs is True:
                cv2.putText(image, ("Locked!"), (500, 50), font, 1, (0, 255, 0), 3, cv2.LINE_AA)
            else:
                cv2.putText(image, ("Locking..."), (500, 50), font, 1, (0, 255, 255), 3, cv2.LINE_AA)

            return w, h, topcorner1, topcorner2, cornerlist[0], cornerlist[1]
        cornerlist[0] = intersections[0]
        cornerlist[1] = intersections[1]
        # topcorner1 = intersections[2]
        # topcorner2 = intersections[3]
        runs = True


def houghLinesCorners(image, b1, b2, t1, t2, oldintersections):
    """---------------------Hough Lines---------------------------"""

    # Finds Maximum and minimum x and y af the points supplied to find where to crop the image
    xmin = None
    xmax = None
    ymin = None
    ymax = None
    for p in [b1, b2, t1, t2]:
        if xmin is None:  # Initialise the variables
            xmin = p[0]
        if xmax is None:
            xmax = p[0]
        if ymin is None:
            ymin = p[1]
        if ymax is None:
            ymax = p[1]

        if p[0] < xmin:  # Find min and max
            xmin = p[0]
        elif p[0] > xmax:
            xmax = p[0]
        if p[1] < ymin:
            ymin = p[1]
        elif p[1] > ymax:
            ymax = p[1]

    offsetX1 = -8  # An adjustable offset to fine tune the cropping
    offsetX2 = 8
    offsetY1 = 40
    offsetY2 = 20

    cropX1 = int(
        np.heaviside(0, xmin + offsetX1))  # Uses the heavyside/unit step function to make sure that the cropping points
    cropX2 = int(np.heaviside(0, xmax + offsetX2))  # are inside the picture i.e. if they are < 0 make them 0
    cropY1 = int(np.heaviside(0, ymin + offsetY1))
    cropY2 = int(np.heaviside(0, ymax + offsetY2))

    magfactor = 2  # factor for magnifying the image that is being worked on, called løl "name subject to change"

    # cv2.imshow("what i crop", image)
    lel = image[cropY1:cropY2, cropX1:cropX2]  # The function works on a cropped and magnified image "løl"
    if len(lel) == 0:  # if "løl" is empty, due to bad cropping or bad points fed to the function,
        # print("bad search") # return non and print bad search
        return None

    # Hvis kortet ligger helt ude til siden prøver den at tage et billede udenfor billedet, og den exception skal fanges
    try:
        lel = cv2.resize(lel, (0, 0), fx=magfactor,
                         fy=magfactor)  # magnify løl by the magfactor for better line/ edge detection
    except cv2.error:
        return None

    edges = cv2.Canny(lel, 150, 400, apertureSize=3)  # find edges

    ##cv2.circle(image, (cropX1, cropY1), 6, (255, 0, 255), -1)
    ##cv2.circle(image, (cropX2, cropY2), 6, (255, 0, 255), -1)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 125)  #
    # lines = [[[-184, 3.0717795]]]
    # print(lines)
    vlines = []
    hlines = []

    # Returnerer vi ikke bare de gamle her?
    if len(oldintersections[0]) != 0 and lines is None:
        if distance.euclidean(oldintersections[0], b1) > distance.euclidean(oldintersections[1], b1):
            if distance.euclidean(oldintersections[1], b1) > 200:
                return [b1, b2]
            else:
                return oldintersections
        else:
            if distance.euclidean(oldintersections[1], b1) > 200:
                return [b1, b2]
            else:
                return oldintersections

    if lines is not None:
        # print("found lines")
        for i in lines:
            rho = i[0][0]
            theta = i[0][1]

            # Determine if lines are vertical based on the angle theta from origin (origo) to the normal of the line
            if (theta >= 0 and theta < np.pi / 4) or (theta > 3 * np.pi / 4 and theta < 5 * np.pi / 4) or (
                    theta > 7 * np.pi / 4 and theta < 8 * np.pi / 4):
                vlines.append(i)

            # Determine if lines are horizontal based on the angle theta from origin (origo) to the normal of the line
            if (theta > np.pi / 4 and theta < 3 * np.pi / 4) or (theta > 5 * np.pi / 4 and theta < 7 * np.pi / 4):
                hlines.append(i)
            """
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            #cv2.line(frame, (x1, y1), (x2, y2), (188, 0, 188), 2)
            cv2.line(edges, (x1, y1), (x2, y2), (188, 0, 188), 1)
            """

        # print("vlines")
        # print(vlines)
        # print("hlines")
        # print(hlines)

        while len(vlines) > 2:
            mTheta = 0  # mean angle of the horizontal lines in radians
            sumTheta = 0
            j = 0
            outlinerX = 0  # index of the line whose angle deviates the most from the mean
            outlinerV = 0  # Variance of the line whose angle deviates the most
            for hl in vlines:
                sumTheta += hl[0][1]
            mTheta = sumTheta / len(vlines)
            for hl in vlines:
                currentV = (hl[0][1] - mTheta) ** 2
                if currentV > outlinerV:
                    outlinerV = currentV
                    outlinerX = j
                j += 1
            # print(outlinerV)
            del vlines[outlinerX]
        # print(len(vlines))

        for i in vlines:
            rho = i[0][0]
            theta = i[0][1]

            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            # cv2.line(frame, (x1, y1), (x2, y2), (188, 0, 188), 2)
            cv2.line(edges, (x1, y1), (x2, y2), (188, 0, 188), 1)

        for i in hlines:
            rho = i[0][0]
            theta = i[0][1]

            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho  # Find the first coordinate of the point on the line where it is orthogonal with the line to origin
            y0 = b * rho  # Find seccond coordinate here
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            # cv2.line(frame, (x1, y1), (x2, y2), (188, 0, 188), 2)
            cv2.line(edges, (x1, y1), (x2, y2), (188, 0, 188), 1)

        if len(vlines) == 2 and len(hlines) > 0:

            rline = 0
            lline = 0
            bline = hlines[0]

            if abs(vlines[0][0][0]) > abs(vlines[1][0][0]):
                rline = vlines[0]
                lline = vlines[1]
            else:
                rline = vlines[1]
                lline = vlines[0]

            for ln in hlines:
                if abs(ln[0][0]) > abs(bline[0][0]):
                    bline = ln

            # Calculate instersections
            intersections = []

            ah = -(np.cos(bline[0][1]) / np.sin(bline[0][1]))
            bh = bline[0][0] / np.sin(bline[0][1])

            for vl in [lline, rline]:

                if vl[0][1] == 0:  # if the current vertical line is exactly vertical (slope of line = ∞) with theta = 0
                    pointX = vl[0][0]
                    pointY = ah * pointX + bh

                elif vl[0][1] == np.pi:  # if theta = pi (perfectly vertical line flipped 180 degrees)
                    pointX = -vl[0][0]
                    pointY = ah * pointX + bh

                else:
                    av = -(np.cos(vl[0][1]) / np.sin(vl[0][1]))
                    bv = vl[0][0] / np.sin(vl[0][1])

                    pointX = (bh - bv) / (av - ah)
                    pointY = (av * bh - ah * bv) / (av - ah)

                intersections.append([pointX, pointY])

            estimates = []

            sideRelationFactor = 1.5

            # Pythagoras theorem to find distance between intersections
            intrsctwidth = np.sqrt(
                ((intersections[1][0] - intersections[0][0]) ** 2) + ((intersections[1][1] - intersections[0][1]) ** 2))

            # calculates the angle between the x-axis and the line for each side of the card
            if lline[0][0] >= 0:
                lsideangle = (lline[0][1] + 3 * np.pi / 2) % (2 * np.pi)
            else:
                lsideangle = (lline[0][1] + np.pi / 2) % (2 * np.pi)

            if rline[0][0] >= 0:
                rsideangle = (rline[0][1] + 3 * np.pi / 2) % (2 * np.pi)
            else:
                rsideangle = (rline[0][1] + np.pi / 2) % (2 * np.pi)

            lestimate = [intersections[0][0] + np.cos(lsideangle) * intrsctwidth * sideRelationFactor,
                         intersections[0][1] + np.sin(lsideangle) * intrsctwidth * sideRelationFactor]

            restimate = [intersections[1][0] + np.cos(rsideangle) * intrsctwidth * sideRelationFactor,
                         intersections[1][1] + np.sin(rsideangle) * intrsctwidth * sideRelationFactor]

            intersections.append(lestimate)
            intersections.append(restimate)

            cv2.circle(edges, (int(intersections[0][0]), int(intersections[0][1])), 6, (0, 255, 255), -1)
            cv2.circle(edges, (int(intersections[1][0]), int(intersections[1][1])), 6, (0, 255, 255), -1)
            cv2.circle(lel, (int(restimate[0]), int(restimate[1])), 6, (0, 255, 255), -1)
            cv2.circle(lel, (int(lestimate[0]), int(lestimate[1])), 6, (0, 255, 255), -1)

            # cv2.circle(løl, (int(intersections[0][0]), int(intersections[0][1])), 6, (0, 255, 255), -1)
            # cv2.circle(løl, (int(intersections[1][0]), int(intersections[1][1])), 6, (0, 255, 255), -1)

            for p in intersections:
                p[0] = p[0] / magfactor
                p[1] = p[1] / magfactor

                p[0] += cropX1
                p[1] += cropY1

            # cv2.circle(image, (int(intersections[0][0]), int(intersections[0][1])), 6, (0, 255, 255), -1)
            # cv2.circle(image, (int(intersections[1][0]), int(intersections[1][1])), 6, (0, 255, 255), -1)

            # print("Have found lines")
            # cv2.imshow("Lel", lel)
            # cv2.imshow("Lines", edges)

            if len(oldintersections[0]) != 0:
                if distance.euclidean(oldintersections[0], b1) < distance.euclidean(intersections[0], b1):
                    return oldintersections
                else:
                    return intersections

            return intersections

    # print("have not found lines")
    # cv2.imshow("Lel", lel)
    cv2.imshow("Lines", edges)
    return None

    """
        for i in [bline]:
            rho = i[0][0]
            theta = i[0][1]

            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho  # Find the first coordinate of the point on the line where it is orthogonal with the line to origin
            y0 = b * rho  # Find seccond coordinate here
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
    """


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
        "KING": "K",
        "UNKNOWN": "U"
    }
    return switcher.get(rank, " ")
