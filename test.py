# Import necessary packages
import os

import cv2
from PIL import Image

import Cards

image = Image.open('training_imgs/1_image.png')

# Load the train rank and suit images
path = os.path.dirname(os.path.abspath(__file__))
train_ranks = Cards.load_ranks(path + '/card_Imgs/')
train_suits = Cards.load_suits(path + '/card_Imgs/')

# Pre-process camera image (gray, blur, and threshold it)
pre_proc = Cards.preprocces_image(image)

# Find and sort the contours of all cards in the image (query cards)
cnts_sort, cnt_is_card = Cards.find_cards(pre_proc)

# If there are no contours, do nothing
if len(cnts_sort) != 0:

    # Initialize a new "cards" list to assign the card objects.
    # k indexes the newly made array of cards.
    cards = []
    k = 0

    # For each contour detected:
    for i in range(len(cnts_sort)):
        if (cnt_is_card[i] == 1):
            # Create a card object from the contour and append it to the list of cards.
            # preprocess_card function takes the card contour and contour and
            # determines the cards properties (corner points, etc). It generates a
            # flattened 200x300 image of the card, and isolates the card's
            # suit and rank from the image.
            cards.append(Cards.preprocess_card(cnts_sort[i], image))

            # Find the best rank and suit match for the card.
            cards[k].best_rank_match, cards[k].best_suit_match, cards[k].rank_diff, cards[
                k].suit_diff = Cards.match_card(cards[k], train_ranks, train_suits)

            # Draw center point and match result on the image.
            image = Cards.draw_results(image, cards[k])
            k = k + 1

        # Draw card contours on image (have to do contours all at once or
        # they do not show up properly for some reason)
        if (len(cards) != 0):
            temp_cnts = []
            for i in range(len(cards)):
                temp_cnts.append(cards[i].contour)
            cv2.drawContours(image, temp_cnts, -1, (255, 0, 0), 2)

# Finally, display the image with the identified cards!
cv2.imshow("Card Detector", image)
cv2.imshow("Preprossed image", Cards.preprocces_image(image))
