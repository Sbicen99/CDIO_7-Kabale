import cv2
import numpy as np
from datetime import datetime

class Card(object):

    # constructor
    def __init__(self,
                 card_cascade_path, card_cascade_minneighbors, card_is_red,
                 figure_cascade_path, figure_cascade_minneighbors, figure_amount,
                 motif_cascade_path, motif_cascade_minneighbors, motif_amount):
        self.card_cascade = self._set_cascade(card_cascade_path)
        self.card_cascade_minneighbors = card_cascade_minneighbors
        self.card_is_red = card_is_red
        self.figure_cascade = self._set_cascade(figure_cascade_path)
        self.figure_cascade_minneighbors = figure_cascade_minneighbors
        self.figure_amount = figure_amount
        self.motif_cascade = self._set_cascade(motif_cascade_path)
        self.motif_cascade_minneighbors = motif_cascade_minneighbors
        self.motif_amount = motif_amount

    # set cascade
    def _set_cascade(self, cascade_path):

        if cascade_path == None:
            return None

        return cv2.CascadeClassifier(cascade_path)