import cv2
import imutils
import numpy as np
import os

import Cards

input_from_user = input("If you want to use computer webcam press 1, "
                        "for IP Cam Server press ENTER ")

if input_from_user == '1':
  cap = cv2.VideoCapture(0)
else:
  pasted_URL = input("Paste the IP Camera Server URL ")
  cap = cv2.VideoCapture(
    f'{pasted_URL}/video')  # Ændres, hvis der skal testes. Skrives der '1' i stedet, vil webcam kunne anvendes

while True:
  ret, frame = cap.read()
  frame = imutils.resize(frame, 640)
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Ændrer farven til grå

  dilate = Cards.preprocces_image(gray)

  contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  cv2.imshow('Dialated', dilate)

  # Draw all contours
  temp_contours = []

  for cnt in contours:
    area = cv2.contourArea(cnt)
    print(cv2.contourArea(cnt))
    if area >= 750:
      temp_contours.append(cnt)

  cv2.drawContours(frame, temp_contours, -1, (0, 255, 0), 3)

  cv2.imshow('Contours', frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()
