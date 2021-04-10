import cv2
import time
import mediapipe as mp
import numpy as np

capture = cv2.VideoCapture(0)
#capture.set(3,640)
#capture.set(4,480)

mpHands = mp.solutions.hands
nTime = 0


def print_hi(name):
 pass

if __name__ == '__main__':
 while True:
     cTime = time.time()
     fPs = 1 / (cTime - nTime)
     nTime = time.time()

     print(fPs)

     success , img = capture.read()
     print(success)
     cv2.putText(img, f'FPS is : {int(fPs)}', (40, 40), cv2.FONT_ITALIC, 1, (255, 255, 255), 3)
     cv2.imshow("My Image ", img)

     cv2.waitKey(1)
