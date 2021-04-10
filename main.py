import cv2
import time
import mediapipe as mp

import numpy as np

capture = cv2.VideoCapture(0)
#capture.set(3,640)
#capture.set(4,480)

mpHands = mp.solutions.hands
mpDrawing = mp.solutions.drawing_utils



nTime = 0


if __name__ == '__main__':
    while capture.isOpened():
        cTime = time.time()
        fPs = 1 / (cTime - nTime)
        nTime = time.time()


        success , img = capture.read()



        finalImage = cv2.cvtColor(cv2.flip(img,1) , cv2.COLOR_RGBA2RGB)

        with mpHands.Hands() as hands:
            result = hands.process(finalImage)

            if result.multi_hand_landmarks:

                for handLndmrks in result.multi_hand_landmarks :
                    mpDrawing.draw_landmarks(finalImage , handLndmrks , mpHands.HAND_CONNECTIONS )




            finalImage.flags.writeable = True

        cv2.putText(finalImage, f'FPS is : {int(fPs)}', (40, 40), cv2.FONT_ITALIC, 1, (255, 255, 255), 3)
        cv2.imshow("My Image ", finalImage)

        cv2.waitKey(1)
