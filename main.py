import cv2
import time
import mediapipe as mp

import numpy as np

capture = cv2.VideoCapture(0)
#capture.set(3,640)
#capture.set(4,480)

mpHands = mp.solutions.hands
mpDrawing = mp.solutions.drawing_utils
mpFace = mp.solutions.face_detection

import autopy



nTime = 0

nPosition = 0


if __name__ == '__main__':


    while capture.isOpened():
        cTime = time.time()
        fPs = 1 / (cTime - nTime)
        nTime = time.time()


        success , img = capture.read()
        finalImage = cv2.cvtColor(cv2.flip(img,1) , cv2.COLOR_BGR2RGB)

        h, w, c = finalImage.shape

        with mpFace.FaceDetection() as face:
            detection = face.process(finalImage)

            if detection.detections:
                for facialFeatures in detection.detections:
                    pass
                    #mpDrawing.draw_detection(finalImage, facialFeatures)

        with mpHands.Hands() as hands:
            result = hands.process(finalImage)

            if result.multi_hand_landmarks:

                finalImage.flags.writeable = True

                for handLndmrks in result.multi_hand_landmarks :




                    # if (result.multi_handedness[0]  == 'right  '):


                    for id , lm in enumerate(handLndmrks.landmark):


                        lmposX , lmposY = int(lm.x * w) , int(lm.y * h)

                        if id == 8 :

                            pPosition = lmposY
                            lmVelocity = nPosition - pPosition
                            nPosition = lmposY

                            cv2.circle(finalImage, (lmposX,lmposY) , 25 , (255,100,255) , cv2.FILLED)
                            cv2.putText(finalImage , f'Velocity is : {int(lmVelocity)}' ,
                                        (40,100),cv2.FONT_ITALIC, 1, (255, 255, 255), 3)

                            if lmVelocity < -150 :
                                #autopy.key.tap(autopy.key.Code.F2 )
                                print("done" , lmVelocity)



                mpDrawing.draw_landmarks(finalImage, handLndmrks, mpHands.HAND_CONNECTIONS)








        cv2.putText(finalImage, f'FPS is : {int(fPs)}', (40, 40), cv2.FONT_ITALIC, 1, (255, 255, 255), 3)
        cv2.imshow("My Image ", finalImage)

        cv2.waitKey(1)
