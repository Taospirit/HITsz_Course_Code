import cv2
import numpy as np
import time

cap = cv2.VideoCapture(1)

# while(cap.isOpened()):
#     rect, frame = cap.read()
#     #print(frame.shape)
#     print(frame[240,320])
#     cv2.imshow('cap',frame)
#     cv2.waitKey(1)

if (cap.isOpened() == False):
    print('failed to open the camera')
else:
    while(True):
        #rect, frame = cap.read()
        frame = cv2.imread('D:\cv.png')

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        img = cv2.GaussianBlur(frame_gray, (5, 5), 0)

        #img = cv2.Canny(img, 50, 150, 3)
        #cv2.imshow('img', img)

        circles = np.zeros([1, 8, 3])
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1,20, param1=59, param2=30, minRadius=60, maxRadius=110)

        #print(len(circles))
        #print(circles)

        if circles is None:
            print('No circle in sight!')
        else:
            for i in circles[0, :]:
                cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 1)
                cv2.circle(frame, (i[0], i[1]), 1, (0, 0, 255), 3)
                print(circles)

        cv2.imshow('frame', frame)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

cv2.destroyAllWindows()