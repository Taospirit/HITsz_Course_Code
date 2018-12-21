import cv2
import numpy as np

cap = cv2.VideoCapture(1)
ret = cap.set(3, 1280)
ret = cap.set(4, 720)


while(cap.isOpened()):
    rec, frame = cap.read()

    f_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    f_thresh = cv2.adaptiveThreshold(f_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 0)

    #f_bulr = cv2.
    f = cv2.morphologyEx(f_thresh, cv2.MORPH_OPEN, (10, 10))






    cv2.imshow('thresh', f_thresh)
    cv2.imshow('morph', f)

    #cv2.imshow('frame', frame)



    k = cv2.waitKey(1) & 0xff

    if k == 27:
        break
    if k == ord('s'):
        cv2.imwrite('ell_test.jpg', frame)


cv2.destroyAllWindows()