import cv2
import numpy as np

cap = cv2.VideoCapture(1)

if (cap.isOpened() == False):
    print('Failed to open camear')
else:
    ret = cap.set(3, 1280)
    ret = cap.set(4, 720)
    num = 0
    while(1):
        ret, img = cap.read()
        cv2.imshow('frame', img)

        k = cv2.waitKey(1) & 0xff

        if k == 32: # 空格存图
            num += 1
            cv2.imwrite('D:/camera_calibration/test4/'+str(num)+'.png', img)

        if k == 27:
            break
    cv2.destroyAllWindows()