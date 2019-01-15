import cv2
import numpy as np
# 设置图像尺寸
WIDTH = 1280
HIGHT = 720
cap = cv2.VideoCapture(1)

if (cap.isOpened() == False):
    print('Failed to open camear')
else:
    ret = cap.set(3, WIDTH)
    ret = cap.set(4, HIGHT)
    num = 0
    while(1):
        ret, img = cap.read()
        k = cv2.waitKey(1) & 0xff
        if k == 32: # 空格存图
            num += 1
            cv2.imwrite('D:/camera_calibration/test4/'+str(num)+'.png', img)

        if k == 27:
            break
    cv2.destroyAllWindows()