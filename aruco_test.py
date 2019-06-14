# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 11:59:46 2019

@author: lintao
"""
import numpy as np
import cv2
import cv2.aruco as aruco

camera_matrix = np.array(([693.2, 0, 666.8], # 内参矩阵
                          [0, 693.4, 347.7],
                          [0, 0, 1]), dtype=np.double)
dist_coefs = np.array([-0.050791, 0.217163, 0.0000878, -0.000388, -0.246122],
dtype=np.double) # k1 k2 p1 p2 k3
WIDETH = 1280
HEIGHT = 720

cap = cv2.VideoCapture(1)
ret = cap.set(3, WIDETH)
ret = cap.set(4, HEIGHT)
font = cv2.FONT_HERSHEY_SIMPLEX

while(True):
    ret, frame = cap.read()
    #cv2.imshow("frame", frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)  
    
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict)
    if ids is not None:
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coefs)
        for i in range(rvec.shape[0]):
            aruco.drawAxis(frame, camera_matrix, dist_coefs, rvec[i, :, :], tvec[i, :, :], 0.03)
            aruco.drawDetectedMarkers(frame, corners)
    else:
        cv2.putText(frame, "No Ids in sight!", (50, 100), font, 1, (0,255,0),2,cv2.LINE_AA)  
        
    cv2.imshow("frame", frame)
    if cv2.waitKey(1)&0xff == 27:
        break
cap.release()
cv2.destroyAllWindows()
    