'''
@Description: Copyright Reserved
@Author: lintao
@Github: https://github.com/Taospirit
@Date: 2019-07-10 11:42:51
@LastEditors: lintao
@LastEditTime: 2019-07-10 11:42:55
'''

import numpy as np
import cv2 as cv
import cv2.aruco as aruco

def main():
    camera_matrix = np.array(([693.2, 0, 666.8], # 内参矩阵
                            [0, 693.4, 347.7],
                            [0, 0, 1]), dtype=np.double)
    dist_coefs = np.array([-0.050791, 0.217163, 0.0000878, -0.000388, -0.246122],
    dtype=np.double) # k1 k2 p1 p2 k3
    WIDETH = 640
    HEIGHT = 480

    cap = cv.VideoCapture(0)
    ret = cap.set(3, WIDETH)
    ret = cap.set(4, HEIGHT)
    font = cv.FONT_HERSHEY_SIMPLEX

    cv.namedWindow('frame', cv.WINDOW_AUTOSIZE)
    cv.moveWindow('frame', 700, 300)

    frame = cv.imread("./1.png")
    # print (type(frame))
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict)
    corners = np.array(corners)
    ids = np.array(ids)
    print ('*********************')
    print ("len of ids is {}, ids is {}, ids shape is {}".format(len(ids), ids, ids.shape))
    print ("len of corners is {}, corners is {}".format(len(corners), corners))
    print ("corners shape is {}".format(corners.shape))

    if ids is not None:
        id_show = [[ids[i][0], corners[i][0][0][0], corners[i][0][0][1]] for i in range(len(corners))]

        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.053, camera_matrix, dist_coefs)
        print ("rvec is {}, tvec is {}, shape is {}, {}".format(rvec, tvec, np.array(rvec).shape, np.array(tvec).shape))
        
        for i in range(rvec.shape[0]):
            aruco.drawAxis(frame, camera_matrix, dist_coefs, rvec[i, :, :], tvec[i, :, :], 0.03)
            aruco.drawDetectedMarkers(frame, corners, ids)

        for elem in id_show:
            cv.putText(frame, 'id='+str(elem[0]), (elem[1], elem[2]), font, 0.8, (0, 0, 255), 2, cv.LINE_AA)
    else:
        cv.putText(frame, "No Ids in sight!", (50, 100), font, 1, (0, 255,0),2,cv.LINE_AA)  
        
    cv.imshow("frame", frame)

    if cv.waitKey(0)&0xff == 27:
        cv.destroyAllWindows()
    
if __name__ == "__main__":
    main()