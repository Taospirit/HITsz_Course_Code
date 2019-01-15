import cv2
import numpy as np
import math


max_radius = 500
min_radius = 10
radius_alpha = 0.5

img = cv2.imread("D:/img/ell.png")

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, img_thresh = cv2.threshold(img_gray, 125, 255, cv2.THRESH_BINARY_INV)  # 阈值127，变成255
#cv2.imshow('thresh', img_thresh)

img_canny = cv2.Canny(img_thresh, 50, 150, 3)
#cv2.imshow('canny', img_canny)

image, contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

font = cv2.FONT_HERSHEY_SIMPLEX  # 字体设置
if len(contours) == 0:
    cv2.putText(img, 'No enough ellipse in sight!', (50, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
else:
    print("椭圆个数：", len(contours))
    count = 0
    sum = 0
    for cnt in contours:
        if len(cnt) > 4:
            ell = cv2.fitEllipse(cnt)

            b_double = ell[1][0]  # 拟合的矩阵宽，即短半轴2倍
            a_double = ell[1][1]  # 拟合的矩阵长，即长半轴2倍

            # -----椭圆基本筛选-----#
            if a_double > max_radius or b_double > max_radius:
                continue
            if a_double < min_radius or b_double < min_radius:
                continue
            if a_double < b_double * radius_alpha or b_double < a_double * radius_alpha:
                continue
            # -----结束-----#

            # 开始对选定的进行处理
            cen_x = int(np.around(ell[0][0]))
            cen_y = int(np.around(ell[0][1]))
            a = int(np.around(a_double / 2))  # 长半轴
            b = int(np.around(b_double / 2))  # 短半轴
            theta = ell[2] * math.pi/180.0  # 转换成弧度

            offset_x = b * math.cos(theta)
            offset_y = b * math.sin(theta)
            if theta > 90:
                offset_x = -offset_x

            cenx = int(np.around(ell[0][0] + offset_x))
            ceny = int(np.around(ell[0][1] + offset_y))

            cv2.circle(img, (cenx, ceny), 5, (0, 255, 0), -1)



            # ------画椭圆及圆心-----#
            cv2.ellipse(img, ell, (0, 0, 255), 2)
            cv2.circle(img, (cen_x, cen_y), 2, (0, 0, 255), -1)
            #cv2.circle(img, (cen_x, cen_y), b, (255, 0, 0), -1)

            # cv2.putText(img, str(int(np.around(theta))), (cen_x, cen_y), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            # cv2.imwrite("D:/test_theta.png", img)

    #found, rvec, tvec = cv2.solvePnP(object_points, lpoint_list, )
    cv2.imshow('ell', img)
    cv2.waitKey(0)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        cv2.destroyAllWindows()


