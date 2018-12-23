import cv2
import numpy as np
import math

WIDETH = 1280
HEIGHT = 1280

error_alpha = 0.5
radius_alpha = 0.5
max_radius = 200
min_radius = 10

method_num = 0
point_list = []
lpoint_list = []
choose_point_list = []

object_points = np.array(([566.0, 547.0, 0.0], [566.0, 349.0, 0.0], [566.0, 150.0, 0.0],
                          [330.0, 547.0, 0.0], [330.0, 349.0, 0.0], [330.0, 150.0, 0.0],
                          [164.0, 303.0, 0.0], [731.0, 303.0, 0.0]), dtype=np.double)
camera_matrix = np.array(([], []), dtype=np.double)
#dist_coefs = np.array((), dtype=np.double)


class Point:
    def __init__(self, x_param, y_param, num):
        self.x = x_param
        self.y = y_param
        self.n = num

def isEllipse(img, x, y, a, b):
    # 越界pass
    if x + a+10 > WIDETH or y + a+10 > HEIGHT:
        return False
    if x - a <0 or y - a <0:
        return False
    # 框内有黑pass
    for i in range(0, b-1):
        if img[y, x + i] == 0 or img[y, x - i] == 0:
            return False
        if img[y - i, x] == 0 or img[y + i, x] == 0:
            return False
    # 框外一点儿有白色pass
    for m in range(a+1, a+10):
        if img[y, x + m] == 255 or img[y, x - m] == 255:
            return False
        if img[y - m, x] == 255 or img[y + m,x] == 255:
            return False
    return True

def locatePoint(p_list, radius):
    temp = p_list[:]  #复制p_list给temp
    #                   1 4
    #   标准位置定义:   7     8
    #                   2 5
    #                   3 6

    #-----筛选出中间6点位置-----#
    Error = radius * error_alpha  # 参考误差待修正
    num = 0
    for i in range(0, len(p_list)):
        for j in range(i+1, len(p_list)):
            med_x = (p_list[i].x + p_list[j].x) / 2
            med_y = (p_list[i].y + p_list[j].y) / 2

            for m in range(0, len(p_list)):
                if m == i or m == j:
                    continue
                error = math.sqrt(pow(med_x - p_list[m].x, 2) + pow(med_y - p_list[m].y, 2))
                if error < Error:
                    addPoint(p_list, lpoint_list, i, num+1)
                    addPoint(p_list, lpoint_list, m, num+2)
                    addPoint(p_list, lpoint_list, j, num+3)
                    num += 3

                    #print(i+1, m+1, j+1)
                    temp[i].n = -1  # -1说明已经选定
                    temp[m].n = -1
                    temp[j].n = -1
    #-----筛选6点完毕------#

    #-----确定7\8点------#
    for p in range(0, len(temp)):
        if temp[p].n == -1:
            continue
        else:
            num += 1
            addPoint(p_list, lpoint_list, p, num)
            #lpoint_list.append(Point(p_list[p].x, p_list[p].y, num)) #先添加进list

    cen_78_y = (lpoint_list[6].y + lpoint_list[7].y)/2   # 7 和 8 两点的中点的纵坐标
    cen_25_y = (lpoint_list[1].y + lpoint_list[4].y)/2   # 2 和 5 两点的中点的纵坐标

    if cen_78_y < cen_25_y and lpoint_list[6].x > lpoint_list[7].x: # 图像是正的
        swapPoint(lpoint_list, 6, 7)
    if cen_78_y > cen_25_y and lpoint_list[6].x < lpoint_list[7].x: # 图像是倒的
        swapPoint(lpoint_list, 6, 7)
    #-----2点位置确定完毕------#


    #-----定位方案待选------#
    #把点7定位为1
    addPoint(lpoint_list, choose_point_list, 6, 1)


    #----- 定位方案一 ------#
    #           2  3
    #         1      4
    #           0  0
    #           0  0
    if method_num == 1:
        if cen_78_y < cen_25_y: # 图像是正的
            if distance(lpoint_list, 2, 6) > distance(lpoint_list, 5, 6): #待选点是点3和点6
                swapPoint(lpoint_list, 2, 5)
            addPoint(lpoint_list, choose_point_list, 2, 2)
            addPoint(lpoint_list, choose_point_list, 5, 3)


        if cen_78_y > cen_25_y: # 图像是倒的
            if distance(lpoint_list, 0, 6) > distance(lpoint_list, 3, 6): #待选点是点1和点4
                swapPoint(lpoint_list, 0, 3)
            addPoint(lpoint_list, choose_point_list, 0, 2)
            addPoint(lpoint_list, choose_point_list, 3, 3)

    #------ 定位方案二 ------#
    #用7\8点确定最中间2点，最终决定选取中间4点作为参考点，如下图：
    #          0  0
    #        1      4
    #          2  3
    #          0  0
    #-----问题:点距离太近,可能导致解算误差较大-------#
    if method_num == 2:
        if distance(lpoint_list, 1, 6) > distance(lpoint_list, 4, 6): #待选点是点2和点5
            swapPoint(lpoint_list, 1, 4)
        addPoint(lpoint_list, choose_point_list, 1, 2)
        addPoint(lpoint_list, choose_point_list, 4, 3)

    #----- 定位方案三 ----#
    #           0  0
    #         1      4
    #           0  0
    #           2  3
    #----- 问题:2、3点的位置过低，减小测距有效区间
    if method_num == 3:
        if cen_78_y < cen_25_y: # 图像是正的
            if distance(lpoint_list, 0, 6) > distance(lpoint_list, 3, 6): #带选点是点1和点4
                swapPoint(lpoint_list, 0, 3)
            addPoint(lpoint_list, choose_point_list, 0, 2)
            addPoint(lpoint_list, choose_point_list, 3, 3)

        if cen_78_y > cen_25_y: # 图像是倒的
            if distance(lpoint_list, 2, 6) > distance(lpoint_list, 5, 6): #待选点是点3和点6
                swapPoint(lpoint_list, 2, 5)
            addPoint(lpoint_list, choose_point_list, 2, 2)
            addPoint(lpoint_list, choose_point_list, 5, 3)
    # 点8为定位为4
    addPoint(lpoint_list, choose_point_list, 7, 4)
    #------排序完毕------#


def drawCenters(p_list, img):
    for i in range(0, len(p_list)):
        for j in range(i, len(p_list)):
            center_x = (p_list[i].x + p_list[j].x)/2
            center_y = (p_list[i].y + p_list[j].y)/2

            center_x_int = int(np.around(center_x))
            center_y_int = int(np.around(center_y))

            #cv2.circle(img, (center_x_int, center_y_int), 4, (0, 255, 0), -1)
            #cv2.imwrite('D:/img/test/'+str(n)+'.jpg', img)

def distance(p_list, i, j):
    if i > len(p_list) or j > len(p_list):
        return -1
    dis = math.sqrt(pow(p_list[i].x - p_list[j].x, 2) + pow(p_list[i].y - p_list[j].y, 2))
    return dis

def swapPoint(p_list, i, j): # 交换list中第i个索引和第j个索引数据的位置、索引
    p_list[i].n = j+1
    p_list[j].n = i+1
    p_list[i], p_list[j] = p_list[j], p_list[i]

def addPoint(src_list, new_list, i, n): #将src_list中的第i个索引的数据添加进new_list，且num为n
    p_new = Point(src_list[i].x, src_list[i].y, n)
    new_list.append(p_new)

number = 1
img = cv2.imread("D:/img/source/"+str(number)+".png")

method_num = 1

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, img_thresh = cv2.threshold(img_gray, 125, 255, cv2.THRESH_BINARY_INV)  # 阈值127，变成255
#cv2.imshow('thresh', img_thresh)

img_canny = cv2.Canny(img_thresh, 50, 150, 3)
#cv2.imshow('canny', img_canny)

image, contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours is None:
    print('No ellipse in sight!')
else:
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
            theta = ell[2]


            if isEllipse(img_thresh, cen_x, cen_y, a, b):
                count += 1
                sum = sum + b
                p_new = Point(cen_x, cen_y, count)

                #-----标记测试------#

                point_list.append(p_new)

                #font = cv2.FONT_HERSHEY_SIMPLEX
                #cv2.putText(img, str(count), (cen_x, cen_y), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

                #-----标记测试------#




                #print(cen_x, cen_y)
                #cv2.ellipse(img, ell, (0, 0, 255), 2)
                #cv2.circle(img, (cen_x, cen_y), 2, (0, 0, 255), -1)


    #---一帧检测完毕-----#
    #-----排序测试------#
    if len(point_list) != 8:
        print('no enough ellipse in sight')
    else:
        locatePoint(point_list, sum/count)

        #print(len(lpoint_list))
        # for i in range(0, len(lpoint_list)):
        #     #print(point_list[i].x, point_list[i].y, point_list[i].n)
        #     font = cv2.FONT_HERSHEY_SIMPLEX
        #     cv2.putText(img, str(lpoint_list[i].n), (lpoint_list[i].x, lpoint_list[i].y), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            #print(i, lpoint_list[i].x, lpoint_list[i].y, lpoint_list[i].n)
        for i in range(0, len(choose_point_list)):
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, str(choose_point_list[i].n), (choose_point_list[i].x, choose_point_list[i].y), font, 2, (0, 255, 0), 2,
                        cv2.LINE_AA)
            print(i, choose_point_list[i].x, choose_point_list[i].y, choose_point_list[i].n)
    #-----排序测试------#

    drawCenters(point_list, img)
    #found, rvec, tvec = cv2.solvePnP(object_points, lpoint_list, )
    cv2.imshow('ell', img)
    cv2.waitKey(0)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        cv2.destroyAllWindows()


