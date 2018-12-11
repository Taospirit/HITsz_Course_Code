import cv2
import numpy as np
import math
WIDETH = 1280
HEIGHT = 720
point_list = []
lpoint_list = []


class Point:
    def __init__(self, x_param, y_param, num):
        self.x = x_param
        self.y = y_param
        self.n = num

def is_ellipse(img, x, y, a, b): # x\y是椭圆质点坐标点，a\b是椭圆长短半轴
    # 越界pass
    if x + a+3 > WIDETH or y + a+3 > HEIGHT:#这里的3是留出的小裕量
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
    for m in range(a+1, a+3):#同样留出小裕量
        if img[y, x + m] == 255 or img[y, x - m] == 255:
            return False
        if img[y - m, x] == 255 or img[y + m,x] == 255:
            return False
    return True

def locate_point(p_list, lp_list):
    print(len(p_list))
    Error = 10 #参考误差
    num = 0
    for i in range(0, len(p_list)):
        for j in range(i+1, len(p_list)):
            med_x = (p_list[i].x + p_list[j].x) / 2
            med_y = (p_list[i].y + p_list[j].y) / 2
            
            for m in range(0, len(p_list)):
                if m == i or m == j:
                    continue         
                    
                error = pow(med_x - p_list[m].x, 2) + pow(med_y - p_list[m].y, 2)

                if error < Error:
                    p_top = Point(p_list[i].x, p_list[i].y, num+1)
                    p_med = Point(p_list[m].x, p_list[m].y, num+2)
                    p_but = Point(p_list[j].x, p_list[j].y, num+3)
                    num += 3  # 一次会找出3个点，所以这里加3
                    lpoint_list.append(p_top)
                    lpoint_list.append(p_med)
                    lpoint_list.append(p_but)
                    print(i+1, m+1, j+1)

cap = cv2.VideoCapture(1)

if (cap.isOpened() == False):
    print("Failed to open the camera...")
else:
    ret = cap.set(3, WIDETH) # 设置显示尺寸
    ret = cap.set(4, HEIGHT)
    
    while(True):
        ret, frame = cap.read()
        
        #-----基础图形处理-----#
        f_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        f_bulr = cv2.GaussianBlur(f_gray, (5, 5), 0)
        
        ret,f_thresh = cv2.threshold(f_bulr,125,255,cv2.THRESH_BINARY_INV) # 阈值127，变成255
        #cv2.imshow("thresh", f_thresh)
        f_can = cv2.Canny(f_thresh, 50, 150, 3)
        #-----结束-----#

        #-----一帧椭圆检测-------#
        image, contours, hierarchy = cv2.findContours(f_can, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        count =0  # 用来计数
        for cnt in contours:
            if len(cnt) > 4: # 点数超过5才能拟合椭圆
                ell = cv2.fitEllipse(cnt)

                b_double = ell[1][0] #拟合的矩阵宽，即短半轴2倍
                a_double = ell[1][1] #拟合的矩阵长，即长半轴2倍

                #-----椭圆基本筛选-----#
                if a_double > 130 or b_double > 130:
                    continue
                if a_double < 10 or b_double < 10:
                    continue
                if a_double < b_double*0.5:
                    continue
                if a_double > b_double*1.5:
                    continue
                # -----结束-----#

                # 整形处理，获得椭圆参数
                cen_x = int(np.around(ell[0][0]))
                cen_y = int(np.around(ell[0][1]))
                a = int(np.around(a_double/2))
                b = int(np.around(b_double/2))

                if  is_ellipse(f_thresh, cen_x, cen_y, a, b) == True:

                    #-----添加进组并计数、标记-----#
                    count += 1
                    p_new = Point(cen_x, cen_y, count)
                    point_list.append(p_new)

                    font = cv2.FONT_HERSHEY_SIMPLEX # 数字标记
                    cv2.putText(frame, str(count), (cen_x, cen_y), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    #-----结束------#
                  
                    # 画椭圆和画圆心
                    cv2.ellipse(frame, ell, (0, 0, 255), 2) 
                    cv2.circle(frame, (cen_x, cen_y), 2, (0, 0, 255), -1)
                    
        #----一帧椭圆检测结束-----#

        #-----处理圆心坐标点集-----#
        if len(point_list) == 0:
            print('No ellipse in sight!')
        else:
            #-----排序测试------#
            locate_point(point_list, lpoint_list) # 确定坐标点与靶标圆的对应关系
            for i in range(0, len(lpoint_list)):
                print(lpoint_list[i].x, lpoint_list[i].y, lpoint_list[i].n)

                # font = cv2.FONT_HERSHEY_SIMPLEX
                # cv2.putText(frame, str(lpoint_list[i].n), (lpoint_list[i].x, lpoint_list[i].y), font, 1, (0, 255, 0), 2,
                #            cv2.LINE_AA)
            #-----排序测试------#
            
            point_list = [] # 结束后置空
        #------处理圆心坐标结束----#

        cv2.imshow('ell', frame)

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
cv2.destroyAllWindows()
#retvel, rvev, tvec = cv2.solvePnP()
#
# if __name__ == '__main__':
#
#     main()

