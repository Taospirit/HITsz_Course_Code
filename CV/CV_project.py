import cv2
import numpy as np
import math

# 守夜人宣言：
# 长夜将至，我从今开始守望，至死方休。
# 我将不娶妻，不封地，不生子。
# 我将不戴王冠，不争荣宠。
# 我将尽忠职守，生死于斯。
# 我是黑暗中的利剑，长城上的守卫。
# 抵御寒冷的烈焰，破晓时分的光线，唤醒眠者的号角，守护王国的坚盾。
# 我将生命与荣耀献给守夜人，今夜如此，夜夜皆然。

WIDETH = 1280
HEIGHT = 720

error_alpha = 0.5
radius_alpha = 0.5
max_radius = 120
min_radius = 10

method_num = 0
point_list = []
lpoint_list = []
choose_point_list = []

camera_matrix = np.array(([693.2, 0, 666.8], # 内参矩阵
                          [0, 693.4, 347.7],
                          [0, 0, 1]), dtype=np.double)
dist_coefs = np.array([-0.050791, 0.217163, 0.0000878, -0.000388, -0.246122], dtype=np.double) # k1 k2 p1 p2 k3
object_3D_points = np.array(([23, 0, 0], [0, 23, 0], [0, 58, 0], [23, 81, 0]), dtype=np.double)   # 世界坐标系上的坐标,A4纸实际测量
image_2D_points = []   # 图像坐标系上的坐标，待获取

#  坐标构建如下：
#  ---2--3--------->Y
#  1        4
#  |
#  |
#  V
#  X

class Point:
    def __init__(self, x_param, y_param, num):
        self.x = x_param
        self.y = y_param
        self.n = num

#------筛选椭圆函数，待修正------#
def checkEllipse_simple(img, cen_x, cen_y, a_double, b_double): # 函数功能：初步识别检测出靶标的椭圆（不稳定）
    # 近似化，img像素数组只考虑整数位置
    x = int(np.around(cen_x))
    y = int(np.around(cen_y))
    a = int(np.around(a_double / 2))
    b = int(np.around(b_double / 2))
    # 越界pass
    if x + a+3 > WIDETH or y + a+3 > HEIGHT:
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
    for m in range(a+1, a+3):
        if img[y, x + m] == 255 or img[y, x - m] == 255:
            return False
        if img[y - m, x] == 255 or img[y + m,x] == 255:
            return False
    return True

# -------对符合的椭圆重排序-------#
def locatePoint(p_list, lp_list, radius): # 函数功能：稳定完整的实现靶标定位
    temp = []
    for i in range(0, len(p_list)):  # 复制p_list给temp,为了保留p_list
        addPoint(p_list, temp, i, i+1)
    #                   6 3
    #   标准位置定义:   7     8
    #                   5 2
    #                   4 1

    #-----筛选出中间6点位置-----#
    Error = radius * error_alpha  # 用距离圆心占半径的百分比评估误差
    num = 0
    for i in range(0, len(p_list)):
        for j in range(i+1, len(p_list)):
            med_x = (p_list[i].x + p_list[j].x) / 2
            med_y = (p_list[i].y + p_list[j].y) / 2

            for m in range(0, len(p_list)):
                if m == i or m == j:
                    continue
                # 用距离和半径的百分比做误差评估
                error = math.sqrt(pow(med_x - p_list[m].x, 2) + pow(med_y - p_list[m].y, 2))
                if error < Error:
                    addPoint(p_list, lp_list, i, num+1)     #  1  4
                    addPoint(p_list, lp_list, m, num+2)     #  2  5
                    addPoint(p_list, lp_list, j, num+3)     #  3  6
                    num += 3
                    # 以上一定要找出需要的6点，否则算作无法进行进一步的计算。
                    temp[i].n = -1  # -1说明已经选定
                    temp[m].n = -1
                    temp[j].n = -1
    #-----筛选6点完毕------#

    #-----确定7\8点------#
    if len(lp_list) == 6:
        for p in range(0, len(p_list)):
            if temp[p].n == -1:
                continue
            if distance(p_list, lp_list, p, 2) > 8*radius : # 根据距离进一步筛选
                continue
            else:
                num += 1
                addPoint(p_list, lp_list, p, num)   # 7 和 8 添加进数组

    if len(lp_list) < 8:
        pass
    else:
        cen_78_y = (lp_list[6].y + lp_list[7].y)/2   # 7 和 8 两点的中点的纵坐标
        cen_25_y = (lp_list[1].y + lp_list[4].y)/2   # 2 和 5 两点的中点的纵坐标

        if cen_78_y < cen_25_y and lp_list[6].x > lp_list[7].x: # 图像是正的
            swapPoint(lp_list, 6, 7)    # 7\8点交换
        if cen_78_y > cen_25_y and lp_list[6].x < lp_list[7].x: # 图像是倒的
            swapPoint(lp_list, 6, 7)    # 7\8点交换

        # 至此，lpoint_list已经实现了对p_list中元素的进一步筛选和重排序

        #-----定位方案待选------#
        # 把点7定位为1
        addPoint(lp_list, choose_point_list, 6, 1)

        #----- 2/3定位方案一 ------#
        #           2  3
        #         1      4
        #           0  0
        #           0  0
        if method_num == 1:
            if cen_78_y < cen_25_y: # 图像是正的
                if distance(lp_list, lp_list, 2, 6) > distance(lp_list, lp_list, 5, 6): #待选点是点3和点6
                    swapPoint(lp_list, 2, 5)
                addPoint(lp_list, choose_point_list, 2, 2)
                addPoint(lp_list, choose_point_list, 5, 3)
            if cen_78_y > cen_25_y: # 图像是倒的
                if distance(lp_list, lp_list, 0, 6) > distance(lp_list, lp_list, 3, 6): #待选点是点1和点4
                    swapPoint(lp_list, 0, 3)
                addPoint(lp_list, choose_point_list, 0, 2)
                addPoint(lp_list, choose_point_list, 3, 3)

        #------ 2/3定位方案二 ------#
        #用7\8点确定最中间2点，最终决定选取中间4点作为参考点，如下图：
        #          0  0
        #        1      4
        #          2  3
        #          0  0
        #-----问题:点距离太近,可能导致解算误差较大-------#
        if method_num == 2:
            if distance(lpoint_list, lpoint_list, 1, 6) > distance(lpoint_list, lpoint_list, 4, 6): #待选点是点2和点5
                swapPoint(lpoint_list, 1, 4)
            addPoint(lpoint_list, choose_point_list, 1, 2)
            addPoint(lpoint_list, choose_point_list, 4, 3)

        #----- 2/3定位方案三 ----#
        #           0  0
        #         1      4
        #           0  0
        #           2  3
        #----- 问题:2、3点的位置过低，减小测距有效区间
        if method_num == 3:
            if cen_78_y < cen_25_y: # 图像是正的
                if distance(lpoint_list, lpoint_list, 0, 6) > distance(lpoint_list, lpoint_list, 3, 6): #带选点是点1和点4
                    swapPoint(lpoint_list, 0, 3)
                addPoint(lpoint_list, choose_point_list, 0, 2)
                addPoint(lpoint_list, choose_point_list, 3, 3)
            if cen_78_y > cen_25_y: # 图像是倒的
                if distance(lpoint_list, lpoint_list, 2, 6) > distance(lpoint_list, lpoint_list, 5, 6): #待选点是点3和点6
                    swapPoint(lpoint_list, 2, 5)
                addPoint(lpoint_list, choose_point_list, 2, 2)
                addPoint(lpoint_list, choose_point_list, 5, 3)

        # 点8为定位为4
        addPoint(lp_list, choose_point_list, 7, 4)
        #------排序完毕------#

def distance(list_1, list_2, i, j): # list_1第i个索引点和list_2第j个索引点的距离
    if i > len(list_1) or j > len(list_2):
        return -1
    dis = math.sqrt(pow(list_1[i].x - list_2[j].x, 2) + pow(list_1[i].y - list_2[j].y, 2))
    return dis

def swapPoint(list, i, j): # 交换list中第i个索引和第j个索引数据的位置、索引
    list[i].n = j+1
    list[j].n = i+1
    list[i], list[j] = list[j], list[i]

def addPoint(src_list, new_list, i, n): #将src_list中的第i个索引的数据添加进new_list，且num为n
    new_list.append(Point(src_list[i].x, src_list[i].y, n))


# -----计算欧拉角------#
#代码参考：https://www.cnblogs.com/subic/p/8296794.html
#原理参考：http://www.cnblogs.com/singlex/p/RotateMatrix2Euler.html
def RotateByZ(Cx, Cy, thetaZ):
    rz = thetaZ*math.pi/180.0
    outX = math.cos(rz)*Cx - math.sin(rz)*Cy
    outY = math.sin(rz)*Cx + math.cos(rz)*Cy
    return outX, outY

def RotateByY(Cx, Cz, thetaY):
    ry = thetaY*math.pi/180.0
    outZ = math.cos(ry)*Cz - math.sin(ry)*Cx
    outX = math.sin(ry)*Cz + math.cos(ry)*Cx
    return outX, outZ

def RotateByX(Cy, Cz, thetaX):
    rx = thetaX*math.pi/180.0
    outY = math.cos(rx)*Cy - math.sin(rx)*Cz
    outZ = math.sin(rx)*Cy + math.cos(rx)*Cz
    return outY, outZ


cap = cv2.VideoCapture(1)
if (cap.isOpened() == False):
    print("Failed to open the camera...")
else:
    method_num = 1  # 使用第一种定位方法
    ret = cap.set(3, WIDETH) # 设置显示尺寸 1280*720
    ret = cap.set(4, HEIGHT)

    while(True):
        ret, frame = cap.read()
        ret, img = cap.read()

        # 每次循环清空列表
        point_list = []
        lpoint_list = []
        choose_point_list = []
        image_2D_points = []

        #-----一帧基础图形处理-----#
        f_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        f_bulr = cv2.GaussianBlur(f_gray, (5, 5), 0)

        ret,f_thresh = cv2.threshold(f_bulr, 127, 255, cv2.THRESH_BINARY_INV)  # 阈值127，变成255

        f_can = cv2.Canny(f_thresh, 50, 150, 3)
        #-----结束-----#

        #-----一帧椭圆检测-------#
        image, contours, hierarchy = cv2.findContours(f_can, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        count = 0 # 用来计数
        sum = 0  # 用来统计半径
        for cnt in contours:
            if len(cnt) > 10: # 点数超过5才能拟合椭圆
                ell = cv2.fitEllipse(cnt)

                b_double = ell[1][0] #拟合的矩阵宽，即短半轴2倍
                a_double = ell[1][1] #拟合的矩阵长，即长半轴2倍

                #-----椭圆基本筛选-----#
                if a_double > max_radius or b_double > max_radius:
                    continue
                if a_double < min_radius or b_double < min_radius:
                    continue
                if a_double < b_double * radius_alpha or b_double < a_double * radius_alpha:
                    continue
                # -----结束-----#

                # 开始对选定的进行处理
                cen_x = ell[0][0]
                cen_y = ell[0][1]

                theta = ell[2] # 旋转角度,暂时没用到

                if  checkEllipse_simple(f_thresh, cen_x, cen_y, a_double, b_double):
                    count += 1
                    sum += b_double/2 # sum是所有短半轴的集合
                    point_list.append(Point(cen_x, cen_y, count))
                    # font = cv2.FONT_HERSHEY_SIMPLEX # 数字标记
                    # cv2.putText(frame, str(count), (cen_x, cen_y), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

                    cv2.ellipse(frame, ell, (0, 0, 255), 2)
                    cv2.circle(frame, (int(cen_x), int(cen_y)), 2, (0, 0, 255), -1)
        #----一帧椭圆检测结束-----#

        #-----处理圆心坐标点集-----#
        if len(point_list) < 8:
            print('No enough ellipse in sight!')
        else:
            #-----重定位和排序------#
            locatePoint(point_list, lpoint_list, sum/count)

            font = cv2.FONT_HERSHEY_SIMPLEX
            for i in range(0, len(lpoint_list)):
                cv2.putText(frame, str(lpoint_list[i].n), (int(lpoint_list[i].x), int(lpoint_list[i].y)), font, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)

            # for i in range(0, len(point_list)):
            #     cv2.putText(img, str(point_list[i].n), (point_list[i].x, point_list[i].y), font, 1, (0, 255, 0), 2,
            #                 cv2.LINE_AA)

            for i in range(0, len(choose_point_list)):
                cv2.putText(img, str(choose_point_list[i].n), (int(choose_point_list[i].x), int(choose_point_list[i].y)), font, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)
                #print(i, choose_point_list[i].x, choose_point_list[i].y, choose_point_list[i].n)
            #-------位置确定完毕-------#

            #-------位姿解算solvePNP-------#
            if len(choose_point_list) == 4: # 运动太快会导致画面模糊，无法获取椭圆位置，跳过这种情况
                for i in range(0, len(choose_point_list)):
                    image_2D_points.append((choose_point_list[i].x, choose_point_list[i].y))

                image_2D_points = np.array(image_2D_points, dtype=np.double) # list转array
                # print("数组测试：\n",image_2D_points)

                found, rvec, tvec = cv2.solvePnP(object_3D_points, image_2D_points, camera_matrix, dist_coefs)
                # print(found)
                # print("旋转向量：\n", rvec)
                # print("平移向量：\n", tvec)

                rotM = cv2.Rodrigues(rvec)[0]   # 旋转向量转换成旋转矩阵
                # print("旋转矩阵", rotM)

                #camera_postion = -np.matrix(rotM).T * np.matrix(tvec)
                # print(camera_postion.T)

                # -----计算欧拉角----#
                thetaZ = math.atan2(rotM[1, 0], rotM[0, 0]) * 180.0 / math.pi
                thetaY = math.atan2(-1.0 * rotM[2, 0], math.sqrt(rotM[2, 1] ** 2 + rotM[2, 2] ** 2)) * 180.0 / math.pi
                thetaX = math.atan2(rotM[2, 1], rotM[2, 2]) * 180.0 / math.pi
                # 相机坐标系下值
                x = tvec[0]
                y = tvec[1]
                z = tvec[2]

                (x, y) = RotateByZ(x, y, -1.0 * thetaZ)
                (x, z) = RotateByY(x, z, -1.0 * thetaY)
                (y, z) = RotateByX(y, z, -1.0 * thetaX)
                Cx = x * -1
                Cy = y * -1
                Cz = z * -1
                # 输出相机位置
                print("相机位置：", Cx, Cy, Cz)
                # 输出相机旋转角
                print("相机旋转角", thetaX, thetaY, thetaZ)

                # 对第五个点进行验证，此处选取靶标中最上面2点的中点作为验证点
                extrinsics_matrix = np.concatenate((rotM, tvec), axis=1) # 矩阵拼接，旋转矩阵R和平移矩阵t组成齐次矩阵
                projection_matrix = np.dot(camera_matrix, extrinsics_matrix)     # np.dot(a,b):a*b矩阵相乘
                pixel = np.dot(projection_matrix, np.array([0, 40.5, 0, 1], dtype=np.double))
                pixel_unit = pixel / pixel[2]   # 归一化
                cv2.circle(img, (int(np.around(pixel_unit[0])), int(np.around(pixel_unit[1]))), 2, (0, 0, 255), -1)

        cv2.imshow('ell', frame)
        cv2.imshow('locate', img)

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

cv2.destroyAllWindows()
#retvel, rvev, tvec = cv2.solvePnP()
#
# if __name__ == '__main__':
#
#     main()

