import numpy as np
import cv2


# 找到目标函数
def find_marker(image):
    # convert the image to grayscale, blur it, and detect edges
    # 将图像转换成灰度、模糊和检测边缘

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 35, 125)
    # find the contours in the edged image and keep the largest one;
    # 在边缘图像中找到轮廓并保持最大的轮廓
    # we'll assume that this is our piece of paper in the image
    # 我们假设这是我们在图像中的一张纸
    (_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # 求最大面积
    c = max(cnts, key=cv2.contourArea)

    # compute the bounding box of the of the paper region and return it
    # 计算纸张区域的边界框并返回它
    # cv2.minAreaRect() c代表点集，返回rect[0]是最小外接矩形中心点坐标，
    # rect[1][0]是width，rect[1][1]是height，rect[2]是角度
    return cv2.minAreaRect(c)


# 距离计算函数
def distance_to_camera(knownWidth, focalLength, perWidth):
    # compute and return the distance from the maker to the camera
    # 计算并返回从目标到相机的距离
    return (knownWidth * focalLength) / perWidth


# initialize the known distance from the camera to the object, which
# in this case is 24 inches
# 初始化已知距离从相机到对象，在这种情况下是24英寸
KNOWN_DISTANCE = 24.0

# initialize the known object width, which in this case, the piece of
# paper is 11 inches wide
# 初始化已知的物体宽度，在这种情况下，纸是11英寸宽。
# A4纸的长和宽(单位:inches)
KNOWN_WIDTH = 11.69
KNOWN_HEIGHT = 8.27

# initialize the list of images that we'll be using
# 初始化我们将要使用的图像列表
IMAGE_PATHS = ["D:\c1.JPG", "D:\c2.JPG"]

# load the furst image that contains an object that is KNOWN TO BE 2 feet
# from our camera, then find the paper marker in the image, and initialize
# the focal length
# 加载包含一个距离我们相机2英尺的物体的第一张图像，然后找到图像中的纸张标记，并初始化焦距
# 读入第一张图，通过已知距离计算相机焦距

image = cv2.imread("E:\\lena.jpg")  # 应使用摄像头拍的图

marker = find_marker(image)
focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH  # D’ = (W x F) / P

# 通过摄像头标定获取的像素焦距
# focalLength = 811.82
print('focalLength = ', focalLength)

# 打开摄像头
camera = cv2.VideoCapture(0)

while camera.isOpened():
    # get a frame
    (grabbed, frame) = camera.read()
    marker = find_marker(frame)
    if marker == 0:
        print(marker)
    continue
    inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])

    # draw a bounding box around the image and display it
    # 在图像周围绘制一个边界框并显示它
    box = cv2.boxPoints(marker)
    box = np.int0(box)
    cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)

    # inches 转换为 cm
    cv2.putText(frame, "%.2fcm" % (inches * 30.48 / 12),
                (frame.shape[1] - 200, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                2.0, (0, 255, 0), 3)

    # show a frame
    cv2.imshow("capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()