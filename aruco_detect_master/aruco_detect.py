import cv2 as cv
import cv2.aruco as aruco
import numpy as np
import copy
# IMG_WIDTH =  
camera_matrix = np.array(([693.2, 0, 666.8], # 内参矩阵
                          [0, 693.4, 347.7],
                          [0, 0, 1]), dtype=np.double)
dist_coefs = np.array([-0.050791, 0.217163, 0.0000878, -0.000388, -0.246122],
dtype=np.double) # k1 k2 p1 p2 k3

VIDEO_WIDTH, VIDEO_HEIGHT = 640, 480
SHOW_WIDTH = 550

def drawPolyLines(img, raw_point_list):
	point_list = [[elem[0], elem[1]] for elem in raw_point_list]
	pts = np.array(point_list, np.int32)
	pts = pts.reshape((-1, 1, 2))
	cv.polylines(img, [pts], True, (0, 255, 255))

def saveVideo(cap_save, num):
	fourcc = cv.VideoWriter_fourcc(*'XVID')
	out = cv.VideoWriter('./aurco_test'+str(num)+'.avi', fourcc, 20.0, (VIDEO_WIDTH, VIDEO_HEIGHT))
	while cap_save.isOpened():
		ret, frame = cap_save.read()
		if ret:
			out.write(frame)
			cv.imshow('frame', frame)
			if cv.waitKey(1) & 0xFF == ord('s'):
				print ('End record video!')
				break
		else:
			print ('ret is False...break out!')
			break
	out.release()

def detectMarkersOrigin(img_origin):
	frame = copy.deepcopy(img_origin)

	gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
	corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict)

	if ids is not None:
		id_show = [[ids[i][0], corners[i][0][0][0], corners[i][0][0][1]] for i in range(len(corners))]
        # print (len(ids), type(ids), ids)
		rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coefs)
		for i in range(rvec.shape[0]):
			aruco.drawAxis(frame, camera_matrix, dist_coefs, rvec[i, :, :], tvec[i, :, :], 0.03)
			aruco.drawDetectedMarkers(frame, corners, ids)
		for elem in id_show:
			cv.putText(frame, 'id='+str(elem[0]), (elem[1], elem[2]), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv.LINE_AA)
	else:
		cv.putText(frame, "No Aruco_Markers in sight!", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
	
	cv.namedWindow('Marker_Detect', cv.WINDOW_NORMAL)
	cv.resizeWindow('Marker_Detect', (SHOW_WIDTH, int(SHOW_WIDTH*480/640)))
	cv.moveWindow('Marker_Detect', 50, 50)
	cv.imshow('Marker_Detect', frame)

def detectMarkersMaster(img_origin):
	img = copy.deepcopy(img_origin)
	cv.namedWindow('Origin_Img', cv.WINDOW_NORMAL)
	cv.moveWindow('Origin_Img', 650, 50)
	cv.resizeWindow('Origin_Img', (SHOW_WIDTH, int(SHOW_WIDTH*480/640)))
	cv.imshow('Origin_Img', img)

	cv.namedWindow('Canny_Img', cv.WINDOW_NORMAL)
	cv.moveWindow('Canny_Img', 1250, 50)
	cv.resizeWindow('Canny_Img', (SHOW_WIDTH, int(SHOW_WIDTH*480/640)))

	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	edges = cv.Canny(gray, 100, 200)
	cv.imshow('Canny_Img', edges)

	drawing = np.zeros(img.shape[:], dtype=np.uint8)
	
	#TODO:
	lines_p = cv.HoughLinesP(edges, 0.5, np.pi / 180, 90, minLineLength=10, maxLineGap=15)
	if lines_p is not None:
		for line in lines_p:
			x1, y1, x2, y2 = line[0]
			cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3, lineType=cv.LINE_AA)
	# cv.imshow('Hough_p', img)

	#寻找Harris角点
	gray = np.float32(gray)
	dst = cv.cornerHarris(gray, 2, 3, 0.04)
	dst = cv.dilate(dst,None)
	img[dst > 0.01*dst.max()]=[0, 0, 255]
	cv.imshow('dst', img)
	# ret, dst = cv.threshold(dst,0.01*dst.max(),255,0)
	# dst = np.uint8(dst)
	# #找到重心
	# ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)

	# #定义迭代次数
	# criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
	# corners = cv.cornerSubPix(gray, np.float32(centroids),(5,5),(-1,-1),criteria)
	# #返回角点
	# #绘制
	# res = np.hstack((centroids,corners))
	# res = np.int0(res)
	# img[res[:,1],res[:,0]]=[0,0,255]
	# img[res[:,3],res[:,2]] = [0,255,0]

	# cv.imwrite('./subpixel5.png',img)



def main():
	cap, num = cv.VideoCapture(1), 1
	if not cap.isOpened():
		print ('Failed to open the camera...')
		return -1

	while cap.isOpened():
		ret, img = cap.read()
		
		detectMarkersOrigin(img)
		detectMarkersMaster(img)

		key = cv.waitKey(1) & 0xff
		if key == 27:
			print ("close window for keyboard break")
			break
		if key == ord('s'):
			print ('Start to record video...') 
			saveVideo(cap, num)
			num += 1

	cap.release()
	cv.destroyAllWindows()

if __name__ == "__main__":
	main()

