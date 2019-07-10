'''
@Description: Copyright Reserved
@Author: lintao
@Github: https://github.com/Taospirit
@Date: 2019-07-10 11:38:35
@LastEditors: lintao
@LastEditTime: 2019-07-10 21:53:09
'''
import os
import argparse
import numpy as np
import cv2 as cv
import cv2.aruco as aruco
import copy

class ArucoDataCollection:
    def __init__(self):
        self.camera_matrix = np.array(([693.2, 0, 666.8], # 内参矩阵
                          [0, 693.4, 347.7],
                          [0, 0, 1]), dtype=np.double)
        self.dist_coefs = np.array([-0.050791, 0.217163, 0.0000878, -0.000388, -0.246122],
                            dtype=np.double) # k1 k2 p1 p2 k3
        self.img_width, self.img_height = self.setImgSize()

        self.cap = cv.VideoCapture(0)
        self.ret = self.cap.set(3, self.img_width)
        self.ret = self.cap.set(4, self.img_height)

        self.outpath_data = "./data"
        self.outpath_img_origin = "./data/img_orgin"
        self.outpath_img_detect = "./data/img_detect"
        self.outpath_data_file = os.path.join(self.outpath_data, "data.txt")
        self.save_origin, self.save_detect = 's', 'p'

        self.data_resumed = False
        self.img_num_get = False

        self.img_origin_num_get = False
        self.img_detect_num_get = False
        
        self.img_num = 0
        self.marker_size = 0.053 # meter

        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--resume", help="delete dir & resume", action="store_true")
        # self.parser.add_argument("--resume", help="deleta dir & resume", action="store_true")


    def setImgSize(self, width=640, height=480):
        return width, height

    def start(self):
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.namedWindow('frame', cv.WINDOW_AUTOSIZE)
        cv.moveWindow('frame', 700, 300)
        
        args = self.parser.parse_args()
        if args.resume:
            self.clearPath(self.outpath_img_origin) # delete data & resume
            self.clearPath(self.outpath_img_detect)
            os.remove(self.outpath_data_file)
            self.data_resumed = True
            
        img_num = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            temp = copy.deepcopy(frame)
            #cv.imshow("frame", frame)
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
            
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict)
            corners = np.array(corners)
            ids_array = np.array(ids)

            if ids is not None:
                print ("*************")
                print ("corners is {}, shape is {}, len{}".format(corners, corners.shape, len(corners)))
                print ("ids is {}, shape is {}, len{}".format(ids_array, ids_array.shape, len(ids)))
                id_show = [[ids[i][0], corners[i][0][0][0], corners[i][0][0][1]] for i in range(len(corners))]

                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, self.marker_size, self.camera_matrix, self.dist_coefs)

                print ("rvec is {}, tvec is {}, shape is {}, {}".format(rvec, tvec, rvec.shape, tvec.shape))

                for i in range(rvec.shape[0]):
                    aruco.drawAxis(frame, self.camera_matrix, self.dist_coefs, rvec[i, :, :], tvec[i, :, :], 0.03)
                    aruco.drawDetectedMarkers(frame, corners, ids)

                for elem in id_show:
                    cv.putText(frame, 'id='+str(elem[0]), (elem[1], elem[2]), font, 0.8, (0, 0, 255), 2, cv.LINE_AA)
            else:
                cv.putText(frame, "No Ids in sight!", (50, 100), font, 1, (0, 255,0), 2, cv.LINE_AA)  

            key = cv.waitKey(1)&0xff
            if key == 27:
                self.release()

            if key == ord(self.save_origin): # save origin image
                if not os.path.exists(self.outpath_img_origin):
                    os.makedirs(self.outpath_img_origin)

                if not os.path.exists(self.outpath_img_detect):
                    os.makedirs(self.outpath_img_detect)

                if not self.img_origin_num_get:
                    img_num = self.getImgNum(self.outpath_img_origin)
                    self.img_origin_num_get = True

                img_num += 1
                img_name = str(img_num) + ".png"
                
                cv.imwrite(os.path.join(self.outpath_img_origin, img_name), temp)
                cv.imwrite(os.path.join(self.outpath_img_detect, img_name), frame)

                self.saveData(img_num, ids, corners, tvec, rvec)
                print ("{} saved in {}".format(img_name, self.outpath_img_origin))
                #TODO: add data for detected image

            if key == ord(self.save_detect): # save image to detect
                if not os.path.exists(self.outpath_img_detect):
                    os.makedirs(self.outpath_img_detect)

                # if not self.img_detect_num_get:
                #     img_num = self.getImgNum(self.outpath_img_detect)
                #     self.img_detect_num_get = True
                    
                # img_num += 1
                # img_name = str(img_num) + ".png"
                cv.imwrite(os.path.join(self.outpath_img_detect, img_name), frame)
                print ("{} saved in {}".format(img_name, self.outpath_img_detect))

            cv.imshow("frame", frame)

    def saveData(self, img_num, ids, corners, tvec, rvec):
        # if not os.path.exists(self.outpath_data_file):
        f = open(self.outpath_data_file, 'a')
        if ids is not None:
            for i in range(len(ids)):
                data_list = []
                data_list.append(img_num)
                data_list.append(ids[i][0]) # id

                for n in range(0, 4):
                    data_list.append(corners[i][0][n][0]) # corner
                    data_list.append(corners[i][0][n][1])

                data_list.append(tvec[i][0][0]) # tvec
                data_list.append(tvec[i][0][1])
                data_list.append(tvec[i][0][2])

                data_list.append(rvec[i][0][0]) # rvec
                data_list.append(rvec[i][0][1])
                data_list.append(rvec[i][0][2])
                
                data_str = ""
                for item in data_list:
                    data_str += str(item)
                    data_str += " "
                f.write(data_str + "\n")

        f.close()

    def clearPath(self, path):
        try:
            for root, dirs, files in os.walk(path, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
        except OSError as e:
            print (e)

    def getImgNum(self, path):
        for _, __, f in os.walk(path):
            # temp = []
            # for item in f:
            #     temp.append(int(item[:len(item) - 4]))
            # return len(temp)
            # self.img_num = len(f)
            # self.img_num_get = True
            # return self.img_num
            return len(f)

    def release(self):
        self.cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    ArucoDataCollection().start()
    # try:
    #     ArucoDataCollection().start()
    # except SyntaxError, NameError as e:sss
    #     print (e)
