#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <vector>
using namespace cv;
using namespace std;

cv::Mat cameraMatrix = (Mat_<double>(3,3)<<486.4204611770197, 0, 343.5190607462192, 0, 486.9108422121211, 240.4400796678946, 0, 0, 1);
cv::Mat distCoeffs = (Mat_<double>(1,5)<<0.01182979280903908, 0.0127936338324327, 0.00311748652195563, 0.002833341560742986, 0);
std::vector<Vec3d > rvecs, tvecs;

int dectectMarkers() {
    cv::VideoCapture cap(0); //如果用笔记本改成１

    cv::Mat frame;
    vector<int> markerIds;
    vector<vector<cv::Point2f> > markerCorners;//, rejectedCandidates;
    Ptr<aruco::Dictionary> dic = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);

    while (cap.isOpened())
    {
        cap >> frame;
        // detect Markers in inputImage, search in the dic. save the corners and ids in last two parameters.
        aruco::detectMarkers(frame, dic, markerCorners, markerIds);

        if (markerIds.empty())
            cout << "No marker in sight..." << endl;

        else
        {   // draw detected Markers in the inputImage
            aruco::drawDetectedMarkers(frame, markerCorners, markerIds);
            aruco::estimatePoseSingleMarkers(markerCorners, 0.05, cameraMatrix, distCoeffs, rvecs, tvecs);
            for (int i = 0; i < markerIds.size(); ++i)
                aruco::drawAxis(frame, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.05);
        }
        imshow("test", frame);

        if(waitKey(1) == 27)
            break;
    }
    destroyAllWindows();
    return 0;
}

int main()
{
    dectectMarkers();
    return 0;
}
