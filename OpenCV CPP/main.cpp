//
//  main.cpp
//  OpenCV CPP
//
//  Created by user on 3/16/17.
//  Copyright © 2017 someCompanyNameHere. All rights reserved.
//

#include <iostream>
#include "opencv2/features2d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;

Mat ORBDetector(Mat image) {

    //create vector to store points
    vector<KeyPoint> keypoints;

    //find the keypoints with ORB detector
    Ptr<FeatureDetector> orb = ORB::create();
    orb->detect(image, keypoints);

    //Draw keypoints on Mat image
    Mat img_keypoints;
    drawKeypoints( image, keypoints, img_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

    return img_keypoints;

}

/* convert a Mat image to HSV and get only the pixels in scalar range */
Mat HSVPixelScalarRange(Mat image, Scalar low, Scalar high) {
    Mat HSVImage;
    Mat processedImage;
    cvtColor(image, HSVImage, CV_RGB2HSV); //convert image to HSV
    inRange(HSVImage, low, high, processedImage); //get only pixels in scalar range
    int count = countNonZero(processedImage); //count non-zero pixels
    cout << count;
    return processedImage;
}

enum option { ORBDector, HSVPixel };

int main(int argc, const char * argv[]) {

    std::cout << "Running application OpenCV version" << CV_VERSION << "\n";

    //set the desired processing option
    int option = HSVPixel;

    VideoCapture cap(0);

    if(!cap.isOpened()) {
        cout << "Cannot open video capture";
    }

    while(true){

        Mat cameraFrame;
        Mat processedImage;
        cap.read(cameraFrame);

        switch (option) {
            case ORBDector:
                processedImage = ORBDetector(cameraFrame);
                break;
            case HSVPixel:
                processedImage = HSVPixelScalarRange(cameraFrame, Scalar(0,0,0), Scalar(75,100,200));
                break;
            default:
                break;
        }

        imshow("Webcam", processedImage);
        waitKey(10);
    }

    return 0;
}
