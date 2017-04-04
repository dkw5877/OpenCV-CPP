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


//still cant get dlib to work nicely
#include "dlib/algs.h"
#include "dlib/image_io.h"


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

/* find and highlight the contours of shapes in an image */
Mat displayContours(Mat image) {

    int thresh = 100;
    RNG rng(12345);

    Mat src_gray;
    Mat canny_output;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    /// Convert image to gray and blur it
    cvtColor( image, src_gray, CV_BGR2GRAY );
    blur( src_gray, src_gray, Size(3,3) );

    /// Detect edges using canny
    Canny( src_gray, canny_output, thresh, thresh*2, 3 );

    /// Find contours
    findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

    /// Draw contours
    Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
    for( int i = 0; i< contours.size(); i++ ) {
        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
    }

    return drawing;
}

/* load a default image if one if not provided */
Mat shapeDetection() {
    Mat img = imread("/Developer/Practice Projects/OpenCV_CPP/OpenCV_CPP/FindingContours.png");
    return displayContours(img);
}

//enum to switch between options
enum option { ORBDector, HSVPixel, Contours };

int main(int argc, const char * argv[]) {

    std::cout << "Running application OpenCV version" << CV_VERSION << "\n";

    //set the desired processing option
    int option = ORBDector;

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
            case Contours:
                processedImage = shapeDetection();
                break;
            default:
                break;
        }

        imshow("Webcam", processedImage);
        waitKey(10);
    }

    return 0;
}
