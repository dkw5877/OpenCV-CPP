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

/* find and highlight the edges of objects in an image. This uses the same method as displaycontours
   however, it uses different threshold value (for removing a noise) and parameters (mode and method) in the findContours method */
Mat edgeDetection(Mat image) {

    int threshold_value = 0;
    int const max_BINARY_value = 2147483647;

    //convert image to grey scale
    cv::Mat src_gray = image;
    cv::Mat dst;
    dst = src_gray;
    cv::cvtColor(src_gray, dst, cv::COLOR_RGB2GRAY);

    //detect edges using canny
    cv::Mat canny_output;
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::RNG rng(12345);

    //set a threshold 
    cv::threshold( dst, dst, threshold_value, max_BINARY_value,cv::THRESH_OTSU );

    cv::Mat contourOutput = dst.clone();
    cv::findContours( contourOutput, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE );

    //Draw the contours
    cv::Mat contourImage(dst.size(), CV_8UC3, cv::Scalar(0,0,0));
    cv::Scalar colors[3];
    colors[0] = cv::Scalar(255, 0, 0);
    colors[1] = cv::Scalar(0, 255, 0);
    colors[2] = cv::Scalar(0, 0, 255);
    for (int idx = 0; idx < contours.size(); idx++) {
        cv::drawContours(contourImage, contours, idx, colors[idx % 3]);
    }

    return contourImage;
    
}

/* load a default image if one if not provided */
Mat shapeDetection(Mat image = Mat()) {

    if(image.empty()) {
        image = imread("/Developer/Practice Projects/OpenCV_CPP/OpenCV_CPP/FindingContours.png");
    }

    return displayContours(image);
}

/* we declare these variables outside of the the function so that they are not reset on each camera
    frame that we pass in. The imgLine Mat needs to be updated with each frame to show the tracking of
    the red object. The iLastX and iLastY values should on be less then zero when the method starts, 
    otherwise the line will not be drawn. Also as the author notes "If there are 2 or more objects 
    in the image, we cannot use this method". Only one red object will be tracked.
*/

Mat imgLines;
int iLastX = -1;
int iLastY = -1;

Mat colorTracker(Mat imgOriginal, Mat &imgThresholded) {

    namedWindow("Control", CV_WINDOW_AUTOSIZE); //create a window called "Control"

    int iLowH = 170;
    int iHighH = 179;

    int iLowS = 150;
    int iHighS = 255;

    int iLowV = 60;
    int iHighV = 255;

    //Create trackbars in "Control" window
    createTrackbar("LowH", "Control", &iLowH, 179); //Hue (0 - 179)
    createTrackbar("HighH", "Control", &iHighH, 179);

    createTrackbar("LowS", "Control", &iLowS, 255); //Saturation (0 - 255)
    createTrackbar("HighS", "Control", &iHighS, 255);

    createTrackbar("LowV", "Control", &iLowV, 255);//Value (0 - 255)
    createTrackbar("HighV", "Control", &iHighV, 255);

    Mat imgHSV;

    cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV

    inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image

    //morphological opening (removes small objects from the foreground)
    erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
    dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );

    //morphological closing (removes small holes from the foreground)
    dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
    erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );

    //Calculate the moments of the thresholded image
    Moments oMoments = moments(imgThresholded);

    double dM01 = oMoments.m01;
    double dM10 = oMoments.m10;
    double dArea = oMoments.m00;

    // if the area <= 10000, I consider that the there are no object in the image and it's because of the noise, the area is not zero
    if (dArea > 10000)
    {
        //calculate the position of the ball
        int posX = dM10 / dArea;
        int posY = dM01 / dArea;

        if (iLastX >= 0 && iLastY >= 0 && posX >= 0 && posY >= 0)
        {
            //Draw a red line from the previous point to the current point
            line(imgLines, Point(posX, posY), Point(iLastX, iLastY), Scalar(0,0,255), 2);
        }

        iLastX = posX;
        iLastY = posY;
    }

    imgOriginal = imgOriginal + imgLines;
    return imgOriginal;
}

//enum to switch between options
enum option { ORBDector, HSVPixel, Contours, Edges, ColorDetection };

int main(int argc, const char * argv[]) {

    std::cout << "Running application OpenCV version" << CV_VERSION << "\n";

    //set the desired processing option
    int option = Contours;

    VideoCapture cap(0);

    if(!cap.isOpened()) {
        cout << "Cannot open video capture";
    }

    Mat threshHoldImage;

    if(option == ColorDetection) {

        Mat tmpImg;
        cap.read(tmpImg); //capture a camera frame for sizing

        //create an blank image for line drawing
        imgLines = Mat::zeros(tmpImg.size(), CV_8UC3 );
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
                processedImage = shapeDetection(cameraFrame);
                break;
            case ColorDetection:
            {
                processedImage = colorTracker(cameraFrame, threshHoldImage);
                //imshow("Threshold", threshHoldImage); //show the threshold image 
            }
                break;
            case Edges:
                processedImage = edgeDetection(cameraFrame);
                break;
            default:
                break;
        }

        imshow("Webcam", processedImage);

        if (waitKey(10) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
        {
            cout << "esc key is pressed by user" << endl;
            break;
        }

    }

    return 0;
}

//#include <iostream>
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//
//using namespace cv;
//using namespace std;
//
//int main( int argc, char** argv )
//{
//    VideoCapture cap(0); //capture the video from webcam
//
//    if ( !cap.isOpened() )  // if not success, exit program
//    {
//        cout << "Cannot open the web cam" << endl;
//        return -1;
//    }
//
//    namedWindow("Control", CV_WINDOW_AUTOSIZE); //create a window called "Control"
//
//    int iLowH = 170;
//    int iHighH = 179;
//
//    int iLowS = 150;
//    int iHighS = 255;
//
//    int iLowV = 60;
//    int iHighV = 255;
//
//    //Create trackbars in "Control" window
//    createTrackbar("LowH", "Control", &iLowH, 179); //Hue (0 - 179)
//    createTrackbar("HighH", "Control", &iHighH, 179);
//
//    createTrackbar("LowS", "Control", &iLowS, 255); //Saturation (0 - 255)
//    createTrackbar("HighS", "Control", &iHighS, 255);
//
//    createTrackbar("LowV", "Control", &iLowV, 255);//Value (0 - 255)
//    createTrackbar("HighV", "Control", &iHighV, 255);
//
//    int iLastX = -1;
//    int iLastY = -1;
//
//    //Capture a temporary image from the camera
//    Mat imgTmp;
//    cap.read(imgTmp);
//
//    //Create a black image with the size as the camera output
//    Mat imgLines = Mat::zeros( imgTmp.size(), CV_8UC3 );
//
//
//    while (true)
//    {
//        Mat imgOriginal;
//
//        bool bSuccess = cap.read(imgOriginal); // read a new frame from video
//
//        if (!bSuccess) //if not success, break loop
//        {
//            cout << "Cannot read a frame from video stream" << endl;
//            break;
//        }
//
//        Mat imgHSV;
//
//        cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV
//
//        Mat imgThresholded;
//
//        inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image
//
//        //morphological opening (removes small objects from the foreground)
//        erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
//        dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
//
//        //morphological closing (removes small holes from the foreground)
//        dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
//        erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
//
//        //Calculate the moments of the thresholded image
//        Moments oMoments = moments(imgThresholded);
//
//        double dM01 = oMoments.m01;
//        double dM10 = oMoments.m10;
//        double dArea = oMoments.m00;
//
//        // if the area <= 10000, I consider that the there are no object in the image and it's because of the noise, the area is not zero
//        if (dArea > 10000)
//        {
//            //calculate the position of the ball
//            int posX = dM10 / dArea;
//            int posY = dM01 / dArea;
//
//            if (iLastX >= 0 && iLastY >= 0 && posX >= 0 && posY >= 0)
//            {
//                //Draw a red line from the previous point to the current point
//                line(imgLines, Point(posX, posY), Point(iLastX, iLastY), Scalar(0,0,255), 2);
//            }
//
//            iLastX = posX;
//            iLastY = posY;
//        }
//        
////        imshow("Thresholded Image", imgThresholded); //show the thresholded image
//        imshow("line img", imgLines);
//        
//        imgOriginal = imgOriginal + imgLines;
////        imshow("Original", imgOriginal); //show the original image
//
//        if (waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
//        {
//            cout << "esc key is pressed by user" << endl;
//            break; 
//        }
//    }
//    
//    return 0;
//}
