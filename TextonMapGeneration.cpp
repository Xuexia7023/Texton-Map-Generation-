#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <ctime>

#include "../include/filters.hpp"
using namespace std;
using namespace cv;
/*
 * Rotate an image
 
void rotate(Mat &src, double angle, Mat &dst) {
    int len = max(src.cols, src.rows);
    Point2f pt(len/2., len/2.);
    Mat r = getRotationMatrix2D(pt, angle, 1.0);

    warpAffine(src, dst, r, Size(len, len));
}
*/
void magnitude(int rot[][2]) {
    return;
}
int main(int argc, char* argv[]) {
    int k = atoi(argv[1]);
    int numTrainingImages = atoi(argv[2]);
    Textons textonMap(k);
    int rot[2][2] = {1, 2, 3, 4};



/*
    Mat img(3,3, CV_32F);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            img.at<float>(i,j) = (i*3-j);
        }
    }
     for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++){
            cout << (float)img.at<float>(i,j) <<", ";
        }
        cout << endl;
    }
   Mat imgSobel;
    Sobel(img, imgSobel, -1, 1, 0, 3);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++){
            cout << (float)imgSobel.at<float>(i,j) <<", ";
        }
        cout << endl;
    }
return 0;
}*/



    textonMap.makeRFSFilters();
    int iter = 1;
    while (iter <= numTrainingImages) {
        char fname[100];
        sprintf(fname, "example_roads/img%d.jpg", iter);
        Mat input_image_ = imread(fname);
        resize(input_image_, input_image_, Size(0,0), 0.125,0.125);
        //resize(input_image_, input_image_, Size(0,0), 0.00625,0.00625);
        clock_t start1 = clock();
        textonMap.createFilterResponses(input_image_, 1);
        clock_t end1 = clock();
     //   cout << (end1 - start1)/(double)CLOCKS_PER_SEC << endl;
        iter++;
    }
    
    //cout << "before Kmeans" << endl;
    textonMap.computeKmeans();
    //cout << "after Kmeans" << endl;

    Mat test_image_ = imread("street_image.jpg");
    resize(test_image_, test_image_, Size(0,0), 0.125, 0.125);
    //resize(test_image_, test_image_, Size(0,0), 0.00625,0.00625);
    textonMap.createFilterResponses(test_image_, 0);
    Mat textonMap1 = textonMap.generateTextonMap(test_image_);

    
//    Mat test_image2_ = imread("street_image.jpg");
    Mat test_image2_;
    flip(test_image_, test_image2_, -1);
//    resize(test_image2_, test_image2_, Size(0,0), 0.00625, 0.00625);
    textonMap.createFilterResponses(test_image2_, 0);
    Mat textonMap2 = textonMap.generateTextonMap(test_image2_);
//    Mat difference  = textonMap1 - textonMap2;
//    rotate(textonMap2, 180, textonMap2);
    flip(textonMap2, textonMap2, -1);
    imshow("TM2", textonMap2);
    waitKey();
    imshow("TM1", textonMap1);
    waitKey();
   

    ofstream tm;
    tm.open("tm.txt");

    tm << "textonMap1: " << endl;
    for (int r = 0; r < textonMap1.rows; r++) {
        for (int c = 0; c < textonMap1.cols; c++) {
            tm << (int)textonMap1.at<uchar>(r,c) << ", " ;
        }
        tm << endl;
    }
    tm << "textonMap2: " << endl;
    for (int r = 0; r < textonMap1.rows; r++) {
        for (int c = 0; c < textonMap1.cols; c++) {
            tm << (int)textonMap2.at<uchar>(r,c) << ", " ;
        }
        tm << endl;
    }

//    tm.close();


    tm << "difference Map: " << endl;
    Mat diff(textonMap1.rows, textonMap1.cols, CV_8UC1);
    for (int r = 0; r < textonMap1.rows; r++) {
        for (int c = 0; c < textonMap1.cols; c++) {
            diff.at<uchar>(r,c) = textonMap1.at<uchar>(r,c) - textonMap2.at<uchar>(r,c);
            tm << (int)diff.at<uchar>(r,c) << ", ";
        }
        tm << endl;
    }
  tm.close();  
  imshow("diff", diff);
  waitKey();

/*
    
    Mat exp1 = imread("tm1.jpg");
    Mat exp2_unflipped = imread("tm2.jpg");
    Mat exp2;
    flip(exp2_unflipped, exp2, -1);
    imshow("exp1", exp1);
    waitKey();
    imshow("exp2", exp2);
    waitKey();
    Mat difference;
    ofstream diff, tm1, tm2;
    diff.open("diff.txt");
    tm1.open("tm1.txt");
    tm2.open("tm2.txt");
    difference.create(exp1.rows, exp1.cols, CV_8UC1);
    for (int r = 0; r < exp2.rows; r++) {
        for (int c = 0; c < exp2.cols; c++) {
            difference.at<uchar>(r,c) = (int)((int)exp1.at<uchar>(r,c) - (int)exp2.at<uchar>(r,c));
            if ((int)difference.at<uchar>(r,c)!=0)
            diff <<(int) exp1.at<uchar>(r,c) <<", " << (int)exp2.at<uchar>(r,c) << ", " <<  (int)difference.at<uchar>(r,c) << endl;
            //diff  <<  (int)difference.at<uchar>(r,c) ;
            tm1 << (int)exp1.at<uchar>(r,c) << ", ";
            tm2 << (int)exp2.at<uchar>(r,c) << ", ";
        }
        tm1 << endl;
        tm2 << endl;
    } 

    imwrite("diff.jpg", difference);
    imshow("difference", difference);
    waitKey();
*/
    //Mat F[36];
/*
    Mat input_image_ = imread("street_image.jpg");
    //resize(input_image_, input_image_, Size(0,0), 0.0625, 0.0625);
    resize(input_image_, input_image_, Size(0,0), 0.5, 0.5);
    textonMap.makeRFSFilters();
    textonMap.createFilterResponses(input_image_, 1);
    textonMap.computeKmeans();

    textonMap.createFilterResponses(input_image_, 0);
    Mat textonMap1 = textonMap.generateTextonMap(input_image_);
    imshow("textonmap", textonMap1);
    waitKey();
*/
    return 0;
}
