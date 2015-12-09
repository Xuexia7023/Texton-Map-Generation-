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
    if (argc != 4) {
        cout << "error: 2 arguments required" << endl;
        cout << "Enter 1 for computing kMeans or 0 for texton Map, Enter number of centers for K means, the number of training images to be used " << endl; 
        return 0;
    }
    int k = atoi(argv[2]);
    int numTrainingImages = atoi(argv[3]);
    Textons textonMap(k);
    int rot[2][2] = {1, 2, 3, 4};

    int flagKmeans = atoi(argv[1]);

textonMap.makeRFSFilters();
if (flagKmeans == 1) {

    int iter = 1;
    while (iter <= numTrainingImages) {
        char fname[100];
        sprintf(fname, "example_roads/img%d.jpg", iter);
        cout << fname << endl;
        Mat input_image_ = imread(fname);

        cvtColor(input_image_, input_image_, CV_BGR2GRAY);
//        resize(input_image_, input_image_, Size(0,0), 0.125,0.125);
//        resize(input_image_, input_image_, Size(0,0), 0.5,0.5);
//        resize(input_image_, input_image_, Size(0,0), 0.00625,0.00625);
        clock_t start1 = clock();
        textonMap.createFilterResponses(input_image_, 1);

        cout << "Creating Filter Responses for train image #" << iter << endl;
        clock_t end1 = clock();
     //   cout << (end1 - start1)/(double)CLOCKS_PER_SEC << endl;
        iter++;
    }

    
    textonMap.computeKmeans();
}
else {
        string s("kmeans.txt");
        textonMap.KmeansCentersReadFromFile(s);
        Mat test_image_ = imread("street_image.jpg");
        cvtColor(test_image_, test_image_, CV_BGR2GRAY);
     //   resize(test_image_, test_image_, Size(0,0), 0.125, 0.125);
        resize(test_image_, test_image_, Size(0,0), 0.5, 0.5);
    //    resize(test_image_, test_image_, Size(0,0), 0.00625,0.00625);
     

        cout << "Creating Filter Responses for test image" << endl;
        textonMap.createFilterResponses(test_image_, 0);
        cout << "Generating Texton Map for test image " << endl;
    time_t s1 = time(0);
        Mat textonMap1 = textonMap.generateTextonMap(test_image_);
    time_t s2 = time(0);
    cout << "Time taken for Texton map generation : " << (s2-s1) << "seconds" << endl;

        Mat test_image2_ = imread("street_image2.jpg");
        cvtColor(test_image2_, test_image2_, CV_BGR2GRAY);
        resize(test_image2_, test_image2_, Size(0,0), 0.5, 0.5);


       // resize(test_image2_, test_image2_, Size(0,0), 0.00625, 0.00625);

        cout << "Creating Filter Responses for rotated test image" << endl;
        textonMap.createFilterResponses(test_image2_, 0);
        cout << "Generating Texton Map for test image " << endl;
        Mat textonMap2 = textonMap.generateTextonMap(test_image2_);
    }
   

    return 0;
}
