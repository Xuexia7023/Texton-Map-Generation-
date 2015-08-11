#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <string>
#include <fstream>

#include "../include/filters.hpp"
using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
    int k = atoi(argv[1]);
    int numTrainingImages = atoi(argv[2]);
    Textons textonMap(k);

    int iter = 1;
    while (iter < numTrainingImages) {
        char fname[100];
        sprintf(fname, "example_roads/img%d.jpg", iter);
        Mat input_image_ = imread(fname);
        resize(input_image_, input_image_, Size(0,0), 0.25,0.25);
        textonMap.createFilterBank(input_image_, 1);
        iter++;
    }
    
    textonMap.computeKmeans();
    Mat test_image_ = imread("street_image.jpg");
    resize(test_image_, test_image_, Size(0,0), 0.25, 0.25);
    textonMap.createFilterBank(test_image_, 0);
    textonMap.generateTextonMap(test_image_);
    return 0;
}
