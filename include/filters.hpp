#include <iostream>
#include <cstdio>
#include <ctime>
#include <limits>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>
#include <string>
#include <fstream>
#define PI 3.14
using namespace std;
using namespace cv;

typedef struct feature {
    float Filter1B;
    float Filter1G;
    float Filter1R;
    float Filter2B;
    float Filter2G;
    float Filter2R;
}pixelFeatures;

typedef struct {
    float Filters[42];
}FilterResponses;
class Textons {
    public:
        int k;
        static const int NumFilters = 42;
        static const int SUP = 9;
        vector<FilterResponses> TrainFilterResponses;
        vector<FilterResponses> Dictionary;
        vector<FilterResponses> TestImageTextonMap;

        Mat FilterResponsesKmeans;//(1,1, CV_32F, Scalar(0));
        Mat centers;
        Mat TextonMap;
        Mat F[36];

        void makeRFSFilters();
        void createFilterResponses(InputArray input_image_, int FlagTrainTest);
        void pushToDictionary(Mat DoG_DDoG[], Mat G[], Mat LoG[]);
        //void pushToDictionary(Mat G[], Mat LoG[]);
        void pushToImageTextons(Mat DoG_DDoG[], Mat G[], Mat LoG[]);
        //void pushToImageTextons(Mat G[], Mat LoG[]);
        void computeKmeans();
        Mat generateTextonMap(InputArray input_image_);
        void writeTextonMapToFile();
        
        Textons(int DictionarySize);
        ~Textons();
};

