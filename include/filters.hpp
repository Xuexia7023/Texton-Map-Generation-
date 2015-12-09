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
#define NUM_THREADS 8
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
    float Filters[39];
}FilterResponses;
class Textons {
    public:
        int k;

        static const int SUP = 51;
        static const int NSCALES = 3; //Number of Scales
        static const int NORIENT = 6; // Number of orientatioons
             
        static const int NROTINV = 3;
        static const int NBAR = NSCALES*NORIENT;
        static const int NEDGE = NSCALES*NORIENT;
             
        static const int NumFilters = NROTINV + NBAR + NEDGE;
        static const int NF = NBAR+NEDGE;
        vector<FilterResponses> TrainFilterResponses;
        vector<FilterResponses> Dictionary;
        vector<FilterResponses> TestImageTextonMap;

        Mat FilterResponsesKmeans;//(1,1, CV_32F, Scalar(0));
        Mat centers;
        Mat TextonMap;
        Mat F[NF]; //kernels

        void makeRFSFilters();
        void KmeansCentersReadFromFile(string s);
        void createFilterResponses(InputArray input_image_, int FlagTrainTest);
        void pushToDictionary(Mat FilterResponses[]);
        void pushToImageTextons(Mat FilterResponses[]);
        void computeKmeans();
        Mat generateTextonMap(InputArray input_image_);
        void writeTextonMapToFile();
        
        Textons(int DictionarySize);
        ~Textons();
};

