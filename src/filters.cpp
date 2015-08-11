#include "../include/filters.hpp"

using namespace std;
using namespace cv;
Textons::Textons(int DictionarySize):
k(DictionarySize){
}

Textons::~Textons(){
}
string type2str(int type) {
    string r;
    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);
    switch (depth) {
        case CV_8U:
            r = "8U";
            break;
        case CV_8S:
            r = "8S";
            break;
        case CV_16U:
            r = "16U";
            break;
        case CV_16S:
            r = "16S";
            break;
        case CV_32S:
            r = "32S";
            break;
        case CV_32F:
            r = "32F";
            break;
        case CV_64F:
            r = "64F";
            break;
        default: 
            r = "User";
            break;
    }
    r += "C";
    r += (chans+'0');
    return r;
}
 
void Textons::createFilterBank(InputArray input_image_, int FlagTrainTest) {
    FilterResponses temp;
    int num_orientations = 6;
    int scalesX[3] = {1,2,4};
    int scalesY[3] = {3,6,12};
    Mat grey_image_, input_image;
    //resize(input_image_, input_image, Size(0,0), 0.25, 0.25);
    cvtColor(input_image_, grey_image_, CV_BGR2GRAY);

    //normalize the input image
    //normalize(grey_image_, grey_image_);

    int num_rotInvariants = 2;
    Mat DoG[3];
    Mat DDoG[3];
    Mat G, LoG;

    G.create(grey_image_.rows, grey_image_.cols, CV_32F);
    LoG.create(grey_image_.rows, grey_image_.cols, CV_32F);

    for (int i = 0; i < 3; i++) {
        //DoG[i].create(grey_image_.rows, grey_image_.cols, CV_32F);
        //DDoG[i].create(grey_image_.rows, grey_image_.cols, CV_32F);
        Mat DoG_temp[6];
        Mat DDoG_temp[6];
        Mat output_image_gaussian_dx(grey_image_.rows, grey_image_.cols, CV_32F);
        Mat output_image_gaussian_dy(grey_image_.rows, grey_image_.cols, CV_32F);
        Mat output_image_gaussian(grey_image_.rows, grey_image_.cols, CV_32F);

        GaussianBlur(grey_image_, output_image_gaussian_dx, Size(3,3), scalesX[i]);
        GaussianBlur(grey_image_, output_image_gaussian_dy, Size(3,3), scalesY[i]);

        //For Gaussian sigma = 10
        GaussianBlur(grey_image_, G, Size(3,3), 10);
        //For Laplacian of Gaussian sigma = 10
        Laplacian(G, LoG, -1);
        Mat gaussian_image_dx, gaussian_image_dy;
        Mat gaussian_image_ddx, gaussian_image_ddy;

        //First derivative of Gaussian 
        Sobel(output_image_gaussian_dx, gaussian_image_dx, -1, 1, 0, 3);
        Sobel(output_image_gaussian_dy, gaussian_image_dy, -1, 0, 1, 3);

        //Second derivative of Gaussian
        Sobel(gaussian_image_dx, gaussian_image_ddx, -1, 1, 0, 3);
        Sobel(gaussian_image_dy, gaussian_image_ddy, -1, 0, 1, 3);

        //imshow("ddx", gaussian_image_ddx);
        //waitKey();
        //imshow("ddy", gaussian_image_ddy);
        //waitKey();
        for (int j = 0; j < 6; j++) {
            DoG_temp[j].create(grey_image_.rows, grey_image_.cols, CV_32F);
            DDoG_temp[j].create(grey_image_.rows, grey_image_.cols, CV_32F);
            double angle = (j+1)*PI/7;
            //DoG_temp[j] = cos(angle)*gaussian_image_dx + sin(angle)*gaussian_image_dy;
            Mat tempx, tempy, tempxx, tempyy;
            gaussian_image_dx.copyTo(tempx);
            gaussian_image_dy.copyTo(tempy);

            gaussian_image_ddx.copyTo(tempxx);
            gaussian_image_ddy.copyTo(tempyy);

            tempx *= (float)cos(angle);
            tempy *= (float)sin(angle);


            tempxx *= (float)cos(angle);
            tempyy *= (float)sin(angle);

            DoG_temp[j] = tempx + tempy;
            DDoG_temp[j] = tempxx + tempyy;

    //        cout << "J: " << j << endl;
    //        imshow("angles", DoG_temp[j]);
    //        waitKey();
    //        imshow("angles", DDoG_temp[j]);
    //        waitKey(); 
        }
        
        Mat tempDoG, tempDDoG;
        DoG_temp[0].copyTo(tempDoG);
        DDoG_temp[0].copyTo(tempDDoG);
        for (int orient = 1; orient < 6; orient++) {
            tempDoG += DoG_temp[orient];
            tempDDoG += DDoG_temp[orient];
        }

        tempDoG /= 6;
        tempDDoG /= 6;
        tempDoG.copyTo(DoG[i]);
        tempDDoG.copyTo(DDoG[i]);
    }
//    imshow("DoG" , DoG[0]);
//    waitKey();
//    imshow("DoG" , DoG[1]);
//    waitKey();   
//    imshow("DoG" , DoG[2]);
//    waitKey();
//    imshow("DDoG", DDoG[0]);
//    waitKey();
//    imshow("DDoG", DDoG[1]);
//    waitKey();
//    imshow("DDoG", DDoG[2]);
//    waitKey();
//    imshow("G", G);
//    waitKey();
//    imshow("LoG", LoG);
//    waitKey();

//    normalize(DoG[0], DoG[0]);
//    normalize(DoG[1], DoG[1]);
//    normalize(DoG[2], DoG[2]);
//    normalize(DDoG[0], DDoG[0]);
//    normalize(DDoG[1], DDoG[1]);
//    normalize(DDoG[2], DDoG[2]);
//    normalize(G, G);
//    normalize(LoG, LoG);

    if (FlagTrainTest == 1) 
        Textons::pushToDictionary(DoG, DDoG, G, LoG);
    else
        Textons::pushToImageTextons(DoG, DDoG, G, LoG);
    return;
     
}
void Textons::pushToDictionary(Mat DoG[], Mat DDoG[], Mat G, Mat LoG){
    for (int r = 0; r < G.rows; r++) {
        for (int c = 0; c < G.cols; c++) {
            FilterResponses temp;// = new FilterResponses;
            temp.Filters[0] = DoG[0].at<uchar>(r,c);
            temp.Filters[1] = DoG[1].at<uchar>(r,c);
            temp.Filters[2] = DoG[2].at<uchar>(r,c);
            temp.Filters[3] = DDoG[0].at<uchar>(r,c);
            temp.Filters[4] = DDoG[1].at<uchar>(r,c);
            temp.Filters[5] = DDoG[2].at<uchar>(r,c);
            temp.Filters[6] = G.at<uchar>(r,c);
            temp.Filters[7] = LoG.at<uchar>(r,c);
            TrainFilterResponses.push_back(temp);
        }
    }
}
void Textons::pushToImageTextons(Mat DoG[], Mat DDoG[], Mat G, Mat LoG){
    /*
    for(int i = 0; i < TestImageTextonMap.size(); i++) {
        FilterResponses temp = TestImageTextonMap.at(i);
        delete temp;
    }
    */
    TestImageTextonMap.clear();
    //vector<FilterResponses *> (TestImageTextonMap).swap(TestImageTextonMap);
    for (int r = 0; r < G.rows; r++) {
        for (int c = 0; c < G.cols; c++) {
            FilterResponses temp;// = new FilterResponses;
            temp.Filters[0] = DoG[0].at<uchar>(r,c);
            temp.Filters[1] = DoG[1].at<uchar>(r,c);
            temp.Filters[2] = DoG[2].at<uchar>(r,c);
            temp.Filters[3] = DDoG[0].at<uchar>(r,c);
            temp.Filters[4] = DDoG[1].at<uchar>(r,c);
            temp.Filters[5] = DDoG[2].at<uchar>(r,c);
            temp.Filters[6] = G.at<uchar>(r,c);
            temp.Filters[7] = LoG.at<uchar>(r,c);
            TestImageTextonMap.push_back(temp);
        }
    }
}
/* --- compute K Means for PixelFeatureVector ----- */ 
void Textons::computeKmeans(){
    ofstream fileFilters;
    fileFilters.open("filterResponses.txt");
    FilterResponsesKmeans.create(1,1, CV_32F);
    resize(FilterResponsesKmeans, FilterResponsesKmeans, Size(NumFilters, TrainFilterResponses.size()),0,0);
    for (int i = 0; i < TrainFilterResponses.size(); i++) {
        for (int j = 0; j < NumFilters; j++) {
            FilterResponsesKmeans.at<float>(i,j) = TrainFilterResponses[i].Filters[j];
            fileFilters << FilterResponsesKmeans.at<float>(i,j) << "; ";
        }
        fileFilters << endl;
    }
    fileFilters.close();
    Mat labels;
    kmeans(FilterResponsesKmeans, k, labels, TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10,1.0), NumFilters, KMEANS_RANDOM_CENTERS, centers);

   for (int i = 0; i < k; i++) {
       FilterResponses temp;
       for (int j = 0; j < NumFilters; j++) {
           temp.Filters[j] = centers.at<float>(i,j);
       }
       Dictionary.push_back(temp);
   }
   for (int i = 0; i < Dictionary.size(); i++) {
       for (int j = 0; j < NumFilters; j++) {
           cout << Dictionary[i].Filters[j] << ", " ;
       }
       cout << endl;
   }
}

double computeDistance(Mat a, Mat b) {
    double dist = 0;
    for (int i = 0; i < Textons::NumFilters; i++) {
        dist += pow((a.at<float>(i,0) - b.at<float>(i,0)), 2);
    }
    dist = sqrt(dist);
    return dist;
}
/* ----- Generate Texton Map for a given Image ----- */
void Textons::generateTextonMap(InputArray input_image_) {//, Mat TextonMap) {
    int width = input_image_.cols();
    int height = input_image_.rows();
    TextonMap.create(1,1,CV_8UC3);

    Mat TextonMapLocal(input_image_.rows(), input_image_.cols(), CV_8UC1);
    resize(TextonMap, TextonMap, Size(input_image_.cols(), input_image_.rows()), 0, 0);

    ofstream file;
    file.open("distances.txt");

    for (int r = 0; r < TextonMapLocal.rows; r++) {
        for (int c = 0; c < TextonMapLocal.cols; c++) {
            double dist1 = (double) numeric_limits<int>::max(); 
            Mat a,b;
            a.create(NumFilters,1, CV_32F);
            b.create(NumFilters,1, CV_32F);
            for (int j = 0; j < NumFilters; j++) {
                a.at<float>(j,0) = TestImageTextonMap[r*TextonMapLocal.cols + c].Filters[j];
            }
            
            int TextonLabel;
            for (int j = 0; j < k; j++) {
                for (int l = 0; l < NumFilters; l++) {
                    b.at<float>(l,0) = Dictionary[l].Filters[j];
                }
                double dist2 = computeDistance(a, b);    
                if (dist2 < dist1){
                    TextonLabel = j;
                    dist1 = dist2;
                }
            }
            TextonMapLocal.at<uchar>(r,c) = 255/(TextonLabel+1);
            file << ", " << (int)TextonLabel;
        }
        file << endl;
    }
    file.close();

    uchar colors[k][3];
    for (int i = 0; i < k; i++) {
        colors[i][0] = (uchar)random();
        colors[i][1] = (uchar)random();
        colors[i][2] = (uchar)random();
    }

    Mat TextonMapColors(TextonMapLocal.rows, TextonMapLocal.cols, CV_8UC3);
    for (int i = 0; i < TextonMapLocal.rows; i++) {
        for (int j = 0; j < TextonMapLocal.cols; j++) {
            uchar TextonLabel = TextonMapLocal.at<uchar>(i,j);
            TextonMapColors.at<Vec3b>(i,j)[0] = colors[k-TextonLabel][0];
            TextonMapColors.at<Vec3b>(i,j)[1] = colors[k-TextonLabel][1];
            TextonMapColors.at<Vec3b>(i,j)[2] = colors[k-TextonLabel][2];
        }
    }
    imshow("textonMapColor", TextonMapColors);
    waitKey();
    imshow("textonMap", TextonMapLocal);
    waitKey();
    return;
}

