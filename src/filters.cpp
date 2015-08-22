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
void gauss1d(double *g, int size, int scale, float *pts, int order) {

    float sqrPts[size];
    int variance  = scale*scale;
    float denom = 2*variance;
    for (int i = 0; i < size; i++) {
        sqrPts[i] = pts[i]*pts[i];
        g[i] = exp(-1*sqrPts[i]/denom);
        g[i] /= sqrt(PI*denom);

        switch (order) {
            case 1: 
                g[i] = -1*g[i]*pts[i]/variance;
                break;
            case 2:
                g[i] = g[i]*(sqrPts[i]-variance)/(variance*variance);
                break;
        }
        g[i] = (int)(g[i]*10000);
        g[i] = (float)g[i]/10000;
    }
    return;
}

void normalise(Mat *F, double *gx, double *gy, int sup) {

    float sum = 0, absSum = 0;
    ofstream Ffile;
    Ffile.open("f.txt", ios::app);
    for (int i = 0; i < sup; i++) {
        for (int j = 0; j < sup; j++) {
            double temp = double(gx[i*sup+j]*gy[i*sup+j]);
            temp = (int)(temp*10000);
            temp = temp/10000;
            F->at<float>(j,i) = temp;
            sum += temp;
            absSum += abs(temp);
        }
    }

    float mean = sum/(sup*sup);
    for (int i = 0; i < sup; i++) {
        for (int j = 0; j < sup; j++) {
            F->at<float>(i,j) -= mean;
            F->at<float>(i,j) /= absSum;
            F->at<float>(i,j) = (int)(F->at<float>(i,j)*100000);
            F->at<float>(i,j) /= 100000; //(F->at<float>(i,j)/1000);

            Ffile << F->at<float>(i,j) << ", ";

        }
        Ffile << endl;
    }

    Ffile << "--------------------" << endl;

    Ffile.close();
    return;
}
void makeFilter(Mat *F, int scale, int phasex, int phasey, float rotPtsx[], float rotPtsy[], int sup) {
    double gx[sup*sup];
    double gy[sup*sup];
    gauss1d(gx, sup*sup, 3*scale, rotPtsx, phasex);
    gauss1d(gy, sup*sup, scale, rotPtsy, phasey);

    normalise(F, gx, gy, sup);
    return;
} 
void Textons::makeRFSFilters() {

    //int SUP = 9; // Support of the largest filter
    int NSCALES = 3; //Number of Scales
    int SCALEX[3] = {1, 2, 4}; // Sigma_{x} for the oriented filters
    int NORIENT = 6; // Number of orientatioons 

    int NROTINV = 2*3;
    int NBAR = NSCALES*NORIENT;
    int NEDGE = NSCALES*NORIENT;

    int NF = NBAR+NEDGE;
    
    //Mat F[42]; // NF
    for (int i = 0; i < NF; i++) {
        F[i].create(SUP, SUP, CV_32FC1);
    }

    int hsup = (SUP - 1)/2;
    float x[SUP*SUP], y[SUP*SUP];
    ofstream pts;
    pts.open("pts.txt");

    for (int i = 0; i < SUP; i++) {
        for (int j = 0; j < SUP; j++) {
            x[i*SUP + j] = -1*hsup + i;
            y[j*SUP + i] = hsup - i; 
        }
    }

    int count = 0;
    for (int scale = 0; scale < NSCALES; scale++) {
        for (int orient = 0; orient < NORIENT; orient++) {
            float angle = PI*orient/NORIENT; 
            float c = cos(angle);
            float s = sin(angle);

            // Calculate rotated points
            float rotPtsx[SUP*SUP];
            float rotPtsy[SUP*SUP];
            for (int i = 0; i < SUP; i++) {
                for (int j = 0; j < SUP; j++) {
                    float x_prime = x[i*SUP+j];
                    float y_prime = y[i*SUP+j];
                    rotPtsx[i*SUP+j] = (int)((x_prime*c - y_prime*s)*1000);
                    rotPtsy[i*SUP+j] = (int)((x_prime*s + y_prime*c)*1000);
                    rotPtsx[i*SUP+j] /= 1000;
                    rotPtsy[i*SUP+j] /= 1000;
                    pts << rotPtsx[i*SUP+j] << ", ";
                    pts << rotPtsy[i*SUP+j] << ", ";
                    pts << endl;
                }
            }
            pts << endl << "---------------- " << endl;
            makeFilter(&F[count], SCALEX[scale], 0, 1, rotPtsx, rotPtsy, SUP);
            makeFilter(&F[count + NEDGE], SCALEX[scale], 0, 2, rotPtsx, rotPtsy, SUP);
            count++;
        }
    }

for (int k = 0; k < NF; k++) {
    for (int i = 0; i < SUP; i++) {
        for (int j = 0; j < SUP; j++) {
  /*        pts << rotPts[i*SUP + j][0] << ", ";
            pts << rotPts[i*SUP + j][1]; // = -1*hsup + i; 
            pts << endl;
    */
            pts << F[k].at<float>(i,j) << ", ";
        }
        pts << endl;
    }
    pts << endl << "---------------------------------------------" << endl;
}

    pts.close();

    return;
}
 
void Textons::createFilterResponses(InputArray input_image_, int FlagTrainTest) {
    FilterResponses temp;
    Mat grey_image_, input_image, grey_image;
    float scales[3] = {sqrt(2), sqrt(8), sqrt(32)};
    //resize(input_image_, input_image, Size(0,0), 0.25, 0.25);
    cvtColor(input_image_, grey_image, CV_BGR2GRAY);
    grey_image_.create(input_image_.rows(), input_image_.cols(), CV_32F);
    ofstream greyImageFile;
    greyImageFile.open("greyImage.txt");

    for (int r = 0; r < grey_image_.rows; r++) {
        for (int c = 0; c < grey_image_.cols; c++) {
            grey_image_.at<float>(r,c) = (float) grey_image.at<uchar>(r,c);
            greyImageFile << grey_image_.at<float>(r,c) << ", " ;
        }
        greyImageFile << endl;
    }
    Mat FilterResponses[42];
    for (int i = 0; i < 36; i++) {
        filter2D(grey_image_, FilterResponses[i], -1, F[i], Point(-1,-1),
             0, BORDER_DEFAULT);   
        imshow("filterResponses" , FilterResponses[i]);
        waitKey();
    }
    for (int i = 0; i < 3; i++) {
        GaussianBlur(grey_image, FilterResponses[i+36], Size(SUP, SUP), scales[i], 0, BORDER_DEFAULT); 
        imshow("filterResponsesG", FilterResponses[i+36]);
        waitKey();
        Laplacian(FilterResponses[i+36], FilterResponses[i+36+3], -1,
                SUP, scales[i], 0, BORDER_DEFAULT); 
        imshow("filterResponsesLoG", FilterResponses[i+36+3]);
        waitKey();
    }
    //normalize the input image
    //normalize(grey_image_, grey_image_);
/*
    int num_rotInvariants = 2;
    Mat DoG[3];
    Mat DDoG[3];
    Mat G[3], LoG[3];

    Mat DoG_temp[3][6];
    Mat DDoG_temp[3][6];
    Mat DoG_DDoG[36];

    double angles[6] = {0, PI/6, PI/3, PI/2, 2*PI/3, 5*PI/6};

    //G.create(grey_image_.rows, grey_image_.cols, CV_32F);
    //LoG.create(grey_image_.rows, grey_image_.cols, CV_32F);

    for (int i = 0; i < 3; i++) {
        //DoG[i].create(grey_image_.rows, grey_image_.cols, CV_32F);
        //DDoG[i].create(grey_image_.rows, grey_image_.cols, CV_32F);
        Mat output_image_gaussian_dx(grey_image_.rows, grey_image_.cols, CV_32F);
        Mat output_image_gaussian_dy(grey_image_.rows, grey_image_.cols, CV_32F);
        Mat output_image_gaussian(grey_image_.rows, grey_image_.cols, CV_32F);

        GaussianBlur(grey_image_, output_image_gaussian_dx, Size(3,3), scalesX[i]);
        GaussianBlur(grey_image_, output_image_gaussian_dy, Size(3,3), scalesY[i]);

        //For Gaussian sigma = 10
        GaussianBlur(grey_image, G[i], Size(3,3), scalesGL[i]);
        Laplacian(G[i], LoG[i], -1, 3, 3, 0, BORDER_DEFAULT);
        //GaussianBlur(grey_image_, G, Size(3,3), 10);
        //For Laplacian of Gaussian sigma = 10
        //Laplacian(G, LoG, -1);
        Mat gaussian_image_dx, gaussian_image_dy;
        Mat gaussian_image_ddx, gaussian_image_ddy, gaussian_image_ddxy;

        //First derivative of Gaussian 
        Sobel(output_image_gaussian_dx, gaussian_image_dx, -1, 1, 0, 5);
        Sobel(output_image_gaussian_dy, gaussian_image_dy, -1, 0, 1, 5);

        //Second derivative of Gaussian
        Sobel(gaussian_image_dx, gaussian_image_ddx, -1, 1, 0, 5);
        Sobel(gaussian_image_dy, gaussian_image_ddy, -1, 0, 1, 5);

        ofstream sobelFiles;
        sobelFiles.open("sobelOutput.txt");

        for (int r = 0; r < grey_image_.rows; r++) {
            for (int c = 0; c < grey_image_.cols; c++) {
                sobelFiles << (float)G[0].at<uchar>(r,c) << ", ";
            }
            sobelFiles << endl;
        }
        //imshow("ddx", gaussian_image_ddx);
        //waitKey();
        //imshow("ddy", gaussian_image_ddy);
        //waitKey();
        for (int j = 0; j < num_orientations; j++) {
            DoG_temp[i][j].create(grey_image_.rows, grey_image_.cols, CV_32F);
            DDoG_temp[i][j].create(grey_image_.rows, grey_image_.cols, CV_32F);
            double angle = angles[i];
            //double angle = (j+1)*PI/7;
            //DoG_temp[j] = cos(angle)*gaussian_image_dx + sin(angle)*gaussian_image_dy;
            Mat tempx, tempy, tempxx, tempyy;
            gaussian_image_dx.copyTo(tempx);
            gaussian_image_dy.copyTo(tempy);

            gaussian_image_ddx.copyTo(tempxx);
            gaussian_image_ddy.copyTo(tempyy);
            
            Mat tanInv_d, tanInv_dd;
            tempy /= tempx;
            tempy.copyTo(tanInv_d);

            tempyy /= tempxx;
            tempyy.copyTo(tanInv_dd);

            for (int r = 0; r < tanInv_d.rows; r++) {
                for (int c = 0; c < tanInv_d.cols; c++) {
                    tanInv_d.at<float>(r,c) = atan(tanInv_d.at<float>(r,c)) + (PI/2) - angles[j];
                    tanInv_dd.at<float>(r,c) = atan(tanInv_dd.at<float>(r,c)) + (PI/2) - angles[j];
                }
            }
            DoG_temp[i][j] = tanInv_d;
            //sqrt(DoG_temp[i][j], DoG_temp[i][j]);
            DDoG_temp[i][j] = tanInv_dd;
            //sqrt(DDoG_temp[i][j], DDoG_temp[i][j]);


            DoG_temp[i][j].copyTo(DoG_DDoG[i*num_orientations+j]);
            DDoG_temp[i][j].copyTo(DoG_DDoG[(i+3)*num_orientations+j]);

    //        cout << "J: " << j << endl;
    //        imshow("angles", DoG_temp[j]);
    //        waitKey();
    //        imshow("angles", DDoG_temp[j]);
    //        waitKey(); 
        }
        
        Mat tempDoG, tempDDoG;
        DoG_temp[i][0].copyTo(tempDoG);
        DDoG_temp[i][0].copyTo(tempDDoG);
        for (int orient = 1; orient < num_orientations; orient++) {
            tempDoG += DoG_temp[i][orient];
            tempDDoG += DDoG_temp[i][orient];
        }

        tempDoG /= num_orientations;
        tempDDoG /= num_orientations;
        tempDoG.copyTo(DoG[i]);
        tempDDoG.copyTo(DDoG[i]);


    }
    
    for (int r = 0; r < grey_image_.rows; r++) {
        for (int c = 0; c < grey_image_.cols; c++) {
            int max, max_label[6];
            int indices[6][num_orientations];
            int indices_final[num_orientations]; //final indices order
            int temp_value;
            int values[6]; //highest value among all the orientations of one filter

            for (int i = 0; i < 6; i++) {
               int max_index = 0; 
               int temp = abs(DoG_DDoG[i*num_orientations +0].at<float>(r,c));
               for (int j = 1; j < num_orientations; j++) {
                   if(temp > abs(DoG_DDoG[i*num_orientations+j].at<float>(r,c))){
                       temp = abs(DoG_DDoG[i*num_orientations+j].at<float>(r,c));
                       max_index = j;
                   }
               }
               int iter = 0;
               for (int j = max_index; j < num_orientations; j++) {
                  // indices[i][iter] = DoG_temp[i][j].at<uchar>(r,c);
                   indices[i][iter] = j;
                   iter++;
               }
               max_label[i] = max_index;
               values[i] = abs(DoG_DDoG[i*num_orientations + max_index].at<float>(r,c));
               for (int j = 0; j < max_index; j++) {
                   //indices[i][iter] = DoG_DDoG[i][j].at<uchar>(r,c);
                   indices[i][iter] = j;
                   iter++;
               }
               if (i == 0) {
                   max = max_label[i];
                   temp_value = values[i];
                   for (int p = 0; p < num_orientations; p++) {
                       indices_final[p] = indices[0][p];
                   }
               }
               else if (temp_value < values[i]) {
                   max = max_label[i];
                   temp_value = values[i];
                   for (int p = 0; p < num_orientations; p++) {
                       indices_final[p] = indices[i][p];
                   }
               }
            }
            //after getting the indices order, we need to arrange the values in the matrix
            for (int i = 0; i < 6; i++) {
                int temp[num_orientations];
                for (int j = 0; j < num_orientations; j++) {
                    temp[j] = abs(DoG_DDoG[i*num_orientations+indices_final[j]].at<float>(r,c));
                }
                for (int j = 0; j < num_orientations; j++) {
                    DoG_DDoG[i*num_orientations+j].at<float>(r,c) = temp[j];
                }
            }
        }
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
        Textons::pushToDictionary(DoG_DDoG, G, LoG);
    else
        Textons::pushToImageTextons(DoG_DDoG, G, LoG);
  */  
    return; 
}

void Textons::pushToDictionary(Mat DoG_DDoG[], Mat G[], Mat LoG[]){
    int num_orientations = 2;
    for (int r = 0; r < G[0].rows; r++) {
        for (int c = 0; c < G[0].cols; c++) {
            FilterResponses temp;// = new FilterResponses;
            /*
            temp.Filters[0] = DoG[0].at<uchar>(r,c);
            temp.Filters[1] = DoG[1].at<uchar>(r,c);
            temp.Filters[2] = DoG[2].at<uchar>(r,c);
            temp.Filters[3] = DDoG[0].at<uchar>(r,c);
            temp.Filters[4] = DDoG[1].at<uchar>(r,c);
            temp.Filters[5] = DDoG[2].at<uchar>(r,c);
            temp.Filters[6] = G.at<uchar>(r,c);
            temp.Filters[7] = LoG.at<uchar>(r,c);
            */

            for (int i = 0; i < 6; i++) {
                for (int j = 0; j < num_orientations; j++) {
                    temp.Filters[i*num_orientations+j] = DoG_DDoG[i*num_orientations+j].at<float>(r,c);
                }
            }
            for (int g = 0; g < 3; g++) {
            temp.Filters[6*num_orientations+g] = (float)G[g].at<uchar>(r,c);
            temp.Filters[6*num_orientations+3+g] = (float)LoG[g].at<uchar>(r,c);
            }
            TrainFilterResponses.push_back(temp);
        }
    }
}
void Textons::pushToImageTextons(Mat DoG_DDoG[], Mat G[], Mat LoG[]){

    int num_orientations = 2;
    /*
    for(int i = 0; i < TestImageTextonMap.size(); i++) {
        FilterResponses temp = TestImageTextonMap.at(i);
        delete temp;
    }
    */
    TestImageTextonMap.clear();
    //vector<FilterResponses *> (TestImageTextonMap).swap(TestImageTextonMap);
    for (int r = 0; r < G[0].rows; r++) {
        for (int c = 0; c < G[0].cols; c++) {
            FilterResponses temp;// = new FilterResponses;
            /*
            temp.Filters[0] = DoG[0].at<uchar>(r,c);
            temp.Filters[1] = DoG[1].at<uchar>(r,c);
            temp.Filters[2] = DoG[2].at<uchar>(r,c);
            temp.Filters[3] = DDoG[0].at<uchar>(r,c);
            temp.Filters[4] = DDoG[1].at<uchar>(r,c);
            temp.Filters[5] = DDoG[2].at<uchar>(r,c);
            temp.Filters[6] = G.at<uchar>(r,c);
            temp.Filters[7] = LoG.at<uchar>(r,c);
            */
            for (int i = 0; i < 6; i++) {
                for (int j = 0; j < num_orientations; j++) {
                    temp.Filters[i*num_orientations+j] = DoG_DDoG[i*num_orientations+j].at<float>(r,c);
                }
            }
            for (int g = 0; g < 3; g++) {
            temp.Filters[6*num_orientations+g] = (float)G[g].at<uchar>(r,c);
            temp.Filters[6*num_orientations+3+g] = (float)LoG[g].at<uchar>(r,c);
            }
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
            FilterResponsesKmeans.at<float>(i,j) = (float)TrainFilterResponses[i].Filters[j];
            fileFilters << FilterResponsesKmeans.at<float>(i,j) << "; ";
        }
        fileFilters << endl;
    }
    fileFilters.close();
    Mat labels;
    cout << "before calling kmeans function" << endl;
    clock_t start = clock();
    kmeans(FilterResponsesKmeans, k, labels, TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10,1.0), NumFilters, KMEANS_RANDOM_CENTERS, centers);
    clock_t end = clock();
    cout << (end - start)/(double)CLOCKS_PER_SEC << " seconds" << endl;
    cout << "after calling kmeans function" << endl;

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
Mat Textons::generateTextonMap(InputArray input_image_) {//, Mat TextonMap) {
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
            TextonMapLocal.at<uchar>(r,c) = (int)TextonLabel;
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
/*    
    imshow("textonMapColor", TextonMapColors);
    waitKey();
    imshow("textonMap", TextonMapLocal);
    waitKey();
*/    
    return TextonMapLocal;
}

