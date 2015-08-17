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
    int num_orientations = 2;
    int scalesX[3] = {1,2,4};
    int scalesY[3] = {3,6,12};
    int scalesGL[3] = {5, 10, 15};
    Mat grey_image_, input_image, grey_image;
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
    //normalize the input image
    //normalize(grey_image_, grey_image_);

    int num_rotInvariants = 2;
    Mat DoG[3];
    Mat DDoG[3];
    Mat G[3], LoG[3];

    Mat DoG_temp[3][2];
    Mat DDoG_temp[3][2];
    Mat DoG_DDoG[12];

    double angles[2] = {0,PI/2};

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
        Sobel(gaussian_image_dy, gaussian_image_ddxy, -1, 1, 0, 5);
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
            Mat tempx, tempy, tempxx,tempyy, tempxy;
            gaussian_image_dx.copyTo(tempx);
            gaussian_image_dy.copyTo(tempy);

            gaussian_image_ddx.copyTo(tempxx);
            gaussian_image_ddy.copyTo(tempyy);
            gaussian_image_ddxy.copyTo(tempxy);
            
            tempx *= (float)cos(angle);
            tempx.convertTo(tempx, CV_32F);
            pow(tempx, 2, tempx);
            tempy *= (float)sin(angle);
            tempy.convertTo(tempy, CV_32F);
            pow(tempy, 2, tempy);

            tempxx += tempxy;
            tempyy += tempxy;

            tempxx *= (float)cos(angle);
            tempxx.convertTo(tempxx, CV_32F);
            pow(tempxx, 2, tempxx);
            tempyy *= (float)sin(angle);
            tempyy.convertTo(tempyy, CV_32F);
            pow(tempyy, 2, tempyy);

            DoG_temp[i][j] = tempx + tempy;
            sqrt(DoG_temp[i][j], DoG_temp[i][j]);
            DDoG_temp[i][j] = tempxx + tempyy;
            sqrt(DDoG_temp[i][j], DDoG_temp[i][j]);


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
               int temp = DoG_DDoG[i*num_orientations +0].at<float>(r,c);
               for (int j = 1; j < num_orientations; j++) {
                   if(temp < DoG_DDoG[i*num_orientations+j].at<float>(r,c)){
                       temp = DoG_DDoG[i*num_orientations+j].at<float>(r,c);
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
               values[i] = DoG_DDoG[i*num_orientations + max_index].at<float>(r,c);
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
                    temp[j] = DoG_DDoG[i*num_orientations+indices_final[j]].at<float>(r,c);
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

