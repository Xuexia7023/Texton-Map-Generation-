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

    double sqrPts[size];
    int variance  = scale*scale;
    double denom = 2*variance;
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
    }
    return;
}

void normalise(Mat *F, double *gx, double *gy, int sup) {

    double sum = 0, absSum = 0;
    //ofstream f;
    //f.open("f.txt", ios::app);
    Mat Ftemp(sup, sup, CV_64F);
    for (int i = 0; i < sup; i++) {
        for (int j = 0; j < sup; j++) {
            double temp = (double)(gx[i*sup+j]*gy[i*sup+j]);
            Ftemp.at<double>(j,i) = temp;
            sum = sum + temp;
        }
    }
    double mean = sum/(sup*sup);
    for (int i = 0; i < sup; i++) {
        for (int j = 0; j < sup; j++) {
            Ftemp.at<double>(i,j) -= mean;
            absSum += abs(Ftemp.at<double>(i,j));
        }
    }

   for (int i = 0; i < sup; i++) {
       for (int j = 0; j < sup; j++) {
           Ftemp.at<double>(i,j) /= absSum;
           F->at<float>(i,j) = (float)Ftemp.at<double>(i,j);
   //        f << F->at<float>(i,j) << ", ";
       }
   //    f << endl;
   }
   //f << "----------" << endl;
   //f.close();
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

    int SCALEX[3] = {1, 2, 4};
    for (int i = 0; i < NF; i++) {
        F[i].create(SUP, SUP, CV_32FC1);
    }

    int hsup = (SUP - 1)/2;
    float x[SUP*SUP], y[SUP*SUP];

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
                    rotPtsx[i*SUP+j] = x_prime*c - y_prime*s;
                    rotPtsy[i*SUP+j] = x_prime*s + y_prime*c;
                }
            }
            makeFilter(&F[count], SCALEX[scale], 0, 1, rotPtsx, rotPtsy, SUP);
            makeFilter(&F[count + NEDGE], SCALEX[scale], 0, 2, rotPtsx, rotPtsy, SUP);
            count++;
        }
    }
    return;
}
 
void Textons::createFilterResponses(InputArray input_image_, int FlagTrainTest) {
    Mat grey_image_float_, input_image, grey_image_;
    float scales[3] = {sqrt(2), sqrt(8), sqrt(32)};

    ofstream FR;
    FR.open("FR.txt", ios::app);
    //resize(input_image_, input_image, Size(0,0), 0.25, 0.25);
    //cvtColor(input_image_, grey_image_, CV_BGR2GRAY);
    input_image_.copyTo(grey_image_);
    grey_image_.convertTo(grey_image_float_, CV_64F);

    Mat ImageFilterResponses[NumFilters];
    for (int i = 0; i < NF; i++) {
        filter2D(grey_image_float_, ImageFilterResponses[i], -1, F[i], Point(-1,-1),
             0, BORDER_DEFAULT);   

        ImageFilterResponses[i].convertTo(ImageFilterResponses[i], CV_32F);
        //imshow("filterResponses" , ImageFilterResponses[i]);
        //waitKey();
    }
  /*  
    for (int r = 0; r < grey_image_.rows; r++) {
        for (int c = 0; c < grey_image_.cols; c++) {
            for (int i = 0; i < NF; i++) {
                FR << ImageFilterResponses[i].at<float>(r,c) << ", ";
            }
            FR << endl;
        }
        FR << endl;
    }
    
    FR << "------------------------" << endl;
    */
    for (int i = 0; i < 3; i++) {
        GaussianBlur(grey_image_, ImageFilterResponses[i+NF], Size(SUP,SUP),
            scales[i], 0, BORDER_DEFAULT); 

        Laplacian(ImageFilterResponses[i+NF], ImageFilterResponses[i+NF+3], CV_8UC1,
            SUP, scales[i], 0, BORDER_DEFAULT); 
        //imshow("filterResponsesLoG", ImageFilterResponses[i+NF+3]);
        //waitKey();

        ImageFilterResponses[i+NF].convertTo(ImageFilterResponses[i+NF], CV_32F);
        ImageFilterResponses[i+NF+3].convertTo(ImageFilterResponses[i+NF+3], CV_32F);
    }

    /*
    for (int i = 0; i < NumFilters; i++) {
        string ty =  type2str( ImageFilterResponses[i].type() );
        printf("Matrix: %s %dx%d \n", ty.c_str(), ImageFilterResponses[i].cols, ImageFilterResponses[i].rows );
        cout << i << endl;

    }
    */
  
    /*
    for (int r = 0; r < grey_image_.rows; r++) {
        for (int c = 0; c < grey_image_.cols; c++) {
            for (int i = 0; i < NF/NORIENT; i++ ) {
                float temp = abs(ImageFilterResponses[i*NORIENT+0].at<float>(r,c));
                int max_index = 0;
                float values[NORIENT];
                values[0] = ImageFilterResponses[i*NORIENT+0].at<float>(r,c);
                for (int j = 1; j < NORIENT; j++) {
                    if (temp < abs(ImageFilterResponses[i*NORIENT+j].at<float>(r,c))) {
                        temp =  abs(ImageFilterResponses[i*NORIENT+j].at<float>(r,c));
                        max_index = j;
                    }
                    values[j] = ImageFilterResponses[i*NORIENT+j].at<float>(r,c);
                }
      //          FR << "MAX_INDEX: " << max_index << endl;
                int iter = 0;
                if (values[max_index] < 0 && i < NF/(NORIENT*2)) {
                    for (int j = 0; j < NORIENT; j++) {
                        values[j] *= -1;
                    }
                }
                for (int j = max_index; j < NORIENT; j++) {

      //              FR << values[j] << endl;
                    //if (values[max_index] < 0)
                    //    values[j] *= -1;

                    ImageFilterResponses[i*NORIENT+iter].at<float>(r,c) = values[j];
                    
                    iter++;
                }
                for (int j = 0; j < max_index; j++) {

       //              FR << values[j] << endl;
                     //if (values[max_index] < 0)
                     //   values[j] = values[j]*(-1);

                     ImageFilterResponses[i*NORIENT+iter].at<float>(r,c) = values[j];
                     if ( i < NF/(NORIENT*2))
                         ImageFilterResponses[i*NORIENT+iter].at<float>(r,c) *= -1;
                     iter++;
                }
            }
         //   FR << endl;
        }
        //FR << endl;
    }
    */
    
    for (int r = 0; r < grey_image_.rows; r++) {
        for (int c = 0; c < grey_image_.cols; c++) {
            int i = 0;
            int index_bar = 0; 
            float max_temp_bar = abs(ImageFilterResponses[0].at<float>(r,c));
            for (i = 1; i < NBAR; i++) {
                if (max_temp_bar < abs(ImageFilterResponses[i].at<float>(r,c))) {
                    max_temp_bar = abs(ImageFilterResponses[i].at<float>(r,c));
                    index_bar = i;
                }
            }
            int flag1 = 0;
            if (ImageFilterResponses[index_bar].at<float>(r,c) < 0)
                flag1 = 1;
            int index_edge = i; 
            float max_temp_edge = abs(ImageFilterResponses[i].at<float>(r,c));
            for (i = NBAR; i < NF; i++) {
                if (max_temp_edge < abs(ImageFilterResponses[i].at<float>(r,c))) {
                    max_temp_edge = abs(ImageFilterResponses[i].at<float>(r,c));
                    index_edge = i;
                }
            }
            
            //int flag2 = 0;
            //if (ImageFilterResponses[index_edge].at<float>(r,c) < 0)
            //    flag2 = 1;
            
            int max = index_bar;
            if (max_temp_bar < max_temp_edge) 
                max = index_edge;
            max = max%NORIENT;

            for (int i = 0; i < NSCALES*2; i++) {
                float values[NORIENT];
                for (int j = 0; j < NORIENT; j++) {
                    values[j] = ImageFilterResponses[NORIENT*i+j].at<float>(r,c);
                    if (flag1 == 1 && i < NSCALES)
                        values[j] = -1*values[j];
                    //if (flag2 == 1 && i >= 3)
                    //    values[j] = -1*values[j];
                }
                
                int iter = 0;
                for (int k = max; k < NORIENT; k++) {
                    ImageFilterResponses[NORIENT*i + iter].at<float>(r,c) = values[k];
                    iter++;
                }
                for (int k = 0; k < max; k++) {
                    ImageFilterResponses[NORIENT*i + iter].at<float>(r,c) = values[k];
                    iter++;
                }
            }

        }
    }
    
   /* 
    for (int r = 0; r < grey_image_.rows; r++) {
        for (int c = 0; c < grey_image_.cols; c++) {
            for (int i = 0; i < NF; i++) {
                FR << ImageFilterResponses[i].at<float>(r,c) << ", ";
            }
            FR << endl;
        }
        FR << endl;
    }
    FR << "**********************" << endl;
 */
    if (FlagTrainTest == 1) 
        Textons::pushToDictionary(ImageFilterResponses);
    else
        Textons::pushToImageTextons(ImageFilterResponses);
    return; 
}

void Textons::pushToDictionary(Mat Responses[]){
    for (int r = 0; r < Responses[0].rows; r++) {
        for (int c = 0; c < Responses[0].cols; c++) {
            FilterResponses temp;// = new FilterResponses;

            for (int i = 0; i < NumFilters; i++) {
                temp.Filters[i] = Responses[i].at<float>(r,c);
            }
            TrainFilterResponses.push_back(temp);
        }
    }
}
void Textons::pushToImageTextons(Mat Responses[]){

    //ofstream IT;
    //IT.open("imageTextons.txt", ios::app);
    TestImageTextonMap.clear();
    for (int r = 0; r < Responses[0].rows; r++) {
        for (int c = 0; c < Responses[0].cols; c++) {
            FilterResponses temp;// = new FilterResponses;

            for (int i = 0; i < NumFilters; i++) {
                temp.Filters[i] = Responses[i].at<float>(r,c);
      //          IT << Responses[i].at<float>(r,c) << ", ";
            }
      //      IT << endl;
            TestImageTextonMap.push_back(temp);
        }
    }
   // IT << "***************" << endl;
}
/* --- compute K Means for PixelFeatureVector ----- */ 
void Textons::computeKmeans(){
    //cout << "inside k means" << endl;
    //ofstream fileFilters;
    //fileFilters.open("filterResponses.txt");

    FilterResponsesKmeans.create(1,1, CV_32F);
    resize(FilterResponsesKmeans, FilterResponsesKmeans, Size(NumFilters, TrainFilterResponses.size()),0,0);
    for (int i = 0; i < TrainFilterResponses.size(); i++) {
        for (int j = 0; j < NumFilters; j++) {
            FilterResponsesKmeans.at<float>(i,j) = (float)TrainFilterResponses[i].Filters[j];
      //      fileFilters << FilterResponsesKmeans.at<float>(i,j) << "; ";
        }
      //  fileFilters << endl;
    }
    //fileFilters.close();
    Mat labels;
    cout << "Computing K means ... " << endl;
    clock_t start = clock();
    kmeans(FilterResponsesKmeans, k, labels, TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10,1.0), NumFilters, KMEANS_PP_CENTERS, centers);
    clock_t end = clock();
    //cout << (end - start)/(double)CLOCKS_PER_SEC << " seconds" << endl;
    //cout << "after calling kmeans function" << endl;

    //imshow("labels", labels);
    //waitKey();
   for (int i = 0; i < k; i++) {
       FilterResponses temp;
       for (int j = 0; j < NumFilters; j++) {
           temp.Filters[j] = centers.at<float>(i,j);
       }
       Dictionary.push_back(temp);
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

/*
    ofstream cf;
    cf.open("centers.txt");
   for (int i = 0; i < Dictionary.size(); i++) {
       for (int j = 0; j < NumFilters; j++) {
           cf << Dictionary[i].Filters[j] << ", " ;
       }
       cf << endl << "----------------------" << endl;
   }
   cf.close();
   */
    //ofstream file;
    //file.open("distances.txt");

    for (int r = 0; r < TextonMapLocal.rows; r++) {
        for (int c = 0; c < TextonMapLocal.cols; c++) {
            double dist1 = (double) numeric_limits<int>::max(); 
            Mat a,b;
            a.create(NumFilters,1, CV_32F);
            b.create(NumFilters,1, CV_32F);
            //file << "---------------" << endl;
            for (int j = 0; j < NumFilters; j++) {
                a.at<float>(j,0) = TestImageTextonMap[r*TextonMapLocal.cols + c].Filters[j];
              //  file << a.at<float>(j,0) << ", ";
            }
           //file << endl << ".............." << endl; 
            int TextonLabel = 0;
            for (int j = 0; j < Dictionary.size(); j++) {
                for (int l = 0; l < NumFilters; l++) {
                    b.at<float>(l,0) = Dictionary[j].Filters[l];
             //       file << Dictionary[j].Filters[l] << ", ";
                }
             //   file << endl << " ---------------" << endl;
                double dist2 = computeDistance(a, b);    


            //file << "dist2:  " << dist2 << ",  dist1: " << dist1 << endl;;
                if (dist2 < dist1){
                    TextonLabel = j;
                    dist1 = dist2;
                }
            }
            TextonMapLocal.at<uchar>(r,c) = (int)(255/(TextonLabel+1));

        }
        //file << endl;
    }
    //file << " ***************** " << endl;
    //file.close();

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
/*    imshow("textonMap", TextonMapLocal);
    waitKey();
*/    
    return TextonMapLocal;
}

