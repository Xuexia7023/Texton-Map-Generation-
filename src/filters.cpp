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
    float angles[NORIENT] = {PI/6, PI/4, PI/3, 2*PI/3, 3*PI/4, 5*PI/6};
    for (int scale = 0; scale < NSCALES; scale++) {
        for (int orient = 0; orient < NORIENT; orient++) {
            float angle = PI*orient/NORIENT;
            //float angle = angles[orient];
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
cv::Mat fspecialLoG(int WinSize, double sigma){

    cv::Mat xx (WinSize,WinSize,CV_64F);
    for (int i=0;i<WinSize;i++){
        for (int j=0;j<WinSize;j++){
            xx.at<double>(j,i) = (i-(WinSize-1)/2)*(i-(WinSize-1)/2);
        }
    }
    cv::Mat yy;
    cv::transpose(xx,yy);
    cv::Mat arg = -(xx+yy)/(2*pow(sigma,2));
    cv::Mat h (WinSize,WinSize,CV_64F);
    for (int i=0;i<WinSize;i++){
        for (int j=0;j<WinSize;j++){
            h.at<double>(j,i) = pow(exp(1),(arg.at<double>(j,i)));
        }
    }
    double minimalVal, maximalVal;
    minMaxLoc(h, &minimalVal, &maximalVal);
    cv::Mat tempMask = (h>DBL_EPSILON*maximalVal)/255;
    tempMask.convertTo(tempMask,h.type());
    cv::multiply(tempMask,h,h);
    
    if (cv::sum(h)[0]!=0){h=h/cv::sum(h)[0];}
    
    cv::Mat h1 = (xx+yy-2*(pow(sigma,2)))/(pow(sigma,4));
                  cv::multiply(h,h1,h1);
                  h = h1 - cv::sum(h1)[0]/(WinSize*WinSize);
                  return h;
}
 
void Textons::createFilterResponses(InputArray input_image_, int FlagTrainTest) {
    Mat grey_image_float_, input_image, grey_image_;
    float scales[3] = {sqrt(1), sqrt(2), sqrt(3)};


    input_image_.copyTo(grey_image_);
    grey_image_.convertTo(grey_image_float_, CV_64F);


    Mat ImageFilterResponses[NumFilters];
    for (int i = 0; i < NF; i++) {
        filter2D(grey_image_float_, ImageFilterResponses[i], -1, F[i], Point(-1,-1),
             0, BORDER_REPLICATE);

        ImageFilterResponses[i].convertTo(ImageFilterResponses[i], CV_32F);
    }
    
    int sup = 51;
    for (int i = 0; i < 3; i++) {
        Mat h = fspecialLoG(sup, scales[i]);
        filter2D(grey_image_float_, ImageFilterResponses[i+NF], -1, h, Point(-1,-1),0, BORDER_DEFAULT);

        ImageFilterResponses[i+NF].convertTo(ImageFilterResponses[i+NF], CV_32F);
    }

    
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
          
            int max = index_bar;
            max = max%NORIENT;

            for (int i = 0; i < NSCALES*2; i++) {
                float values[NORIENT];
                for (int j = 0; j < NORIENT; j++) {
                    values[j] = ImageFilterResponses[NORIENT*i+j].at<float>(r,c);
                    if (flag1 == 1 && i < NSCALES)
                        values[j] = -1*values[j];
                }
                
                int iter = 0;
                for (int k = max; k < NORIENT; k++) {
                    ImageFilterResponses[NORIENT*i + iter].at<float>(r,c) = values[k];
                    iter++;
                }
                for (int k = 0; k < max; k++) {
                    if ((NORIENT*i + iter) < NBAR)
                        ImageFilterResponses[NORIENT*i + iter].at<float>(r,c) = -1*values[k];
                    else
                        ImageFilterResponses[NORIENT*i + iter].at<float>(r,c) = values[k];                    
                    iter++;
                }
            }

        }
    }

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

    TestImageTextonMap.clear();
    for (int r = 0; r < Responses[0].rows; r++) {
        for (int c = 0; c < Responses[0].cols; c++) {
            FilterResponses temp;

            for (int i = 0; i < NumFilters; i++) {
                temp.Filters[i] = Responses[i].at<float>(r,c);
            }
            TestImageTextonMap.push_back(temp);
        }
    }
}
/* --- compute K Means for PixelFeatureVector ----- */ 
void Textons::computeKmeans(){


    ofstream kmeansCenters;
    kmeansCenters.open("kmeans.txt");
    FilterResponsesKmeans.create(1,1, CV_32F);
    resize(FilterResponsesKmeans, FilterResponsesKmeans, Size(NumFilters, TrainFilterResponses.size()),0,0);
    for (int i = 0; i < TrainFilterResponses.size(); i++) {
        for (int j = 0; j < NumFilters; j++) {
            FilterResponsesKmeans.at<float>(i,j) = (float)TrainFilterResponses[i].Filters[j];
        }
    }
    Mat labels;
    cout << "Computing K means ... " << endl;
    clock_t start = clock();
    kmeans(FilterResponsesKmeans, k, labels, TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10,1.0), NumFilters, KMEANS_PP_CENTERS, centers);
    clock_t end = clock();
    //cout << (end - start)/(double)CLOCKS_PER_SEC << " seconds" << endl;
    //cout << "after calling kmeans function" << endl;

   for (int i = 0; i < k; i++) {
       FilterResponses temp;
       for (int j = 0; j < NumFilters; j++) {
           temp.Filters[j] = centers.at<float>(i,j);
           kmeansCenters << temp.Filters[j] << " ";
       }
       kmeansCenters << endl;
       Dictionary.push_back(temp);
   }
   kmeansCenters.close();
}

void Textons::KmeansCentersReadFromFile(string s) {
    ifstream kmeansCenters("kmeans.txt", ios::in);
    //ifstream kmeansCenters(s, ios::in);

    for (int i = 0; i < k; i++) {
        FilterResponses temp;
        for (int j = 0; j < NumFilters; j++) {
            kmeansCenters >> temp.Filters[j]; 
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
typedef struct str_thread_Args {
    int tid;
    vector <FilterResponses> *dictionary_thread;
    vector <FilterResponses> *pointsThread;
    Mat *textonMapLocal;
    int rows;
    int cols;
    int numFilters;
}thread_args_t;

void *computeTexton(void *ptr) {
    
    //    struct thread_args_t arg;
    //    arg = *(thread_args_t *) ptr;
    thread_args_t *args = (thread_args_t *)ptr;
    int t = args->tid;
    int r = args->rows;
    int c = args->cols;
    int n = args->numFilters;
    vector <FilterResponses> dict_local;
    vector <FilterResponses> points;
    points = *args->pointsThread;
    dict_local = *args->dictionary_thread;
    //    vector <FilterResponses> points_local;
    //    ofstream file;
    //    file.open("vals.txt");
    Mat *textonMapThreadLocal = args->textonMapLocal;
    
    //    cout  << "from thread: " << t << endl;
    //    cout << "dict_local.size() " << dict_local.size() << ", pointsSize: " << points.size() << endl;
    
    for(int i = t; i < r; i+= NUM_THREADS) {
        //      cout << "i: " << i << ", t: " << t << endl;
        for (int c_iter = 0; c_iter < c; c_iter++) {
            //  cout <<" col: " << c_iter << "row:  " << i << endl;
            
            double dist1 = (double) numeric_limits<int>::max();
            Mat a,b;
            a.create(n,1, CV_32F);
            b.create(n,1, CV_32F);
            //file << "---------------" << endl;
            for (int j = 0; j < n; j++) {
                a.at<float>(j,0) = points[i*c + c_iter].Filters[j];
                //    file << a.at<float>(j,0) << ", ";
            }
            
            int TextonLabel = 0;
            //cout << "dict_local.size() " << dict_local.size() << endl;
            for (int j = 0; j < dict_local.size(); j++) {
                for (int l = 0; l < n; l++) {
                    b.at<float>(l,0) = dict_local[j].Filters[l];
                    //           file << dict_local[j].Filters[l] << ", ";
                }
                // file << endl << " ---------------" << endl;
                double dist2 = computeDistance(a, b);
                
                
                //cout << "thread: " << t <<  ",   dist2:  " << dist2 << ",  dist1: " << dist1 << endl;;
                if (dist2 < dist1){
                    TextonLabel = j;
                    dist1 = dist2;
                }
                //cout << "j:  " << j << ",  Texton label: " << TextonLabel << endl;
                
            }
            //cout << "Texton label: " << TextonLabel << endl;
            textonMapThreadLocal->at<uchar>(i,c_iter) = (int)TextonLabel+1;
            
        }
    }
    
    //   file.close();
    return NULL;
    
}

/* ----- Generate Texton Map for a given Image ----- */
Mat Textons::generateTextonMap(InputArray input_image_) {//, Mat TextonMap) {
    int width = input_image_.cols();
    int height = input_image_.rows();
    TextonMap.create(1,1,CV_8UC3);

    Mat TextonMapLocal(input_image_.rows(), input_image_.cols(), CV_8UC1);
    resize(TextonMap, TextonMap, Size(input_image_.cols(), input_image_.rows()), 0, 0);

    int numThreads = TextonMapLocal.rows*TextonMapLocal.cols;
    pthread_t threads[NUM_THREADS];
    thread_args_t thread_args[NUM_THREADS];
    
    for (int tIter = 0; tIter < NUM_THREADS; tIter++) {
        //create a struct with the required data to send to pthread
        //struct pthread_data *data_ = (struct pthread_data*)malloc(sizeof(struct pthread_data));
        //thread_args_t thread_args;
        thread_args[tIter].tid = (int)tIter;
        //       cout << "tid: " << thread_args[tIter].tid << endl;
        // kCenters.copyTo(thread_args.dictionary_thread);
        //        cout << "thread #:   " << tIter << endl;
        
        thread_args[tIter].rows = TextonMapLocal.rows;
        thread_args[tIter].cols = TextonMapLocal.cols;
        
        thread_args[tIter].pointsThread = &TestImageTextonMap;
        
        thread_args[tIter].numFilters = NumFilters;
        
        thread_args[tIter].textonMapLocal = &TextonMapLocal;
        thread_args[tIter].dictionary_thread = &Dictionary;
        
        pthread_create(&threads[tIter], NULL,  computeTexton, (void *) &thread_args[tIter]);
    }
    
    // wait for each thread to complete
    for (int index = 0; index < NUM_THREADS ; index++) {
        // block until thread 'index' completes
        int result_code = pthread_join(threads[index], NULL);
        // printf("In main: thread %d has completed\n", index);
        assert(0 == result_code);
    }
    int colors[64][3];
    int variant[4] = {0, 85, 170, 255};
    int i = 0;
    for (int p = 0; p < 4; p++) {
        for (int q = 0; q < 4; q++) {
            for (int r = 0; r < 4; r++) {
                colors[i][0] = variant[p];
                colors[i][1] = variant[q];
                colors[i][2] = variant[r];

                int temp1 = colors[i][0];
                int temp2 = colors[i][1];
                int temp3 = colors[i][2];
                i++;
            }
        }
    }

    Mat TextonMapColors(TextonMapLocal.rows, TextonMapLocal.cols, CV_8UC3);
    for (int i = 0; i < TextonMapLocal.rows; i++) {
        for (int j = 0; j < TextonMapLocal.cols; j++) {
            uchar TextonLabel = TextonMapLocal.at<uchar>(i,j);

            TextonMapColors.at<Vec3b>(i,j)[0] = colors[k-TextonLabel-1][0];
            TextonMapColors.at<Vec3b>(i,j)[1] = colors[k-TextonLabel-1][1];
            TextonMapColors.at<Vec3b>(i,j)[2] = colors[k-TextonLabel-1][2];
        }
    }
    
    imshow("textonMapColor", TextonMapColors);
//    waitKey();
    //imwrite("textonMapColored.png", TextonMapColors);
    //imwrite("TextonMap.png", TextonMapLocal);
    //waitKey();
/*    imshow("textonMap", TextonMapLocal);
    waitKey();
*/    
    return TextonMapLocal;
}

