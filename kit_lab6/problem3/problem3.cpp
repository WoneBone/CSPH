#include <stdio.h>
#include <algorithm>
#include <string>
#include <fstream>
#include <ctime>
#include <getopt.h>
#include <CL/sycl.hpp>
#include "CycleTimer.h"

struct Point {
  double x;
  double y;
  int cent_idx;
};

void serialKMeans(Point* points, double** cents, int N, int C){
    double norm = 0, min_norm = 1000000;
    int j = 0;

    //Calculate for each point the closest centroid
    for(int i = 0; i < N; i++){
        min_norm = 1000000;
        for(int j = 0; j < C; j++){
            norm = sqrt((points[i].x - cents[j][0])*(points[i].x - cents[j][0]) + (points[i].y - cents[j][1])*(points[i].y - cents[j][1]));
            if(norm < min_norm){
                min_norm = norm;
                points[i].cent_idx = j;
            }
        }   
    }

    //Update Centroids
    double x = 0.0f, y = 0.0f;
    int count = 0;
    for(int i = 0; i < C; i++){
        x = y = 0.0f;
        count = 0;
        for(int j = 0; j < N; j++){
            if(points[j].cent_idx == i){
                x += points[j].x;
                y += points[j].y;
                count++;
            }
        }

        cents[i][0] = x/(double)count;
        cents[i][1] = y/(double)count;
    }
}

void syclKMeans(sycl::queue Queue, Point* points, double** cents, int N, int C, double* total_time){
    //COMPLETE CODE FOR SYCL KERNEL
    return;
}

bool verifyResult(double** gold, double** result, int C) {

    for(int i = 0; i < C; i++){
        if(gold[i][0] - result[i][0] > 0.001){
            printf ("Mismatch [%d][%d] : Expected : %.6f, Actual : %.6f\n", i, 0, gold[i][0], result[i][0]);
            return 0;
        }

        if(gold[i][1] - result[i][1] > 0.001){
            printf ("Mismatch [%d][%d] : Expected : %.6f, Actual : %.6f\n", i, 1, gold[i][1], result[i][1]);
            return 0;
        }
    }

    return 1;
}

int main(int argc, char** argv) {
    int N = 0, C = 0;
    if(argc > 3){
        printf("Usage: ./kmeans N C");
    }
    else if (argc == 3){
        N = atoi(argv[1]);
        C = atoi(argv[2]);
    }
    else if (argc == 2){
        N = atoi(argv[1]);
        C = 10;
    }
    else if (argc == 1){
        N = 1024;
        C = 10;
    }

    srand(time(NULL));

    //Memory allocation (serial)
    Point* serial = (Point*)calloc(N, sizeof(Point));
    double** serialCents = (double**)calloc(C, sizeof(double));

    for(int i = 0; i < C; i++){
        serialCents[i] = (double*)calloc(2, sizeof(double));
        serialCents[i][0] = (rand()%10000)/(double)100;
        serialCents[i][1] = (rand()%10000)/(double)100;
    }

    for(int i = 0; i < N; i++){
        serial[i].x = (rand()%10000)/(double)100;
        serial[i].y = (rand()%10000)/(double)100;
    }

    //SYCL Queue creation and memory allocation. Note that the selected
    //device is CPU, but you are free to also test on GPU (just be careful
    //with allocations and initializations)
    sycl::property_list prop_list{sycl::property::queue::enable_profiling()};
    auto defaultQueue = sycl::queue{sycl::cpu_selector_v, prop_list};

    std::cout << "Running on: "  << defaultQueue.get_device().get_info<sycl::info::device::name>() << std::endl;

    Point* syclPoints = sycl::malloc_device<Point>(N, defaultQueue);
    double** syclCents = sycl::malloc_device<double*>(C, defaultQueue);

    for(int i = 0; i < C; i++){
        syclCents[i] = sycl::malloc_device<double>(2, defaultQueue);
        syclCents[i][0] = serialCents[i][0];
        syclCents[i][1] = serialCents[i][1];
    }

    for(int i = 0; i < N; i++){
        syclPoints[i].x = serial[i].x;
        syclPoints[i].y = serial[i].y;
        syclPoints[i].cent_idx = 0;
    }

    //
    // Run the serial implementation. Report the minimum time of three
    // runs for robust timing.
    //
    double minSerial = 1e30;
    for (int i = 0; i < 5; ++i) {
        double startTime = CycleTimer::currentSeconds();
        serialKMeans(serial, serialCents, N, C);
        double endTime = CycleTimer::currentSeconds();
        minSerial = std::min(minSerial, endTime - startTime);
    }

    printf("[K means serial]:\t\t[%.3f] ms\n", minSerial * 1000);

    //
    // Compute the K-Means Clustering using the SYCL implementation
    //
    double minSYCL = 1e30, syclTime;
    for (int i = 0; i < 5; ++i) {
        syclKMeans(defaultQueue, syclPoints, syclCents, N, C, &syclTime);
        minSYCL = std::min(minSYCL, syclTime);
    }

    printf("[K means sycl]:\t\t[%.3f] ms\n", minSYCL * 1000);

    if (!verifyResult (serialCents, syclCents, C)) {
        printf ("Error : SYCL output differs from sequential output\n");
    }
    else printf("\t\t\t\t(%.2fx speedup from SYCL)\n", minSerial/minSYCL);

    for(int i = 0; i < C; i++){
        free(serialCents[i]);
        sycl::free(syclCents[i], defaultQueue);
    }

    free(serial);
    free(serialCents);
    sycl::free(syclCents, defaultQueue);
    sycl::free(syclPoints, defaultQueue);

    return 0;
}
