#include <stdio.h>
#include <algorithm>
#include <string>
#include <fstream>
#include <ctime>
#include <getopt.h>
#include <CL/sycl.hpp>
#include "CycleTimer.h"

void serialHistogram(double* mat, int* bins, int B, int N){
    double norm = 0;
    int j = 0;

    for(int i = 0; i < N; i++){
        j = int(mat[i]*B);
        bins[j] += 1;
    }
}

void syclHistogram(sycl::queue Queue, double* sycldata, int* bins, int N, int B, double* total_time){
    //COMPLETE CODE FOR SYCL KERNEL
    return;
}

bool verifyResult(int* gold, int* result, int B) {

    for(int i = 0; i < B; i++){
        if(gold[i] != result[i]){
            printf ("Mismatch : [%d] , Expected : %d, Actual : %d\n", i, gold[i], result[i]);
            return 0;
        }
    }
    return 1;
}

int main(int argc, char** argv) {
    int N = 0, B = 0;
    if(argc > 3){
        printf("Usage: ./histogram N B");
    }
    else if (argc == 3){
        N = atoi(argv[1]);
        B = atoi(argv[2]);
    }
    else if (argc == 2){
        N = atoi(argv[1]);
        B = 10;
    }
    else if (argc == 1){
        N = 1024*1024;
        B = 10;
    }

    srand(time(NULL));

    //Memory allocation (serial)
    double* serial = (double*)calloc(N, sizeof(double));

    for(int i = 0; i < N; i++){
        serial[i] = (rand()%10000)/(double)10000;
    }

    int* bins = (int*)calloc(B, sizeof(int));

    //
    // Run the serial implementation. Report the minimum time of three
    // runs for robust timing.
    //
    double minSerial = 1e30;
    for (int i = 0; i < 5; ++i) {
        double startTime = CycleTimer::currentSeconds();
        serialHistogram(serial, bins, B, N);
        double endTime = CycleTimer::currentSeconds();
        minSerial = std::min(minSerial, endTime - startTime);
    }

    printf("[histogram serial]:\t\t[%.3f] ms\n", minSerial * 1000);

    //SYCL Queue creation and memory allocation. Note that the selected
    //device is CPU, but you are free to also test on GPU (just be careful
    //with allocations and initializations)
    sycl::property_list prop_list{sycl::property::queue::enable_profiling()};
    auto defaultQueue = sycl::queue{sycl::cpu_selector_v, prop_list};

    std::cout << "Running on: "  << defaultQueue.get_device().get_info<sycl::info::device::name>() << std::endl;

    double* sycldata = sycl::malloc_device<double>(N, defaultQueue);
    int* syclbins = sycl::malloc_device<int>(B, defaultQueue);

    for(int i = 0; i < N; i++){
        sycldata[i] = serial[i];
    }

    for(int i = 0; i < B; i++){
        syclbins[i] = 0;
    }

    //
    // Compute the histogram using the SYCL implementation
    //
    double minSYCL = 1e30, syclTime = 0.0f;
    for (int i = 0; i < 5; ++i) {
        syclHistogram(defaultQueue, sycldata, syclbins, N, B, &syclTime);
        minSYCL = std::min(minSYCL, syclTime);
    }

    printf("[histogram sycl]:\t\t[%.3f] ms\n", minSYCL * 1000);

    if (!verifyResult (bins, syclbins, B)) {
        printf ("Error : SYCL output differs from sequential output\n");
    }
    else printf("\t\t\t\t(%.2fx speedup from SYCL)\n", minSerial/minSYCL);

    free(serial);
    free(bins);
    sycl::free(sycldata, defaultQueue);
    sycl::free(syclbins, defaultQueue);

    return 0;
}
