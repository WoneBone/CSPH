#include <stdio.h>
#include <algorithm>
#include <string>
#include <fstream>
#include <ctime>
#include <getopt.h>
#include <CL/sycl.hpp>
#include "CycleTimer.h"

void serialCovMat(double** data, double** cov, double* means, int N, int D){
    double mean = 0.0f;

    //Center data
    for(int i = 0; i < D; i++){
        mean = 0.0f;
        for(int j = 0; j < N; j++){
            means[i] += data[j][i];
        }

        means[i] /= (double)N;

        for(int j = 0; j < N; j++){
            data[j][i] -= means[i];
        }
    }

    //Calculate Covariance Matrix
    for(int i = 0; i < D; i++){
        for(int j = 0; j < D; j++){
            for(int k = 0; k < N; k++){
                cov[i][j] += (data[k][i])*(data[k][j]);
            }
            cov[i][j] /= (double)(N - 1);
        }
    }
}

void syclCovMatSingle(sycl::queue Queue, double** data, double** cov, double* means, int N, int D, double* total_time){
    //COMPLETE CODE FOR SYCL KERNEL 
    return;
}

void syclCovMatTwo(sycl::queue Queue, double** data, double** cov, double* means, int N, int D, double* total_time){
    //COMPLETE CODE FOR SYCL KERNEL
    return;
}

bool verifyResult(double** gold, double** result, int D) {

    for(int i = 0; i < D; i++){
        for(int j = 0; j < D; j++){
            if(gold[i][j] - result[i][j] > 0.001){
                printf ("Mismatch [%d][%d]: Expected : %.6f, Actual : %.6f\n", i, j, gold[i][j], result[i][j]);
                return 0;
            }
        }
    }

    return 1;
}

int main(int argc, char** argv) {
    int N = 0, D = 0;
    if(argc > 3){
        printf("Usage: ./covmat N D");
    }
    else if (argc == 3){
        N = atoi(argv[1]);
        D = atoi(argv[2]);
    }
    else if (argc == 2){
        N = atoi(argv[1]);
        D = 1024;
    }
    else if (argc == 1){
        N = 1024;
        D = 1024;
    }

    srand(time(NULL));

    //Memory allocation (serial)
    double** serial = (double**)calloc(N, sizeof(double*));
    double** serialCov = (double**)calloc(D, sizeof(double*));
    double* serialMeans = (double*)calloc(D, sizeof(double));
    double serialNorm = 0.0f;

    for(int i = 0; i < N; i++){
        serial[i] = (double*)calloc(D, sizeof(double));
        for(int j = 0; j < D; j++) serial[i][j] = (rand()%10000)/(double)10000;
    }

    for(int i = 0; i < D; i++){
        serialCov[i] = (double*)calloc(D, sizeof(double));
    }

    //
    // Run the serial implementation. Report the minimum time of five
    // runs for robust timing.
    //
    double minSerial = 1e30;
    for (int i = 0; i < 5; ++i) {
        double startTime = CycleTimer::currentSeconds();
        serialCovMat(serial, serialCov, serialMeans, N, D);
        double endTime = CycleTimer::currentSeconds();
        minSerial = std::min(minSerial, endTime - startTime);
    }

    printf("[covariance matrix serial]:\t\t[%.3f] ms\n", minSerial * 1000);

    //SYCL Queue creation and memory allocation. Note that the selected
    //device is CPU, but you are free to also test on GPU (just be careful
    //with allocations and initializations)

    sycl::property_list prop_list{sycl::property::queue::enable_profiling()};
    auto defaultQueue = sycl::queue{sycl::cpu_selector_v, prop_list};

    std::cout << "Running on: "  << defaultQueue.get_device().get_info<sycl::info::device::name>() << std::endl;

    double** sycldata = sycl::malloc_device<double*>(N, defaultQueue);
    double** syclCov = sycl::malloc_device<double*>(D, defaultQueue);
    double* syclMeans = sycl::malloc_device<double>(D, defaultQueue);

    for(int i = 0; i < N; i++){
        sycldata[i] = sycl::malloc_device<double>(D, defaultQueue);
        for(int j = 0; j < D; j++){
            sycldata[i][j] = serial[i][j];
        }
    }

    for(int i = 0; i < D; i++){
        syclCov[i] = sycl::malloc_device<double>(D, defaultQueue);
        for(int j = 0; j < D; j++){
            syclCov[i][j] = 0;
        }
        syclMeans[i] = 0;
    }

    //
    // Compute the Covariance Matrix using the SYCL implementation
    //
    double minSYCL = 1e30, syclTime = 0.0f;
    for (int i = 0; i < 5; ++i) {
        syclCovMatSingle(defaultQueue, sycldata, syclCov, syclMeans, N, D, &syclTime);
        minSYCL = std::min(minSYCL, syclTime);
    }

    printf("[covariance matrix sycl - one kernel]:\t\t[%.3f] ms\n", minSYCL*1000);

    if (!verifyResult (serialCov, syclCov, D)) {
        printf ("Error : SYCL output differs from sequential output\n");
    }
    else printf("\t\t\t\t(%.2fx speedup from SYCL)\n", minSerial/minSYCL);

    for(int i = 0; i < D; i++){
        for(int j = 0; j < D; j++){
            syclCov[i][j] = 0;
        }
        syclMeans[i] = 0;
    }

    minSYCL = 1e30, syclTime = 0.0f;
    for (int i = 0; i < 5; ++i) {
        syclCovMatTwo(defaultQueue, sycldata, syclCov, syclMeans, N, D, &syclTime);
        minSYCL = std::min(minSYCL, syclTime);
    }

    printf("[covariance matrix sycl - two kernels]:\t\t[%.3f] ms\n", minSYCL*1000);

    if (!verifyResult (serialCov, syclCov, D)) {
        printf ("Error : SYCL output differs from sequential output\n");
    }
    else printf("\t\t\t\t(%.2fx speedup from SYCL)\n", minSerial/minSYCL);

    for(int i = 0; i < N; i++){
        free(serial[i]);
        sycl::free(sycldata[i], defaultQueue);
    }

    for(int i = 0; i < D; i++){
        free(serialCov[i]);
        sycl::free(syclCov[i], defaultQueue);
    }

    free(serial);
    free(serialCov);
    free(serialMeans);
    sycl::free(sycldata, defaultQueue);
    sycl::free(syclCov, defaultQueue);
    sycl::free(syclMeans, defaultQueue);

    return 0;
}
