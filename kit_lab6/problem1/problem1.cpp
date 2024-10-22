#include <stdio.h>
#include <algorithm>
#include <string>
#include <fstream>
#include <ctime>
#include <getopt.h>
#include <CL/sycl.hpp>
#include "CycleTimer.h"

double serialFrobenius(double** mat, int N){
    double norm = 0;

    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++)
            norm += mat[i][j]*mat[i][j];
    }

    return sqrt(norm);
}

void syclFrobenius(sycl::queue Queue, double** syclmat, int N, double* total_time, double* sumfrobenius){
    sycl::event event = Queue.submit([&](sycl::handler& h){

        h.parallel_for(sycl::nd_range<2>(sycl::range(std::min(N, 8192), std::min(N, 8192)),sycl::range(32,32)),  
                    [=](sycl::nd_item<2> item) {
            int x = item.get_global_id(0), y = item.get_global_id(1);
            sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> atomic_global_bin(sumfrobenius[0]);
            
            for(int i = x; i < N; i += item.get_global_range(0)){
                for(int j = y; j < N; j += item.get_global_range(1)){
                    atomic_global_bin.fetch_add(syclmat[i][j]*syclmat[i][j]);
                }
            }
        });
    });

    event.wait();

    uint64_t start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
    uint64_t end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
    *total_time = static_cast<double>(end - start) / pow(10,9);

    sumfrobenius[0] = sqrt(sumfrobenius[0]);
}


bool verifyResult(double gold, double result) {

    if(gold - result > 0.0001){
        printf ("Mismatch : Expected : %.6f, Actual : %.6f\n", gold, result);
        return 0;
    }
    return 1;
}

int main(int argc, char** argv) {
    int N = 0;
    if(argc > 2){
        printf("Usage: ./frobenius N");
    }
    else if (argc == 2) N = atoi(argv[1]);
    else if (argc == 1) N = 1024;

    srand(time(NULL));

    //Memory allocation (serial)
    double** serial = (double**)calloc(N, sizeof(double*));
    double serialNorm = 0.0f;

    for(int i = 0; i < N; i++){
        serial[i] = (double*)calloc(N, sizeof(double));
        for(int j = 0; j < N; j++) serial[i][j] = (rand()%10000)/(double)10000;
    }

    //
    // Run the serial implementation. Report the minimum time of three
    // runs for robust timing.
    //
    double minSerial = 1e30;
    for (int i = 0; i < 5; ++i) {
        double startTime = CycleTimer::currentSeconds();
        serialNorm = serialFrobenius(serial, N);
        double endTime = CycleTimer::currentSeconds();
        minSerial = std::min(minSerial, endTime - startTime);
    }

    printf("[frobenius norm serial]:\t\t[%.3f] ms\n", minSerial * 1000);

    //SYCL Queue creation and memory allocation. Note that the selected
    //device is CPU, but you are free to also test on GPU (just be careful
    //with allocations and initializations)
    sycl::property_list prop_list{sycl::property::queue::enable_profiling()};
    auto defaultQueue = sycl::queue{sycl::cpu_selector_v, prop_list};

    std::cout << "Running on: "  << defaultQueue.get_device().get_info<sycl::info::device::name>() << std::endl;

    double** syclmat = sycl::malloc_device<double*>(N, defaultQueue);

    for(int i = 0; i < N; i++){
        syclmat[i] = sycl::malloc_device<double>(N, defaultQueue);
        for(int j = 0; j < N; j++) syclmat[i][j] = serial[i][j];
    }

    double* syclNorm = sycl::malloc_device<double>(1, defaultQueue);
    syclNorm[0] = 0.0f;

    //
    // Compute the Frobenius Norm using the SYCL implementation
    //
    double minSYCL = 1e30, syclTime = 0.0f;
    for (int i = 0; i < 5; ++i) {
        syclFrobenius(defaultQueue, syclmat, N, &syclTime, syclNorm);
        minSYCL = std::min(minSYCL, syclTime);
    }

    printf("[frobenius norm sycl]:\t\t[%.3f] ms\n", minSYCL * 1000);

    if (!verifyResult (serialNorm, syclNorm[0])) {
        printf ("Error : SYCL output differs from sequential output\n");
    }
    else printf("\t\t\t\t(%.2fx speedup from SYCL)\n", minSerial/minSYCL);

    for(int i = 0; i < N; i++){
        free(serial[i]);
        sycl::free(syclmat[i], defaultQueue);
    }

    free(serial);
    sycl::free(syclmat, defaultQueue);
    sycl::free(syclNorm, defaultQueue);

    return 0;
}
