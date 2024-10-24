#include <stdio.h>
#include <algorithm>
#include <string>
#include <fstream>
#include <ctime>
#include <getopt.h>
#include <CL/sycl.hpp>
#include "CycleTimer.h"

void serialImage(int** mat, int N, const int val){

    // TODO: FILL WITH THE CODE PROVIDED TO YOU!

}

void syclImage(sycl::queue Queue, int** syclmat, int N, int val, double* total_time){
    sycl::event event = Queue.submit([&](sycl::handler& h){

        // TODO: CREATE YOUR (SYCL PARALLEL_FOR) KERNEL SUBMISSION AND
        // DEVELOP A SYCL VERSION OF THE SERIAL CODE PROVIDED ABOVE
    });

    event.wait();

    uint64_t start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
    uint64_t end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
    *total_time = static_cast<double>(end - start) / pow(10,9);
}


bool verifyResult(int** gold, int** result, int N) {

    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            if(gold[i][j] != result[i][j]){
                printf ("Mismatch [%d][%d]: Expected : %d, Actual : %d\n", i, j, gold[i][j], result[i][j]);
                return 0;
            }
        }
    }
    
    return 1;
}

void writePPMImage(int** data, int N, const char *filename)
{
    FILE *fp = fopen(filename, "wb");

    // write ppm header
    fprintf(fp, "P6\n");
    fprintf(fp, "%d %d\n", N, N);
    fprintf(fp, "255\n");

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++){
            unsigned char result = static_cast<unsigned char>(data[i][j]);
            for (int k = 0; k < 3; k++)
                fputc(result, fp);
        }
    }
    fclose(fp);
    printf("Wrote image file %s\n", filename);
}

int main(int argc, char** argv) {
    if(argc > 3){
        printf("Usage: ./image N val");
    }
    
    int N = atoi(argv[1]);
    int interval = atoi(argv[2]);

    //Memory allocation (serial)
    int** serial = (int**)calloc(N, sizeof(int*));

    for(int i = 0; i < N; i++){
        serial[i] = (int*)calloc(N, sizeof(int));
    }

    //
    // Run the serial implementation. Report the minimum time of three
    // runs for robust timing.
    //
    double minSerial = 1e30;
    for (int i = 0; i < 5; ++i) {
        double startTime = CycleTimer::currentSeconds();
        serialImage(serial, N, interval);
        double endTime = CycleTimer::currentSeconds();
        minSerial = std::min(minSerial, endTime - startTime);
    }

    printf("[Image serial]:\t\t[%.3f] ms\n", minSerial * 1000);
    writePPMImage(serial, N, "serial.ppm");

    //SYCL Queue creation and memory allocation. Note that the selected
    //device is CPU, but you are free to also test on GPU (just be careful
    //with allocations and initializations)
    sycl::property_list prop_list{sycl::property::queue::enable_profiling()};
    auto defaultQueue = sycl::queue{sycl::cpu_selector_v, prop_list};

    std::cout << "Running on: "  << defaultQueue.get_device().get_info<sycl::info::device::name>() << std::endl;

    int** syclmat = sycl::malloc_device<int*>(N, defaultQueue);

    for(int i = 0; i < N; i++){
        syclmat[i] = sycl::malloc_device<int>(N, defaultQueue);
        for(int j = 0; j < N; j++) syclmat[i][j] = serial[i][j];
    }

    //
    // Compute the image creation using the SYCL implementation
    //
    double minSYCL = 1e30, syclTime = 0.0f;
    for (int i = 0; i < 5; ++i) {
        syclImage(defaultQueue, syclmat, N, interval, &syclTime);
        minSYCL = std::min(minSYCL, syclTime);
    }

    printf("[Image sycl]:\t\t[%.3f] ms\n", minSYCL * 1000);

    if (!verifyResult (serial, syclmat, N)) {
        printf ("Error : SYCL output differs from sequential output\n");
    }
    else printf("\t\t\t\t(%.2fx speedup from SYCL)\n", minSerial/minSYCL);

    writePPMImage(syclmat, N, "output.ppm");

    for(int i = 0; i < N; i++){
        free(serial[i]);
        sycl::free(syclmat[i], defaultQueue);
    }

    free(serial);
    sycl::free(syclmat, defaultQueue);

    return 0;
}
