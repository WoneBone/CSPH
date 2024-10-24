#include <stdio.h>
#include <algorithm>
#include <string>
#include <fstream>
#include <ctime>
#include <getopt.h>
#include <CL/sycl.hpp>
#include "CycleTimer.h"

void serialCounting(int* mat, int* bins, int* result, int N, int B){

    // TODO: FILL WITH THE CODE PROVIDED TO YOU!

}


void syclCounting(sycl::queue Queue, int* sycldata, int* bins, int* result, int N, int B, double* total_time){
    sycl::event event = Queue.submit([&](sycl::handler& h){

        // TODO: CREATE YOUR (SYCL PARALLEL_FOR) KERNEL SUBMISSION AND
        // DEVELOP A SYCL VERSION OF THE FIRST STEP OF THE SERIAL CODE PROVIDED ABOVE

    });

    event.wait();

    uint64_t start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
    uint64_t end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
    *total_time = static_cast<double>(end - start) / pow(10,9);

    event = Queue.submit([&](sycl::handler& h){

        // TODO: CREATE YOUR (SYCL PARALLEL_FOR) KERNEL SUBMISSION AND
        // DEVELOP A SYCL VERSION OF THE SECOND STEP OF THE SERIAL CODE PROVIDED ABOVE

    });

    event.wait();

    start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
    end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
    *total_time += static_cast<double>(end - start) / pow(10,9);
}

bool verifyBins(int B, int *gold, int *result) {

    for (int i=0; i<B; i++) {
        if(gold[i] != result[i]){
            printf ("Mismatch Bins[%d]: Expected : %d, Actual : %d\n", i, gold[i], result[i]);
            return 0;
        }
    }
    return 1;
}

bool verifyResult(int gold, int result) {

    if(gold != result){
        printf ("Mismatch : Expected : %d, Actual : %d\n", gold, result);
        return 0;
    }
    
    return 1;
}

void readImage(char* filename, int* serial, int width, int height){
    std::ifstream ppmFile(filename, std::ios::binary);
    std::string magicNumber;
    ppmFile >> magicNumber;

    char ch;
    ppmFile.get(ch);
    while (ch == '#') {
        ppmFile.ignore(10000, '\n');
        ppmFile.get(ch);
    }
    ppmFile.unget();  // Put back the last read character

    int width_, height_, maxColor;
    ppmFile >> width_ >> height_ >> maxColor;
    ppmFile.get();

    // Read binary pixel data
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            unsigned char r, g, b;
            int index = (i * width + j);
            ppmFile.read(reinterpret_cast<char*>(&r), 1);
            ppmFile.read(reinterpret_cast<char*>(&g), 1);
            ppmFile.read(reinterpret_cast<char*>(&b), 1);
            serial[index] = static_cast<int>(b);
        }
    }

    ppmFile.close();
}

int main(int argc, char** argv) {
    if(argc > 3){
        printf("Usage: ./csort filename groupnum");
    }
    
    const int N = 2048;
    int B = 100 + 5*atoi(argv[2]);

    //Memory allocation (serial)
    int* serial = (int*)calloc(N*N, sizeof(int));
    readImage(argv[1], serial, N, N);

    int* bins = (int*)calloc(B, sizeof(int));
    int result = 0;

    //
    // Run the serial implementation. Report the minimum time of three
    // runs for robust timing.
    //
    double minSerial = 1e30;
    for (int i = 0; i < 5; i++) {
        result = 0;
        double startTime = CycleTimer::currentSeconds();
        serialCounting(serial, bins, &result, N, B);
        double endTime = CycleTimer::currentSeconds();
        minSerial = std::min(minSerial, endTime - startTime);

        if(i != 4){
            for(int j = 0; j < B; j++){
                bins[j] = 0;
            }
        }
    }

    printf("[counting serial]:\t\t[%.3f] ms\n", minSerial * 1000);

    //SYCL Queue creation and memory allocation. Note that the selected
    //device is CPU, but you are free to also test on GPU (just be careful
    //with allocations and initializations)
    sycl::property_list prop_list{sycl::property::queue::enable_profiling()};
    auto defaultQueue = sycl::queue{sycl::cpu_selector_v, prop_list};

    std::cout << "Running on: "  << defaultQueue.get_device().get_info<sycl::info::device::name>() << std::endl;

    int* sycldata = sycl::malloc_device<int>(N*N, defaultQueue);
    int* syclbins = sycl::malloc_device<int>(B, defaultQueue);
    int* syclresult = sycl::malloc_device<int>(1, defaultQueue);

    for(int i = 0; i < N*N; i++){
        sycldata[i] = serial[i];
    }

    for(int i = 0; i < B; i++){
        syclbins[i] = 0;
    }

    //
    // Compute the counting sort using the SYCL implementation
    //
    double minSYCL = 1e30, syclTime = 0.0f;
    for (int i = 0; i < 5; i++) {
        syclresult[0] = 0;
        syclCounting(defaultQueue, sycldata, syclbins, syclresult, N, B, &syclTime);
        minSYCL = std::min(minSYCL, syclTime);
        if(i != 4){
            for(int j = 0; j < B; j++){
                syclbins[j] = 0;
            }
        }
    }

    printf("[counting sycl]:\t\t[%.3f] ms\n", minSYCL * 1000);

    if (!verifyBins (B, bins, syclbins)) {
        printf ("Error : SYCL bins differs from sequential bins\n");
    }
    
    if (!verifyResult (result, syclresult[0])) {
        printf ("Error : SYCL output differs from sequential output\n");
    }
    else printf("\t\t\t\t(%.2fx speedup from SYCL)\n", minSerial/minSYCL);


    free(serial);
    free(bins);
    sycl::free(sycldata, defaultQueue);
    sycl::free(syclbins, defaultQueue);
    sycl::free(syclresult, defaultQueue);

    return 0;
}
