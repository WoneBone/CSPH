#include <stdio.h>
#include <algorithm>
#include <string>
#include <fstream>
#include <ctime>
#include <getopt.h>
#include <CL/sycl.hpp>
#include "CycleTimer.h"

void serialDistance(int** data, int* array, float** dist, float* red, float* res, int N){

    // TODO: FILL WITH THE CODE PROVIDED TO YOU!
    for(int i=0 ; i<N ; i++){

        for(int j=0; j<N; j++ ){
            if(data[i][j]==array[j])
                dist[i][j]=array[j]*3;
        }
    }
    for(int i=0 ; i<N; i++){
        red[i]=0;
        for(int j=0;j<N;j++){
            red[i]+=dist[i][j];
        }
    }

    *res=0;
    for(int i= 0; i<N;i++){
        if(red[i]>*res)
            *res=red[i];
    }

}

void syclDistance(sycl::queue Queue, int** data, int* array, float** dist, float* red, float* res, int N, double* total_time){
    sycl::event event;
     event = Queue.submit([&](sycl::handler& h){
        h.parallel_for(sycl::nd_range<2>(sycl::range(std::min(N, 1024), std::min(N, 1024)),sycl::range(32,32)), [=](sycl::nd_item<2>item){
            int x = item.get_global_id(0), y = item.get_global_id(1);
            for(int i=x ; i<N ; i+= item.get_global_range(0)){
                for(int j=y; j<N; j+= item.get_global_range(1) ){
                    if(data[i][j] == array[j]){
                       dist[i][j] = array[j]*3;
                    } 
                } 
            }  
        });
       
    });

    event.wait(); 

    uint64_t start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
    uint64_t end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
    *total_time = static_cast<double>(end - start) / pow(10,9);

    float* syclRed = sycl::malloc_shared<float>(1, Queue);
    event = Queue.submit([&](sycl::handler& h){
        // TODO: CREATE YOUR (SYCL PARALLEL_FOR) KERNEL SUBMISSION AND
        // DEVELOP A SYCL VERSION OF THE SECOND STEP OF THE SERIAL CODE PROVIDED ABOVE
         h.parallel_for(sycl::nd_range<2>(sycl::range(std::min(N, 1024), std::min(N, 1024)),sycl::range(32,32)),sycl::reduction(red, 0.0f,  sycl::plus<>()), [=](sycl::nd_item<2>item,auto& syclNorm){
            int x = item.get_global_id(0), y = item.get_global_id(1);
            for(int i=x ; i<N; i+= item.get_global_range(0)){
                syclRed[i]=0;
                for(int j=y;j<N;j+= item.get_global_range(1)){
                    syclRed[i] += dist[i][j];
                }
            } 
        });
    });

    event.wait();

    start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
    end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
    *total_time += static_cast<double>(end - start) / pow(10,9);

    event = Queue.submit([&](sycl::handler& h){
        // TODO: CREATE YOUR (SYCL PARALLEL_FOR) KERNEL SUBMISSION AND
        // DEVELOP A SYCL VERSION OF THE THIRD STEP OF THE SERIAL CODE PROVIDED ABOVE
        h.parallel_for(sycl::nd_range<2>(sycl::range(std::min(N, 1024), std::min(N, 1024)),sycl::range(32,32)), [=](sycl::nd_item<2>item){
            int x = item.get_global_id(0);
            *res=0;
            for(int i= x; i<N;i+= item.get_global_range(0)){
                if(red[i]>*res)
                    *res=red[i];
            } 
        });
    });

    event.wait();

    start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
    end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
    *total_time += static_cast<double>(end - start) / pow(10,9);
}

bool verifyDistance(float** gold, float** result, int N) {
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            if(abs(gold[i][j] - result[i][j]) > 1){
                printf ("Distance Matrix Mismatch: [%d][%d] Expected : %.6f, Actual : %.6f\n", i, j, gold[i][j], result[i][j]);
                return 0;
            }
        }
    }
    
    return 1;
}

bool verifyReduction(float* gold, float* result, int N) {

    for(int j = 0; j < N; j++){
        if(abs(gold[j] - result[j]) > 1){
            printf ("First Reduction Mismatch: [%d] Expected : %.6f, Actual : %.6f\n", j, gold[j], result[j]);
            return 0;
        }
    }

    return 1;
}

bool verifyResult(float gold, float result) {

    if(abs(gold - result) > 1){
        printf ("Second Reduction Mismatch: Expected : %.6f, Actual : %.6f\n", gold, result);
        return 0;
    }

    return 1;
}

void readImage(char* filename, int** serial, int width, int height){
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
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            unsigned char r, g, b;
            int index = (i * width + j);
            ppmFile.read(reinterpret_cast<char*>(&r), 1);
            ppmFile.read(reinterpret_cast<char*>(&g), 1);
            ppmFile.read(reinterpret_cast<char*>(&b), 1);
            serial[i][j] = static_cast<int>(b);
        }
    }

    ppmFile.close();
}

int main(int argc, char** argv) {
    if(argc > 3){
        printf("Usage: ./pairwise filename groupnum");
    }

    const int N = 2048;
    int groupnum = atoi(argv[2]);
    srand(groupnum);


    //Memory allocation (serial)
    int** serial = (int**)calloc(N, sizeof(int*));
    int* serialArray = (int*)calloc(N, sizeof(int));
    float** serialDist = (float**)calloc(N, sizeof(float*));
    float* serialRed = (float*)calloc(N, sizeof(float));
    float serialRes = 0;

    for(int i = 0; i < N; i++){
        serial[i] = (int*)calloc(N, sizeof(int));
        serialDist[i] = (float*)calloc(N, sizeof(float));
        serialArray[i] = rand()%256;
    }

    readImage(argv[1], serial, N, N);

    //SYCL Queue creation and memory allocation. Note that the selected
    //device is CPU, but you are free to also test on GPU (just be careful
    //with allocations and initializations)
    sycl::property_list prop_list{sycl::property::queue::enable_profiling()};
    auto defaultQueue = sycl::queue{sycl::cpu_selector_v, prop_list};

    std::cout << "Running on: "  << defaultQueue.get_device().get_info<sycl::info::device::name>() << std::endl;

    int** syclPoints = sycl::malloc_device<int*>(N, defaultQueue);
    int* syclArray = sycl::malloc_device<int>(N, defaultQueue);
    float** syclDist = sycl::malloc_device<float*>(N, defaultQueue);
    float* syclRed = sycl::malloc_device<float>(N, defaultQueue);
    float* syclRes = sycl::malloc_device<float>(1, defaultQueue);

    for(int i = 0; i < N; i++){
        syclDist[i] = sycl::malloc_device<float>(N, defaultQueue);
        syclPoints[i] = sycl::malloc_device<int>(N, defaultQueue);
        syclRed[i] = 0;
        syclArray[i] = serialArray[i];

        for(int j = 0; j < N; j++){
            syclDist[i][j] = 0;
            syclPoints[i][j] = serial[i][j];
        }
    }

    //
    // Run the serial implementation. Report the minimum time of three
    // runs for robust timing.
    //
    double minSerial = 1e30;
    for (int i = 0; i < 5; ++i) {
        double startTime = CycleTimer::currentSeconds();
        serialDistance(serial, serialArray, serialDist, serialRed, &serialRes, N);
        double endTime = CycleTimer::currentSeconds();
        minSerial = std::min(minSerial, endTime - startTime);
    }

    printf("[Distance serial]:\t\t[%.3f] ms\n", minSerial * 1000);

    //
    // Compute the Distance matrix using the SYCL implementation
    //
    double minSYCL = 1e30, syclTime = 0;
    for (int i = 0; i < 5; ++i) {
        syclRes[0] = 0.f;
        syclDistance(defaultQueue, syclPoints, syclArray, syclDist, syclRed, syclRes, N, &syclTime);
        minSYCL = std::min(minSYCL, syclTime);
    }

    printf("[Distance sycl]:\t\t[%.3f] ms\n", minSYCL * 1000);

    if (!verifyDistance (serialDist, syclDist, N)) {
        printf ("Error in distance matrix: SYCL output differs from sequential output\n");
    }

    if (!verifyReduction (serialRed, syclRed, N)) {
        printf ("Error in first reduction: SYCL output differs from sequential output\n");
    }

    if (!verifyResult (serialRes, syclRes[0])) {
        printf ("Error in second reduction: SYCL output differs from sequential output\n");
    }
    else printf("\t\t\t\t(%.2fx speedup from SYCL)\n", minSerial/minSYCL);

    for(int i = 0; i < N; i++){
        free(serial[i]);
        free(serialDist[i]);
        sycl::free(syclDist[i], defaultQueue);
        sycl::free(syclPoints[i], defaultQueue);
    }

    free(serial);
    free(serialDist);
    free(serialRed);
    free(serialArray);
    sycl::free(syclArray, defaultQueue);
    sycl::free(syclRes, defaultQueue);
    sycl::free(syclRed, defaultQueue);
    sycl::free(syclDist, defaultQueue);
    sycl::free(syclPoints, defaultQueue);

    return 0;
}
