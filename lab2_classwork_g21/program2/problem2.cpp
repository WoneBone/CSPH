#include <stdio.h>
#include <algorithm>
#include <string>
#include <fstream>
#include <getopt.h>

#include "CycleTimer.h"
#include "image_ispc.h"

extern void evalSerial(int width, int height, int startRow, int totalRows, int* data, int output[]);

bool verifyResult (int *gold, int *result, int width, int height) {
    int i, j;

    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            if (gold[i * width + j] != result[i * width + j]) {
                printf ("Mismatch : [%d][%d], Expected : %d, Actual : %d\n",
                            i, j, gold[i * width + j], result[i * width + j]);
                return 0;
            }
        }
    }

    return 1;
}

using namespace ispc;

void usage(const char* progname) {
    printf("Usage: %s [options]\n", progname);
    printf("Program Options:\n");
    printf("  -v  --view <INT>   Use specified view settings\n");
    printf("  -?  --help         This message\n");
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

    const unsigned int width = 1600;
    const unsigned int height = 1200;

    if(argc != 2){
        printf("Usage: ./problem2 imagename");
    }

    int* serial = new int[width*height];
    memset(serial, 0, width * height * sizeof(int));

    readImage(argv[1], serial, width, height);
    int useTasks = 1;

    int *output_serial = new int[width*height];
    int *output_ispc = new int[width*height];
    int *output_ispc_tasks = new int[width*height];

    for (unsigned int i = 0; i < width * height; ++i)
        output_serial[i] = 0;

    //
    // Run the serial implementation. Report the minimum time of three
    // runs for robust timing.
    //
    double minSerial = 1e30;
    for (int i = 0; i < 3; ++i) {
        double startTime = CycleTimer::currentSeconds();
        evalSerial(width, height, 0, height, serial, output_serial);
        double endTime = CycleTimer::currentSeconds();
        minSerial = std::min(minSerial, endTime - startTime);
    }

    printf("[image serial]:\t\t[%.3f] ms\n", minSerial * 1000);

    // Clear out the buffer
    for (unsigned int i = 0; i < width * height; ++i)
        output_ispc[i] = 0;

    //
    // Compute the image using the ispc implementation
    //
    double minISPC = 1e30;
    for (int i = 0; i < 3; ++i) {
        double startTime = CycleTimer::currentSeconds();
        image_ispc(width, height, serial, output_ispc);
        double endTime = CycleTimer::currentSeconds();
        minISPC = std::min(minISPC, endTime - startTime);
    }

    printf("[image ispc]:\t\t[%.3f] ms\n", minISPC * 1000);


    if (! verifyResult (output_serial, output_ispc, width, height)) {
        printf ("Error : ISPC output differs from sequential output\n");

        delete[] output_serial;
        delete[] output_ispc;
        delete[] output_ispc_tasks;

        return 1;
    }

    // Clear out the buffer
    for (unsigned int i = 0; i < width * height; ++i) {
        output_ispc_tasks[i] = 0;
    }

    double minTaskISPC = 1e30;
    if (useTasks) {
        //
        // Tasking version of the ISPC code
        //
        for (int i = 0; i < 3; ++i) {
            double startTime = CycleTimer::currentSeconds();
            image_ispc_withtasks(width, height, serial, output_ispc_tasks);
            double endTime = CycleTimer::currentSeconds();
            minTaskISPC = std::min(minTaskISPC, endTime - startTime);
        }

        printf("[image multicore ispc]:\t[%.3f] ms\n", minTaskISPC * 1000);

        if (! verifyResult (output_serial, output_ispc_tasks, width, height)) {
            printf ("Error : ISPC output differs from sequential output\n");
            return 1;
        }
    }

    printf("\t\t\t\t(%.2fx speedup from ISPC)\n", minSerial/minISPC);
    if (useTasks) {
        printf("\t\t\t\t(%.2fx speedup from task ISPC)\n", minSerial/minTaskISPC);
    }

    delete[] output_serial;
    delete[] output_ispc;
    delete[] output_ispc_tasks;


    return 0;
}
