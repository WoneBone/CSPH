#include <stdio.h>
#include <algorithm>
#include <pthread.h>
#include <math.h>

#include "CycleTimer.h"
#include "eval_ispc.h"

using namespace ispc;

extern void evalSerial(int N, float* values, float* output);

static void verifyResult(int N, float* result, float* gold) {
    for (int i=0; i<N; i++) {
        if (fabs(result[i] - gold[i]) > 1e-4) {
            printf("Error: [%d] Got %f expected %f\n", i, result[i], gold[i]);
        }
    }
}

void initializeGold(int groupNum, float values[], float gold[], int N){
    for ( int i=0; i<N; i++){
        if(groupNum%6 == 0) gold[i] = 50*(abs(sin(sqrt(3.65)*(values[i] - groupNum)*(1.48)))/(double)(pow(0.1*values[i] + 1.48, 2) + 1));
        else gold[i] = 50*(abs(sin(sqrt(3.65)*(values[i] - groupNum)*(1.24 + (groupNum%6)*0.04)))/(double)(pow(0.1*values[i] + 1.24 + (groupNum%6)*0.04, 2) + 1));
    }
}

int main(int argc, char* argv[]) {

    const unsigned int N = 20 * 1000 * 1000;

    float* values = new float[N];
    float* output_serial = new float[N];
    float* output = new float[N];
    float* gold = new float[N];

    int groupNum = atoi(argv[1]);

    for (unsigned int i=0; i<N; i++)
    {
        // TODO FOR STUDENTS: Attempt to change the values in the
        // array here to meet the instructions in the handout: we want
        // to you generate best and worse-case speedups
        
        // starter code populates array with random input values
        values[i] = (-14.75f);
    }

    initializeGold(groupNum, values, gold, N);

    //
    // And run the serial implementation 3 times, again reporting the
    // minimum time.
    //
    double minSerial = 1e30;
    for (int i = 0; i < 3; ++i) {
        double startTime = CycleTimer::currentSeconds();
        evalSerial(N, gold, output_serial);
        double endTime = CycleTimer::currentSeconds();
        minSerial = std::min(minSerial, endTime - startTime);
    }

    printf("[computation serial]:\t\t[%.3f] ms\n", minSerial * 1000);

    //
    // Compute the image using the ispc implementation; report the minimum
    // time of three runs.
    //
    double minISPC = 1e30;
    for (int i = 0; i < 3; ++i) {
        double startTime = CycleTimer::currentSeconds();
        eval_ispc(N, gold, output);
        double endTime = CycleTimer::currentSeconds();
        minISPC = std::min(minISPC, endTime - startTime);
    }

    printf("[computation ispc]:\t\t[%.3f] ms\n", minISPC * 1000);

    verifyResult(N, output, output_serial);

    // Clear out the buffer
    for (unsigned int i = 0; i < N; ++i)
        output[i] = 0;

    //
    // Tasking version of the ISPC code
    //
    double minTaskISPC = 1e30;
    for (int i = 0; i < 3; ++i) {
        double startTime = CycleTimer::currentSeconds();
        eval_ispc_withtasks(N, gold, output);
        double endTime = CycleTimer::currentSeconds();
        minTaskISPC = std::min(minTaskISPC, endTime - startTime);
    }

    printf("[computation task ispc]:\t[%.3f] ms\n", minTaskISPC * 1000);

    verifyResult(N, output, output_serial);

    printf("\t\t\t\t(%.2fx speedup from ISPC)\n", minSerial/minISPC);
    printf("\t\t\t\t(%.2fx speedup from task ISPC)\n", minSerial/minTaskISPC);

    delete [] values;
    delete [] output_serial;
    delete [] output;
    delete [] gold;

    return 0;
}
