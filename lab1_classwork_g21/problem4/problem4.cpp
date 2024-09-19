#include "fake_intrinsics.h"
#include <string.h>

int intrinsics_scalar(int N, float* A, float* B, float* C){
    //To implement
    return 0;
}

int intrinsics_simd(int N, float* A, float* B, float* C){
    //To implement 
    return 0;
}

void fillA(float* A){
    for(int i = 0; i < 32; i++) A[i] = i;
    for(int i = 0; i < 32; i++) A[32+i] = 31-i;
    for(int i = 64; i < N; i++) A[i] = rand()%32;
}

void fillB(float* B){
    for(int i = 0; i < N; i++) B[i] = rand()%32;
}

bool verifyResult(float *gold, float *result) {

    for (int i = 0; i < N; i++) {
        if (gold[i] != result[i]) {
            printf ("Mismatch : [%d], Expected : %f, Actual : %f\n",
                        i, gold[i], result[i]);
            return 0;
        }
    }

    return 1;
}

int main(int argc, char* argv[]) 
{
    // DON'T TOUCH THIS
    int groupnum = atoi(argv[1]);
    srand(groupnum);

    float* A = new float[N];
    memset(A, 0, N*sizeof(float));
    float* B = new float[N];
    memset(B, 0, N*sizeof(float));
    float* C_serial = new float[N];
    memset(C_serial, 0, N*sizeof(float));
    float* C_simd = new float[N];
    memset(C_simd, 0, N*sizeof(float));

    fillA(A);
    fillB(B);

    intrinsics_scalar(N, A, B, C_serial);
    intrinsics_simd(N, A, B, C_simd);

    verifyResult(C_serial, C_simd);
    printStats();

    delete[] A;
    delete[] B;
    delete[] C_serial;
    delete[] C_simd;

    return 0;
}
