#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <cmath>

#include "../common/matrix.hpp"

#define MAT_SIZE 1024

typedef void (*het_f)(int N, float *A, float *B, float *C, float *tmp1, float *tmp2, float *tmp3, float *X, float *Y);

struct het_impl
{
    const char *name;
    het_f func;
};

bool result_correct(float* sol, float* gold)
{
    int errors = 0;
    for (int i = 0; i < MAT_SIZE; i++)
    {
        // 1% error tolerance
        if (fabs(sol[i] - gold[i]) > fabs(sol[i] / 100))
        {
            if (errors++ < 10)
                printf("Error at (%d): %.4f, should be %.4f\n", i, sol[i], gold[i]);
        }
    }

    if (errors)
        printf("%sfound %d error\n", errors >= 10 ? "...\n" : "", errors);

    return errors == 0;
}

#define MATMUL(N, A, B, C) \
    for (int r = 0; r < N; r++) \
    { \
        for (int c = 0; c < N; c++) \
        { \
            float val = 0.0; \
            for (int k = 0; k < N; k++) \
            { \
                val += A[r * N + k] * B[k * N + c]; \
            } \
            C[r * N + c] = val; \
        } \
    }

#define MVMUL(N, A, X, Y) \
    for (int r = 0; r < N; r++) \
    { \
        float val = 0.0; \
        for (int c = 0; c < N; c++) \
        { \
            val += A[r * N + c] * X[c]; \
        } \
        Y[r] = val; \
    }

void matsquash(int N, const float *A, float *X)
{
    #pragma omp parallel loop
    for (int r = 0; r < N; r++)
    {
        float val = 0.0;
        for (int c = 0; c < N; c++)
        {
            val += A[r * N + c];
        }
        X[r] = val;
    }
}


void example(int N, float *A, float *B, float *C, float *D, float *E, float *F, float *X, float *Y)
{
    MATMUL(N, A, B, C);

    MATMUL(N, C, D, E);

    matsquash(N, A, X);

    MVMUL(N, E, X, Y);
}


void cpu_only(int N, float *A, float *B, float *C, float *D, float *E, float *F, float *X, float *Y)
{
    // your implementation here
}

void heterogeneous(int N, float *A, float *B, float *C, float *D, float *E, float *F, float *X, float *Y)
{
    // your implementation here
}

int main(int argc, char *argv[])
{
    srand(1);

    Matrix A(MAT_SIZE, MAT_SIZE);
    Matrix B(MAT_SIZE, MAT_SIZE);
    Matrix C(MAT_SIZE, MAT_SIZE);

    Matrix tmp1(MAT_SIZE, MAT_SIZE);
    Matrix tmp2(MAT_SIZE, MAT_SIZE);
    Matrix tmp3(MAT_SIZE, MAT_SIZE);

    Vector X(MAT_SIZE);
    Vector Y(MAT_SIZE);

    cpu_only(MAT_SIZE, &A.data[0], &B.data[0], &C.data[0], &tmp1.data[0],
           &tmp2.data[0], &tmp3.data[0], &X.data[0], &Y.data[0]);

    Vector gold = Y.copy();

    std::vector<het_impl> implementations = {
        {"cpu only", cpu_only},
        {"heterogeneous", heterogeneous},
    };

    for (auto impl : implementations)
    {
        int runs = 0;
        double runtime = 0.;
        while (runtime < 2.)
        {
            double start = omp_get_wtime();
            impl.func(MAT_SIZE, &A.data[0], &B.data[0], &C.data[0], &tmp1.data[0],
                      &tmp2.data[0], &tmp3.data[0], &X.data[0], &Y.data[0]);
            runtime += omp_get_wtime() - start;
            runs++;
        }
        runtime /= static_cast<double>(runs);
        bool okay = result_correct(&Y.data[0], &gold.data[0]);

        printf("%13s: %5s, took %fs\n", impl.name, okay ? "OKAY" : "WRONG", runtime);
    }

    return 0;
}
