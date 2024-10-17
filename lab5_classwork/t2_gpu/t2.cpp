#include <iostream>
#include <vector>
#include <map>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <cmath>

#include "../common/matrix.hpp"
#include "t2.hpp"

int GROUP_NUMBER;
int YOUR_PATTERN;

typedef void (*rand_f)(Matrix_u32 &A, const int *X);

struct rand_impl
{
    const char *name;
    rand_f func;
};


// shuffles the bits of a non-zero number x to randomize it
uint32_t xorshift32(uint32_t x)
{
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

uint64_t build_uniform_vec(size_t size, std::vector<int> &vec)
{
    uint64_t total = 0;
    // every node gets [1024, 1024+128] iterations
    for (size_t i = 0; i < vec.size(); i++)
    {
        vec[i] = 1024 + (rand() % 128);
        total += vec[i];
    }

    return total;
}

uint64_t build_pattern_vec(size_t size, std::vector<int> &vec)
{
    uint64_t total = 0;
    for (size_t i = 0; i < vec.size(); i++)
    {
        // most nodes get one iteration
        vec[i] = 1;

        // 1/YOUR_PATTERN nodes gets a lot more. on average, it should have
        // the same total iterations as the uniform vector
        if (i % YOUR_PATTERN == 0)
            vec[i] = (1024 + (rand() % 128)) * YOUR_PATTERN;

        total += vec[i];
    }

    return total;
}

// generates a random matrix, using X to determine the amount of times to shuffle each element
void randmat_serial(Matrix_u32 &A, const int *X)
{
    const int N = A.rows;
    uint32_t *A_data = &A.data[0];

    // for every matrix element
    for (int r = 0; r < N; r++)
    {
        for (int c = 0; c < N; c++)
        {
            uint32_t val = r * N + c;
            // shuffle it the amount of times dictated by the X vector
            for (int k = 0; k < X[r * N + c]; k++)
                val = xorshift32(val);

            A_data[r * N + c] = val;
        }
    }
}

/* TASK 1: Parallize this function for the GPU, applying
    the appropriate directives to the loop(s)*/
void randmat_gpu(Matrix_u32 &A, const int *X)
{
    const int N = A.rows;
    uint32_t *A_data = &A.data[0];

    // for every matrix element
    for (int r = 0; r < N; r++)
    {
        for (int c = 0; c < N; c++)
        {
            uint32_t val = r * N + c;
            // shuffle it the amount of times dictated by the X vector
            for (int k = 0; k < X[r * N + c]; k++)
                val = xorshift32(val);

            A_data[r * N + c] = val;
        }
    }
}

void test_implementations(Matrix_u32 &mat, const int *shuffle_vec, const uint64_t shuffles)
{
    std::vector<rand_impl> implementations =
    {
        {"serial", randmat_serial},
        {"gpu", randmat_gpu},
    };

    randmat_serial(mat, shuffle_vec);
    auto gold = mat.copy();

    for (auto impl : implementations)
    {
        int runs = 0;
        double runtime = 0.;
        while (runtime < 3.)
        {
            double start = omp_get_wtime();
            impl.func(mat, shuffle_vec);
            runtime += omp_get_wtime() - start;
            runs++;
        }
        runtime /= static_cast<double>(runs);

        mat.compare(gold);
        printf("%10s: %5.0lf million shuffles/s\n",
               impl.name, static_cast<double>(shuffles) / (runtime * 1024. * 1024.));
    }
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cout << "Usage: ./t2 <group_number>" << std::endl;
        exit(1);
    }

    GROUP_NUMBER = atoi(argv[1]);

    if (PATTERNS.count(GROUP_NUMBER) == 1)
    {
        YOUR_PATTERN = PATTERNS[GROUP_NUMBER];
        std::cout << "Your group's (" << GROUP_NUMBER << ") pattern: " << YOUR_PATTERN << std::endl;
    }
    else
    {
        std::cout << "You entered an invalid group number!" << std::endl;
        exit(1);
    }

    srand(1);

    std::vector<int> matrix_size = {64, 1024, 2048};
    for (auto mat_sz : matrix_size)
    {
        std::cout << "\n------------" << std::endl;
        std::cout << "Matrix size:" << mat_sz << std::endl;

        Matrix_u32 mat(mat_sz, mat_sz);

        auto vec_uniform = std::vector<int>(mat_sz * mat_sz);
        uint64_t uni_shuffles = build_uniform_vec(mat_sz * mat_sz, vec_uniform);
        auto vec_pattern = std::vector<int>(mat_sz * mat_sz);
        uint64_t pat_shuffles = build_pattern_vec(mat_sz * mat_sz, vec_pattern);

        std::cout << "Uniform vector (" << uni_shuffles << " total shuffles):" << std::endl;
        test_implementations(mat, &vec_uniform[0], uni_shuffles);
        std::cout << "Pattern vector (" << pat_shuffles << " total shuffles):" << std::endl;
        test_implementations(mat, &vec_pattern[0], pat_shuffles);
    }

    return 0;
}
