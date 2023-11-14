/* C source code is found in dgemm_example.c */

#define min(x,y) (((x) < (y)) ? (x) : (y))

#include <iostream>
#include <chrono>
#include <string>
using namespace std;
#include "mkl.h"

#include "fff.h"
#include "ff.h"

int main(int argc, char** argv) {
    float *W1, *W2;
    float *IN, *OUT;

    size_t batch_size, hidden_dim, depth, n_nodes, layer_size;

    batch_size = 16384, hidden_dim = 768, depth = 12, layer_size = 4095;
    n_nodes = (1 << depth) - 1;
    int N = 10;

    cout << "Initializing weights" << std::endl;
    W1 = (float*)mkl_malloc(hidden_dim * n_nodes * sizeof(float), 64);
    W2 = (float*)mkl_malloc(n_nodes * hidden_dim * sizeof(float), 64);

    cout << "Initializing data" << std::endl;
    IN = (float*)mkl_malloc(batch_size * hidden_dim * sizeof(float), 64);
    OUT = (float*)mkl_calloc(batch_size * hidden_dim, sizeof(float), 64);

    // FFF-L1
    {
        cout << "Running FFF-L1 " << N << " times" << std::endl;
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < 250; ++i) {
            fff_l1(IN, W1, W2, OUT, batch_size, hidden_dim, n_nodes, depth);
        }
        auto end = std::chrono::steady_clock::now();
        auto diff = end - start;
        cout << "This took " << std::to_string(chrono::duration<double, milli>(diff).count() / (double)250) << "ms per iteration" << std::endl;
    }

    // FFF-L2
    {
        cout << "Running FFF-L2 " << N << " times" << std::endl;
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < 250; ++i) {
            fff_l2(IN, W1, W2, OUT, batch_size, hidden_dim, n_nodes, depth);
        }
        auto end = std::chrono::steady_clock::now();
        auto diff = end - start;
        cout << "This took " << std::to_string(chrono::duration<double, milli>(diff).count() / (double)250) << "ms per iteration" << std::endl;
    }

    // FF-L3
    {
        cout << "Running FF-L3 " << N << " times" << std::endl;
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < 250; ++i) {
            ff_l3(IN, W1, W2, OUT, batch_size, hidden_dim, layer_size);
        }
        auto end = std::chrono::steady_clock::now();
        auto diff = end - start;
        cout << "This took " << std::to_string(chrono::duration<double, milli>(diff).count() / (double)250) << "ms per iteration" << std::endl;
    }

    // FF-L2
    {
        cout << "Running FF-L2 " << N << " times" << std::endl;
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < N; ++i) {
            ff_l2(IN, W1, W2, OUT, batch_size, hidden_dim, layer_size);
        }
        auto end = std::chrono::steady_clock::now();
        auto diff = end - start;
        cout << "This took " << std::to_string(chrono::duration<double, milli>(diff).count() / (double)N) << "ms per iteration" << std::endl;
    }

    // FF-L1
    {
        cout << "Running FF-L1 " << N << " times" << std::endl;
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < N; ++i) {
            ff_l1(IN, W1, W2, OUT, batch_size, hidden_dim, layer_size);
        }
        auto end = std::chrono::steady_clock::now();
        auto diff = end - start;
        cout << "This took " << std::to_string(chrono::duration<double, milli>(diff).count() / (double)N) << "ms per iteration" << std::endl;
    }


    printf("\nDeallocating memory \n\n");
    mkl_free(W1);
    mkl_free(W2);
    mkl_free(IN);
    mkl_free(OUT);

    printf("Example completed. \n\n");
    return 0;
}

int main_old()
{
    double* A, * B, * C;
    int m, n, k, i, j;
    double alpha, beta;

    printf("\n This example computes real matrix C=alpha*A*B+beta*C using \n"
        " Intel(R) MKL function dgemm, where A, B, and  C are matrices and \n"
        " alpha and beta are double precision scalars\n\n");

    m = 2000, k = 200, n = 1000;
    printf(" Initializing data for matrix multiplication C=A*B for matrix \n"
        " A(%ix%i) and matrix B(%ix%i)\n\n", m, k, k, n);
    alpha = 1.0; beta = 0.0;

    printf(" Allocating memory for matrices aligned on 64-byte boundary for better \n"
        " performance \n\n");
    A = (double*)mkl_malloc(m * k * sizeof(double), 64);
    B = (double*)mkl_malloc(k * n * sizeof(double), 64);
    C = (double*)mkl_malloc(m * n * sizeof(double), 64);
    if (A == NULL || B == NULL || C == NULL) {
        printf("\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
        mkl_free(A);
        mkl_free(B);
        mkl_free(C);
        return 1;
    }

    printf(" Intializing matrix data \n\n");
    for (i = 0; i < (m * k); i++) {
        A[i] = (double)(i + 1);
    }

    for (i = 0; i < (k * n); i++) {
        B[i] = (double)(-i - 1);
    }

    for (i = 0; i < (m * n); i++) {
        C[i] = 0.0;
    }

    printf(" Computing matrix product using Intel(R) MKL dgemm function via CBLAS interface \n\n");
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        m, n, k, alpha, A, k, B, n, beta, C, n);
    printf("\n Computations completed.\n\n");

    printf(" Top left corner of matrix A: \n");
    for (i = 0; i < min(m, 6); i++) {
        for (j = 0; j < min(k, 6); j++) {
            printf("%12.0f", A[j + i * k]);
        }
        printf("\n");
    }

    printf("\n Top left corner of matrix B: \n");
    for (i = 0; i < min(k, 6); i++) {
        for (j = 0; j < min(n, 6); j++) {
            printf("%12.0f", B[j + i * n]);
        }
        printf("\n");
    }

    printf("\n Top left corner of matrix C: \n");
    for (i = 0; i < min(m, 6); i++) {
        for (j = 0; j < min(n, 6); j++) {
            printf("%12.5G", C[j + i * n]);
        }
        printf("\n");
    }

    printf("\n Deallocating memory \n\n");
    mkl_free(A);
    mkl_free(B);
    mkl_free(C);

    printf(" Example completed. \n\n");
    return 0;
}