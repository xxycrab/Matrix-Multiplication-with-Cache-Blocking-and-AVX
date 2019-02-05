/*
 *  A simple blocked implementation of matrix multiply
 *  Provided by Jim Demmel at UC Berkeley
 *  Modified by Scott B. Baden at UC San Diego to
 *    Enable user to select one problem size only via the -n option
 *    Support CBLAS interface
 */

const char *dgemm_desc = "Simple blocked dgemm.";

#include <immintrin.h>
#include <avx2intrin.h>
#include "debugMat.h"
#include "debugMat.c"

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE_M 1154
#define BLOCK_SIZE_1_M 384
#define BLOCK_SIZE_2_M 48
#define BLOCK_SIZE_N 1154
#define BLOCK_SIZE_1_N 384
#define BLOCK_SIZE_2_N 48
#define BLOCK_SIZE_K 1154
#define BLOCK_SIZE_1_K 384
#define BLOCK_SIZE_2_K 48
// #define BLOCK_SIZE 719
#endif

#define min(a, b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */

/* Vector tiling and loop unrolling */
static void do_block(int lda, int M, int N, int K, double* restrict A, double* restrict B, double* restrict C) {
    /* For each row i of A */
    for (int i = 0; i < M / 4; ++i) {
        /* For each column j of B */
        for (int j = 0; j < N / 12; ++j) {
            register __m256d c_00_03_0 = _mm256_loadu_pd(C + (4 * i) * lda + j * 12);
            register __m256d c_00_03_1 = _mm256_loadu_pd(C + (4 * i) * lda + j * 12 + 4);
            register __m256d c_00_03_2 = _mm256_loadu_pd(C + (4 * i) * lda + j * 12 + 8);
            register __m256d c_10_13_0 = _mm256_loadu_pd(C + (4 * i + 1) * lda + j * 12);
            register __m256d c_10_13_1 = _mm256_loadu_pd(C + (4 * i + 1) * lda + j * 12+4);
            register __m256d c_10_13_2 = _mm256_loadu_pd(C + (4 * i + 1) * lda + j * 12+8);
            register __m256d c_20_23_0 = _mm256_loadu_pd(C + (4 * i + 2) * lda + j * 12);
            register __m256d c_20_23_1 = _mm256_loadu_pd(C + (4 * i + 2) * lda + j * 12+4);
            register __m256d c_20_23_2 = _mm256_loadu_pd(C + (4 * i + 2) * lda + j * 12+8);
            register __m256d c_30_33_0 = _mm256_loadu_pd(C + (4 * i + 3) * lda + j * 12);
            register __m256d c_30_33_1 = _mm256_loadu_pd(C + (4 * i + 3) * lda + j * 12+4);
            register __m256d c_30_33_2 = _mm256_loadu_pd(C + (4 * i + 3) * lda + j * 12+8);

            /* Loop unrolling */
            for (int k = 0; k < K; k += 1) {
                register __m256d b_00_03 = _mm256_loadu_pd(B + k * lda + j * 12);
                register __m256d b_10_03 = _mm256_loadu_pd(B + k * lda + j * 12+4);
                register __m256d b_20_03 = _mm256_loadu_pd(B + k * lda + j * 12+8);
                register __m256d a00 = _mm256_broadcast_sd(A + (4 * i) * lda + k);

                c_00_03_0 = _mm256_fmadd_pd(a00, b_00_03, c_00_03_0);
                c_00_03_1 = _mm256_fmadd_pd(a00, b_10_03, c_00_03_1);
                c_00_03_2 = _mm256_fmadd_pd(a00, b_20_03, c_00_03_2);

                a00 = _mm256_broadcast_sd(A + (4 * i + 1) * lda + k);
                c_10_13_0 = _mm256_fmadd_pd(a00, b_00_03, c_10_13_0);
                c_10_13_1 = _mm256_fmadd_pd(a00, b_10_03, c_10_13_1);
                c_10_13_2 = _mm256_fmadd_pd(a00, b_20_03, c_10_13_2);

                a00 = _mm256_broadcast_sd(A + (4 * i + 2) * lda + k);
                c_20_23_0 = _mm256_fmadd_pd(a00, b_00_03, c_20_23_0);
                c_20_23_1 = _mm256_fmadd_pd(a00, b_10_03, c_20_23_1);
                c_20_23_2 = _mm256_fmadd_pd(a00, b_20_03, c_20_23_2);

                a00 = _mm256_broadcast_sd(A + (4 * i + 3) * lda + k);
                c_30_33_0 = _mm256_fmadd_pd(a00, b_00_03, c_30_33_0);
                c_30_33_1 = _mm256_fmadd_pd(a00, b_10_03, c_30_33_1);
                c_30_33_2 = _mm256_fmadd_pd(a00, b_20_03, c_30_33_2);
            }
            _mm256_storeu_pd(C + (4 * i) * lda + j * 12, c_00_03_0);
            _mm256_storeu_pd(C + (4 * i) * lda + j * 12+ 4 , c_00_03_1);
            _mm256_storeu_pd(C + (4 * i) * lda + j * 12+ 8 , c_00_03_2);
            _mm256_storeu_pd(C + (4 * i + 1) * lda + j * 12, c_10_13_0);
            _mm256_storeu_pd(C + (4 * i + 1) * lda + j * 12+ 4, c_10_13_1);
            _mm256_storeu_pd(C + (4 * i + 1) * lda + j * 12+ 8, c_10_13_2);
            _mm256_storeu_pd(C + (4 * i + 2) * lda + j * 12, c_20_23_0);
            _mm256_storeu_pd(C + (4 * i + 2) * lda + j * 12 + 4, c_20_23_1);
            _mm256_storeu_pd(C + (4 * i + 2) * lda + j * 12 + 8, c_20_23_2);
            _mm256_storeu_pd(C + (4 * i + 3) * lda + j * 12, c_30_33_0);
            _mm256_storeu_pd(C + (4 * i + 3) * lda + j * 12 + 4, c_30_33_1);
            _mm256_storeu_pd(C + (4 * i + 3) * lda + j * 12 + 8, c_30_33_2);
        }
    }
}

/* Third level blocking, L2 cache*/
static void do_block_L2(int lda, int M1, int N1, int K1, double *A, double *B, double *C) {
    for (int i = 0; i < M1; i += BLOCK_SIZE_2_M) {
        for (int j = 0; j < N1; j += BLOCK_SIZE_2_N) {
            for (int k = 0; k < K1; k += BLOCK_SIZE_2_K) {
                int M2 = min(BLOCK_SIZE_2_M, M1 - i);
                int N2 = min (BLOCK_SIZE_2_N, N1 - j);
                int K2 = min (BLOCK_SIZE_2_K, K1 - k);
                do_block(lda, M2, N2, K2, A + i * lda + k, B + k * lda + j, C + i * lda + j);
            }
        }
    }
}

/* Second level blocking, L1 cache*/
static void do_block_L1(int lda, int M, int N, int K, double *A, double *B, double *C) {
    for (int i = 0; i < M; i += BLOCK_SIZE_1_M) {
        for (int j = 0; j < N; j += BLOCK_SIZE_1_N) {
            for (int k = 0; k < K; k += BLOCK_SIZE_1_K) {
                int M1 = min(BLOCK_SIZE_1_M, M - i);
                int N1 = min (BLOCK_SIZE_1_N, N - j);
                int K1 = min (BLOCK_SIZE_1_K, K - k);

                do_block_L2(lda, M1, N1, K1, A + i * lda + k, B + k * lda + j, C + i * lda + j);
            }
        }
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in row-major order
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double *A, double *B, double *C) {
#ifdef TRANSPOSE
    for (int i = 0; i < lda; ++i)
    for (int j = i+1; j < lda; ++j) {
        double t = B[i*lda+j];
        B[i*lda+j] = B[j*lda+i];
        B[j*lda+i] = t;
    }
#endif
    /* Deal with edge condition
     * padding matrix to 4n*4n */
    int r = lda % 12, LDA = lda;   //residual of 4*4 tiling
    if (r != 0) {
        LDA = lda + (12 - r);
    }

    double *A_new = (double *) _mm_malloc(LDA * LDA * sizeof(double), 32);
    double *B_new = (double *) _mm_malloc(LDA * LDA * sizeof(double), 32);
    double *C_new = (double *) _mm_malloc(LDA * LDA * sizeof(double), 32);

    for (int i = 0; i < LDA; i++) {
        for (int j = 0; j < LDA; j++) {
            if (i < lda && j < lda) {
                A_new[i * LDA + j] = A[i * lda + j];
                B_new[i * LDA + j] = B[i * lda + j];
                C_new[i * LDA + j] = C[i * lda + j];
            } else {
                A_new[i * LDA + j] = 0.;
                B_new[i * LDA + j] = 0.;
                C_new[i * LDA + j] = 0.;
            }
        }
    }
    for (int i = 0; i < LDA; i += BLOCK_SIZE_M)
        /* For each block-column of B */
        for (int j = 0; j < LDA; j += BLOCK_SIZE_N)
            /* Accumulate block dgemms into block of C */
            for (int k = 0; k < LDA; k += BLOCK_SIZE_K) {
                /* Correct block dimensions if block "goes off edge of" the matrix */
                int M = min (BLOCK_SIZE_M, LDA - i);
                int N = min (BLOCK_SIZE_N, LDA - j);
                int K = min (BLOCK_SIZE_K, LDA - k);
                /* Perform individual block dgemm */
#ifdef TRANSPOSE
                do_block_L1(LDA, M, N, K, A_new + i*LDA + k, B_new + j*LDA + k, C_new + i*LDA + j);
#else
                do_block_L1(LDA, M, N, K, A_new + i * LDA + k, B_new + k * LDA + j, C_new + i * LDA + j);
                for(int i = 0; i<lda; i++){
                    for(int j = 0; j<lda; j++){
                        C[i*lda+j] = C_new[i*LDA+j];
                    }
                }
#endif
            }
#if TRANSPOSE
    for (int i = 0; i < lda; ++i)
    for (int j = i+1; j < lda; ++j) {
        double t = B[i*lda+j];
        B[i*lda+j] = B[j*lda+i];
        B[j*lda+i] = t;
    }
#endif
    free(A_new);
    free(B_new);
    free(C_new);
}
