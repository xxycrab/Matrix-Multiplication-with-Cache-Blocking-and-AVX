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

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 1024
#define BLOCK_SIZE_2 256
#define BLOCK_SIZE_3_M 64
#define BLOCK_SIZE_3_N 64
#define BLOCK_SIZE_3_K 64
// #define BLOCK_SIZE 719
#endif

#define min(a, b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block(int lda, int M, int N, int K, double *A, double *B, double *C) {
    /* For each row i of A */
    for (int i = 0; i < M / 6; ++i) {
        /* For each column j of B */
        for (int j = 0; j < N / 4; ++j) {
            register __m256d c_00_03 = _mm256_loadu_pd(C + 6 * i * lda + j * 4);
            register __m256d c_10_13 = _mm256_loadu_pd(C + (6 * i + 1) * lda + j * 4);
            register __m256d c_20_23 = _mm256_loadu_pd(C + (6 * i + 2) * lda + j * 4);
            register __m256d c_30_33 = _mm256_loadu_pd(C + (6 * i + 3) * lda + j * 4);
            register __m256d c_40_43 = _mm256_loadu_pd(C + (6 * i + 4) * lda + j * 4);
            register __m256d c_50_53 = _mm256_loadu_pd(C + (6 * i + 5) * lda + j * 4);


            for (int kk = 0; kk < 4 && kk < K - j * 4; kk++) {
                register __m256d a0x = _mm256_broadcast_sd(A + 6 * i * lda + j * 4 + kk);
                register __m256d a1x = _mm256_broadcast_sd(A + (6 * i + 1) * lda + j * 4 + kk);
                register __m256d a2x = _mm256_broadcast_sd(A + (6 * i + 2) * lda + j * 4 + kk);
                register __m256d a3x = _mm256_broadcast_sd(A + (6 * i + 3) * lda + j * 4 + kk);
                register __m256d a4x = _mm256_broadcast_sd(A + (6 * i + 4) * lda + j * 4 + kk);
                register __m256d a5x = _mm256_broadcast_sd(A + (6 * i + 5) * lda + j * 4 + kk);
                for (int k = 0; k < K / 4; k++) {
                    register __m256d b_00_03 = _mm256_loadu_pd(B + 4 * k * lda + j * 4);
                    register __m256d b_10_13 = _mm256_loadu_pd(B + (4 * k + 1) * lda + j * 4);
                    register __m256d b_20_23 = _mm256_loadu_pd(B + (4 * k + 2) * lda + j * 4);
                    register __m256d b_30_33 = _mm256_loadu_pd(B + (4 * k + 3) * lda + j * 4);

                    c_00_03 = _mm256_fmadd_pd(a0x, b_00_03, c_00_03);
                    c_00_03 = _mm256_fmadd_pd(a0x, b_10_13, c_00_03);
                    c_00_03 = _mm256_fmadd_pd(a0x, b_20_23, c_00_03);
                    c_00_03 = _mm256_fmadd_pd(a0x, b_30_33, c_00_03);

                    c_10_13 = _mm256_fmadd_pd(a1x, b_00_03, c_10_13);
                    c_10_13 = _mm256_fmadd_pd(a1x, b_10_13, c_10_13);
                    c_10_13 = _mm256_fmadd_pd(a1x, b_20_23, c_10_13);
                    c_10_13 = _mm256_fmadd_pd(a1x, b_30_33, c_10_13);

                    c_20_23 = _mm256_fmadd_pd(a2x, b_00_03, c_20_23);
                    c_20_23 = _mm256_fmadd_pd(a2x, b_10_13, c_20_23);
                    c_20_23 = _mm256_fmadd_pd(a2x, b_20_23, c_20_23);
                    c_20_23 = _mm256_fmadd_pd(a2x, b_30_33, c_20_23);

                    c_30_33 = _mm256_fmadd_pd(a3x, b_00_03, c_30_33);
                    c_30_33 = _mm256_fmadd_pd(a3x, b_10_13, c_30_33);
                    c_30_33 = _mm256_fmadd_pd(a3x, b_20_23, c_30_33);
                    c_30_33 = _mm256_fmadd_pd(a3x, b_30_33, c_30_33);

                    c_40_43 = _mm256_fmadd_pd(a4x, b_00_03, c_40_43);
                    c_40_43 = _mm256_fmadd_pd(a4x, b_10_13, c_40_43);
                    c_40_43 = _mm256_fmadd_pd(a4x, b_20_23, c_40_43);
                    c_40_43 = _mm256_fmadd_pd(a4x, b_30_33, c_40_43);

                    c_50_53 = _mm256_fmadd_pd(a5x, b_00_03, c_50_53);
                    c_50_53 = _mm256_fmadd_pd(a5x, b_10_13, c_50_53);
                    c_50_53 = _mm256_fmadd_pd(a5x, b_20_23, c_50_53);
                    c_50_53 = _mm256_fmadd_pd(a5x, b_30_33, c_50_53);
                    }
                }
            }
//            /* Compute C(i,j) */
//            double cij = C[i * lda + j];
//            for (int k = 0; k < K; ++k)
//#ifdef TRANSPOSE
//                cij += A[i*lda+k] * B[j*lda+k];
//#else
//                    cij += A[i * lda + k] * B[k * lda + j];
//#endif
//            C[i * lda + j] = cij;
            _mm256_storeu_pd(C + 6 * i * lda + j * 4, c_00_03);
            _mm256_storeu_pd(C + (6 * i + 1) * lda + j * 4, c_10_13);
            _mm256_storeu_pd(C + (6 * i + 2) * lda + j * 4, c_20_23);
            _mm256_storeu_pd(C + (6 * i + 3) * lda + j * 4, c_30_33);
            _mm256_storeu_pd(C + (6 * i + 4) * lda + j * 4, c_40_43);
            _mm256_storeu_pd(C + (6 * i + 5) * lda + j * 4, c_50_53);
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
    /* For each block-row of A */
    for (int i = 0; i < lda; i += BLOCK_SIZE)
        /* For each block-column of B */
        for (int j = 0; j < lda; j += BLOCK_SIZE)
            /* Accumulate block dgemms into block of C */
            for (int k = 0; k < lda; k += BLOCK_SIZE)
                for (int ii = i; ii < i + BLOCK_SIZE && ii < lda; ii += BLOCK_SIZE_2)
                    for (int jj = j; jj < j + BLOCK_SIZE && jj < lda; jj += BLOCK_SIZE_2)
                        for (int kk = k; kk < k + BLOCK_SIZE && kk < lda; kk += BLOCK_SIZE_2)
                            for (int iii = ii; iii < ii + BLOCK_SIZE_2 && iii < lda; iii += BLOCK_SIZE_3_M)
                                for (int jjj = jj; jjj < jj + BLOCK_SIZE_2 && jjj < lda; jjj += BLOCK_SIZE_3_N)
                                    for (int kkk = kk; kkk < kk + BLOCK_SIZE_2 && kkk < lda; kkk += BLOCK_SIZE_3_K) {
                                        /* Correct block dimensions if block "goes off edge of" the matrix */
                                        int M = min (BLOCK_SIZE_3_M, lda - iii);
                                        int N = min (BLOCK_SIZE_3_N, lda - jjj);
                                        int K = min (BLOCK_SIZE_3_K, lda - kkk);

                                        /* Perform individual block dgemm */
#ifdef TRANSPOSE
                                        do_block(lda, M, N, K, A + iii*lda + kkk, B + jjj*lda + kkk, C + iii*lda + jjj);
#else
                                        do_block(lda, M, N, K, A + iii * lda + kkk, B + kkk * lda + jjj,
                                                 C + iii * lda + jjj);
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
}
