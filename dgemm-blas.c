/*
 *  DGEMM Code, provided by Jim Demmel at UC Berkeley
 *  Modified by Scott B. Baden at UC San Diego to
 *  Support CBLAS interface
 */

#ifdef USE_MKL
#include "mkl.h"
#else
#include "cblas.h"
#endif

const char* dgemm_desc = "Reference dgemm.";

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are N-by-N matrices stored in row-major format.
 * On exit, A and B maintain their input values.    
 * This function wraps a call to the BLAS-3 routine DGEMM, via the CBLAS interfce */
/*
 * Users of the CBLAS interface: be aware that the CBLAS are just a C
 * interface to the BLAS, which is based on the FORTRAN standard and
 * subject to the FORTRAN standard restrictions. In particular, the output
 * parameters should not be referenced through more than one argument
 * http://software.intel.com/sites/products/documentation/hpc/compilerpro/en-us/cpp/lin/mkl/refman/appendices/mkl_appD_Intro.html
 */

/* Set up the call for dgemm, to perform the  matrix  multiplication
 * on a square matrix
 *
*/
void square_dgemm (int N, double* A, double* B, double* C)
{
    const double Alpha = 1.0;
    const double Beta  = 1.0;
    const int M = N, K=N;
    const int LDA = N, LDB = N, LDC = N;
    const enum CBLAS_TRANSPOSE transA = CblasNoTrans;
    const enum CBLAS_TRANSPOSE transB = CblasNoTrans;
    /* Don't change this call */
    cblas_dgemm( CblasRowMajor, transA, transB, M, N, K,
                 Alpha, A, LDA, B, LDB, Beta, C, LDC );
	    
}
