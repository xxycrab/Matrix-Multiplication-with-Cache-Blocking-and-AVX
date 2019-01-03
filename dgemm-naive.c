/*
 *  A naive implementation of matrix multiply using the "ijk" algorithm
 *  Provided by Jim Demmel at UC Berkeley
 */

const char* dgemm_desc = "Naive, three-loop dgemm.";

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in row-major order
 * On exit, A and B maintain their input values. */    
void square_dgemm (int n, double* A, double* B, double* C)
{
#ifdef TRANSPOSE
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
        double t = B[i*n+j];
        B[i*n+j] = B[j*n+i];
        B[j*n+i] = t;
  }
#endif
  /* For each row i of A */
  for (int i = 0; i < n; ++i)
    /* For each column j of B */
    for (int j = 0; j < n; ++j) 
    {
      /* Compute C(i,j) */
      double cij = C[i*n+j];
      for( int k = 0; k < n; k++ )
#ifdef TRANSPOSE
	cij += A[i*n+k] * B[j*n+k];
#else
	cij += A[i*n+k] * B[k*n+j];
#endif
      C[i*n+j] = cij;
    }
}
