/*
 *  Driver code for Matrix Multplication
 *  Provided by Jim Demmel at UC Berkeley
 *  Modified by Scott B. Baden at UC San Diego to
 *      Enable user to select one problem size only via the -n option
 *      Support CBLAS interface
 */

#include <stdlib.h> // For: exit, random, malloc, free, NULL, EXIT_FAILURE
#include <stdio.h>  // For: perror
#include <string.h> // For: memset

#include <float.h>  // For: DBL_EPSILON
#include <math.h>   // For: fabs

#ifdef USE_MKL
#include "mkl.h"
#else
#include "cblas.h"
#endif

void cmdLine(int argc, char *argv[], int* n, int* noCheck);
/* reference_dgemm wraps a call to the BLAS-3 routine DGEMM, via the standard FORTRAN interface - hence the reference semantics. */ 
void reference_dgemm (int N, double Alpha, double* A, double* B, double* C)
{
  const double Beta  = 1.0;
  const int M = N, K=N;
  const int LDA = N, LDB = N, LDC = N;
  const enum CBLAS_TRANSPOSE transA = CblasNoTrans;
  const enum CBLAS_TRANSPOSE transB = CblasNoTrans;
  /* Don't change this call */
  cblas_dgemm( CblasRowMajor, transA, transB, M, N, K,
               Alpha, A, LDA, B, LDB, Beta, C, LDC );
}   

/* Your function must have the following signature: */
extern const char* dgemm_desc;
extern void square_dgemm (int, double*, double*, double*);

extern double wall_time();


#include "debugMat.h"

void Fail (const char* message)
{
  perror (message);
  exit (EXIT_FAILURE);
}

void fill (double* p, int n)
{
  long int Rmax   = RAND_MAX;
  long int Rmax_2 = Rmax >> 1;
  long int RM     =  Rmax_2 + 1;
  for (int i = 0; i < n; ++i){
    long int r = random();   // Uniformly distributed ints over [0,RAND_MAX]
                             // Typical value of RAND_MAX: 2^31 - 1
    long int R = r - RM;
    p[i] = (double) R / (double) RM; // Uniformly distributed over [-1, 1]
  }
}

void absolute_value (double *p, int n)
{
  for (int i = 0; i < n; ++i)
    p[i] = fabs (p[i]);
}

/* The benchmarking program */
int main (int argc, char **argv)
{
  printf ("Description:\t%s\n\n", dgemm_desc);
  /* We can pick just one size with the -n flag */
  int n0;
  int noCheck;
  cmdLine(argc,argv,&n0,&noCheck);

  /* Test sizes should highlight performance dips at multiples of certain powers-of-two */

  int test_sizes[] = 

  /* Multiples-of-32, +/- 1. Currently uncommented. */
/*  {31,32,33,63,64,65,95,96,97,127,128,129,159,160,161,191,192,193,223,224,225,255,256,257,287,288,289,319,320,321,351,352,353,383,384,385,415,416,417,447,448,449,479,480,481,511,512,513,543,544,545,575,576,577,607,608,609,639,640,641,671,672,673,703,704,705,735,736,737,767,768,769,799,800,801,831,832,833,863,864,865,895,896,897,927,928,929,959,960,961,991,992,993,1023,1024,1025};  */
  /* Multiples-of-32, +/- 1. Currently uncommented. Large  only */
  {511,512,513,543,544,545,575,576,577,607,608,609,639,640,641,671,672,673,703,704,705,735,736,737,767,768,769,799,800,801,831,832,833,863,864,865,895,896,897,927,928,929,959,960,961,991,992,993,1023,1024,1025}; 
/*
  {31,32,33,63,64,65,95,96,97,127,128,129,159,160,161,191,192,193,223,224,225,255,256,257,287,288,289,319,320,321,351,352,353,383,384,385,415,416,417,447,448,449,479,480,481,511,512,513,543,544,545,575,576,577,607,608,609,639,640,641,671,672,673,703,704,705,735,736,737,767,768,769};
  */

  /* A representative subset of the first list. Currently commented. */ 
  /*
  { 31, 32, 96, 97, 127, 128, 129, 191, 192, 229, 255, 256, 257,
    319, 320, 321, 417, 479, 480, 511, 512, 639, 640, 767, 768, 769 };
    */
  /* N= 31 through 768+-1 in steps of 16 */
  /*
  { 31, 32, 33, 39, 40, 41, 47, 48, 49, 55, 56, 57, 63, 64, 65, 71, 72, 73, 79, 80, 81, 87, 88, 89, 95, 96, 97, 103, 104, 105, 111, 112, 113, 119, 120, 121, 127, 128, 129, 135, 136, 137, 143, 144, 145, 151, 152, 153, 159, 160, 161, 167, 168, 169, 175, 176, 177, 183, 184, 185, 191, 192, 193, 199, 200, 201, 207, 208, 209, 215, 216, 217, 223, 224, 225, 231, 232, 233, 239, 240, 241, 247, 248, 249, 255, 256, 257, 263, 264, 265, 271, 272, 273, 279, 280, 281, 287, 288, 289, 295, 296, 297, 303, 304, 305, 311, 312, 313, 319, 320, 321, 327, 328, 329, 335, 336, 337, 343, 344, 345, 351, 352, 353, 359, 360, 361, 367, 368, 369, 375, 376, 377, 383, 384, 385, 391, 392, 393, 399, 400, 401, 407, 408, 409, 415, 416, 417, 423, 424, 425, 431, 432, 433, 439, 440, 441, 447, 448, 449, 455, 456, 457, 463, 464, 465, 471, 472, 473, 479, 480, 481, 487, 488, 489, 495, 496, 497, 503, 504, 505, 511, 512, 513, 519, 520, 521, 527, 528, 529, 535, 536, 537, 543, 544, 545, 551, 552, 553, 559, 560, 561, 567, 568, 569, 575, 576, 577, 583, 584, 585, 591, 592, 593, 599, 600, 601, 607, 608, 609, 615, 616, 617, 623, 624, 625, 631, 632, 633, 639, 640, 641, 647, 648, 649, 655, 656, 657, 663, 664, 665, 671, 672, 673, 679, 680, 681, 687, 688, 689, 695, 696, 697, 703, 704, 705, 711, 712, 713, 719, 720, 721, 727, 728, 729, 735, 736, 737, 743, 744, 745, 751, 752, 753, 759, 760, 761, 767, 768, 769 }; */

  int nsizes = sizeof(test_sizes)/sizeof(test_sizes[0]);

  /* assume last size is also the largest size */
  int nmax = test_sizes[nsizes-1];

  if (n0){
      nmax = n0;
      test_sizes[0] = n0;
    }

  /* allocate memory for all problems */
  double* buf = NULL;
  buf = (double*) malloc (3 * nmax * nmax * sizeof(double));
  if (buf == NULL){
    if (n0){
        Fail ("Failed to allocate matrix");
    }
    else{
        Fail ("Failed to allocate largest matrix");
    }
  }

  int sizes = sizeof(test_sizes)/sizeof(test_sizes[0]);
  if (n0)
      sizes = 1;

  /* For each test size */
  for (int isize = 0; isize < sizes; ++isize)
  {
    /* Create and fill 3 random matrices A,B,C*/
    int n = test_sizes[isize];

    double* A = buf + 0;
    double* B = A + nmax*nmax;
    double* C = B + nmax*nmax;

    fill (A, n*n);
    fill (B, n*n);
    fill (C, n*n);

    /* Measure performance (in Gflops/s). */

    /* Time a "sufficiently long" sequence of calls to reduce noise */
    double Gflops_s, seconds = -1.0;
    double timeout = 0.1; // "sufficiently long" := at least 1/10 second.
    for (int n_iterations = 1; seconds < timeout; n_iterations *= 2) 
    {
      /* Warm-up */
      square_dgemm (n, A, B, C);

      /* Benchmark n_iterations runs of square_dgemm */
      seconds = -wall_time();
      for (int it = 0; it < n_iterations; ++it)
	square_dgemm (n, A, B, C);
      seconds += wall_time();

      /*  compute Mflop/s rate */
      Gflops_s = 2.e-9 * n_iterations * n * n * n / seconds;
    }
    printf ("Size: %d\tGflop/s: %.3g\n", n, Gflops_s);

    if (!noCheck){
        /* Ensure that error does not exceed the theoretical error bound. */

        /* C := A * B, computed with square_dgemm */
        memset (C, 0, n * n * sizeof(double));
        square_dgemm (n, A, B, C);

        /* Do not explicitly check that A and B were unmodified on square_dgemm exit
        *  - if they were, the following will most likely detect it:   
        * C := C - A * B, computed with reference_dgemm */
        reference_dgemm(n, -1., A, B, C);

        /* A := |A|, B := |B|, C := |C| */
        absolute_value (A, n * n);
        absolute_value (B, n * n);
        absolute_value (C, n * n);

        /* C := |C| - 3 * e_mach * n * |A| * |B|, computed with reference_dgemm */ 
        reference_dgemm (n, -3.*DBL_EPSILON*n, A, B, C);

        /* If any element in C is positive, then something went wrong in square_dgemm */
        for (int i = 0; i < n * n; ++i)
        if (C[i] > 0)
            Fail("*** FAILURE *** Error in matrix multiply exceeds componentwise error bounds.\n" );
    }
  }

  free (buf);

  return 0;
}
