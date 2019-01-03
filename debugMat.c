#include <stdio.h>
void printMat(int M, int N, char *title, double *X){
  printf("%s\n", title);
  for (int i=0; i<M; i++){
    for (int j=0; j<N-1; j++){
      printf("%5.2f ", *(X+i*N + j));
    }
    printf("%5.2f\n", *(X+i*N + N-1));
  }
}

void identMat(int N, double *X){
  for (int i=0; i<N; i++){
    for (int j=0; j<N; j++){
      *(X+i*N+j) = (i==j) ? 1.0 : 0.0;
    }
  }
}

void seqMat(int M, int N, double *X){
  double num=1.0;
  for (int i=0; i<M; i++){
    for (int j=0; j<N; j++){
      *(X+i*N+j) = num;
      num = num + 1.0;
    }
  }
}

void setMat(int M, int N, double *X, double v){
  for (int i=0; i<M; i++){
    for (int j=0; j<N; j++){
      *(X+i*N+j) = v;
    }
  }
}
