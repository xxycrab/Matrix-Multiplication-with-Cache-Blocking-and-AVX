#ifndef _DEBUG_MAT_H
#define _DEBUG_MAT_H

#include <stdio.h>
void printMat(int M, int N, char *title, double *X);
void identMat(int N, double *X);
void seqMat(int M, int N, double *X);
void setMat(int M, int N, double *X, double v);
#endif
