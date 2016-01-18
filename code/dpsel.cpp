/***********************************************************************
 *
 *  UNIVERSVM IMPLEMENTATION
 *  Copyright (C) 2005 Fabian Sinz  Max Planck Institute for Biological Cybernetics
 *
 *  Includes parts of LUSH Lisp Universal Shell:
 *  Copyright (C) 2002 Leon Bottou, Yann Le Cun, AT&T Corp, NECI.
 *
 *  Includes parts of libsvm.
 *
 *
 *  This program is free software except for military or military related use; 
 *  If you don't come under the above exception you can redistribute it 
 *  and/or modify it under the terms of the GNU General Public License as 
 *  published by the Free Software Foundation; either version 2 of the 
 *  License, or (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111, USA
 *
 ***********************************************************************/



#include <stdio.h>
#include <vector>
#include <math.h>
#include <assert.h>

#include <iostream>
#include <fstream>
#include <string>
#include <limits>
#include <set>
#include <map>
#include <algorithm>

#include "mex.h" 
#include <fstream>
#include <ctime>
#include <iostream>
#include <string>

double * Cost;
int C_m;
int C_n;
int B;
double * optsol;
double * optsel;

using namespace std;

inline double cost(int i, int j){
  return Cost[(j-1)*C_m+(i-1)];
}

int solidx(int i, int k, int N){
  return (i-1)*C_m*B + (k-1)*C_m + (N-1);
}

int selidx(int i, int k){
  return (i-1)*B + (k-1);
}


double dpsolve(int i, int k, int N){
  if(N < i || i < k || i < 1 || N < 1 || k < 1)
    return 1e20;

  if(N == 1) // also implies i=1, k=1
    return 0;

  if(optsol[solidx(i, k, N)] != -1)
    return optsol[solidx(i, k, N)];

  if(N > i){
    optsol[solidx(i, k, N)] = dpsolve(i, k, N-1) + cost(i, N);
    //printf("N>i, sol[%d, %d, %d] = %f\n", i, k, N, optsol[solidx(i, k, N)]);
    return optsol[solidx(i, k, N)];
  }

  if(k == 1){
    
    //if(i != 1)
    //return 1e20;
    
    optsol[solidx(i, k, N)] = 0;
    for(int j=1;j<=i;j++)
      optsol[solidx(i, k, N)] += cost(i, j);
    //printf("k==1, sol[%d, %d, %d] = %f\n", i, k, N, optsol[solidx(i, k, N)]);
    return optsol[solidx(i, k, N)];
  }

  double bestsol = -1;
  int best = -1;
  for(int j=k-1;j<i;j++){
    double cursol = dpsolve(j, k-1, i-1);
    //printf("sol[%d, %d, %d] = %f\n", j, k-1, i-1, cursol);
    for(int l=j+1; l<i; l++)
      cursol = cursol - cost(j, l) + min(cost(j, l), cost(i, l));
    if(best == -1 || cursol < bestsol){
      bestsol = cursol;
      best = j;
    }  
  }
  
  optsol[solidx(i, k, N)] = bestsol;
  optsel[selidx(i, k)] = best;
  //printf("optsel[%d, %d] = %d, N = %d\n", i, k, best, N);
  //printf("other, sol[%d, %d, %d] = %f\n", i, k, N, optsol[solidx(i, k, N)]);
  return optsol[solidx(i, k, N)];
}

void mexFunction(int nlhs, mxArray *plhs[],
		 int nrhs, const mxArray *prhs[]) {

  C_m = mxGetM(prhs[0]);
  C_n = mxGetN(prhs[0]);

  // cost matrix
  Cost = mxGetPr(prhs[0]);
  
  //budget
  double * tmp = mxGetPr(prhs[1]);
  B = (int)tmp[0];  

  //printf("matrix size is %d, %d, budget - %d\n", C_m, C_n, B);

  optsol = new double[(C_m+1) * (C_n+1) * B];
  optsel = new double[(C_m+1) * B];
  fill_n(optsol, (C_m+1) * (C_n+1) * B, -1);
  fill_n(optsel, (C_m+1) * B, 0);

  double sol = -1;
  int best = -1;
  for(int i=C_m;i>=1;i--){
    double tmp = dpsolve(i, B, C_m);
    if(sol == -1 || tmp < sol){
      sol = tmp;
      best = i;
    }
  }

  //printf("best solution %d, %f\n", best, sol);

  // initialize plhs
  plhs[0] = mxCreateDoubleMatrix(B, 1, mxREAL);
  double *res = mxGetPr(plhs[0]);
  
  res[B-1] = best;
  for(int k=B-1; k>0;k--)
    res[k-1] = optsel[selidx(res[k], k+1)];

  if(nlhs >1){
    plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
    double * tmp = mxGetPr(plhs[1]);
    tmp[0] = sol;
  }

  delete[] optsol;
  delete[] optsel;
}
