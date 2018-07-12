// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*


 *	Stochastic gradient descent using the regularized dual-averaging algorithm (Xiao, 2009)
        to promote sparseness (i.e., L1 regularization), with automatic step-size adjustment 
	using AdaGrad (Duchi et al, 2010).
	Solves the problem \min_x L(x) + |x|_1, i.e L1 regularized optimization problems.
	Anthor: John Halloran
 *
	Input: 	Continuous Function: c
			Number of training/data instances/samples (numSamples)
			step-size parameter (alpha)
			regularization threshold (lambda)
			Number of samples to compute the gradient within an epoch (miniBatchSize)
			max number of epochs (maxEval)
			Tolerance (TOL)
			Verbosity

	Output: Output on convergence (x)

	Note: The algorithm specifically assumes the initial weights are x0 = 0
 */

#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include "../../utils/utils.h"
using namespace std;

#include "sgdRegularizedDualAveragingAdagrad.h"

namespace jensen {

  inline void regularizedDualAverageUpdate(const Vector& g, const Vector& g2, 
					   const int numIter, const double stepSize, 
					   const double l1Tolerance,
					   Vector& x)
  {
    double absG = 0;

    x = Vector(g.size(), 0);
    for (int i = 0; i < g.size(); i++){
      absG = std::abs(g[i]);
      if (absG <= l1Tolerance)
	x[i] = 0;
      else
	x[i] = -sign(g[i]) * stepSize * numIter * (absG / numIter - l1Tolerance) / pow(g2[i], 0.5);
      // x[i] = -stepSize * numIter / pow(g2[i], 0.5) * (g[i] / double(numIter) - sign(g[i]) * l1Tolerance);
    }
  }

  Vector sgdRegularizedDualAveragingAdagrad(const ContinuousFunctions& c, const ContinuousFunctions& c2, 
					    const Vector& x0, const int numSamples,
					    const double alpha, const double lambda,
					    const int miniBatchSize, 
					    const double TOL, const int maxEval, const int verbosity){
    cout<<"Started Stochastic Gradient Descent with AdaGrad\n";
    Vector x(x0.size(), 0); // implicit assumption of RDA algorithm
    double f = 1e30;
    double f0 = 1e30;
    Vector g(x.size(), 0.0), g2(x.size(), 0.0);
    int gnormType = 1; // use L1 norm, set to 2 if L2 is desired
    double gnorm = 1e2;
    int epoch = 1;
    int startInd = 0;
    int endInd = 0;
    int miniBatchEval = 1;
    // number of minibatches
    int l = numSamples / miniBatchSize;
    double denom = miniBatchSize * numSamples;	

    // create vector of indices and randomly permute
    std::vector<int> indices;
    for(int i = 0; i < numSamples; i++){
      indices.push_back(i);
    }
    std::random_shuffle( indices.begin(), indices.end() );

    std::vector <std::vector<int> > allIndices = std::vector <std::vector<int> >(l-1);
    for (int i = 0; i < l-1; i++){
      startInd = i * miniBatchSize;
      endInd = min((i+1) * miniBatchSize - 1, numSamples-1);
      std::vector<int> currIndices(indices.begin() + startInd, indices.begin() + endInd);
      allIndices[i] = currIndices;
    }

    // dual-averaging average sum of gradients
    Vector gRunningSum(x0.size(), 0); // for feature i at iteration j, \sum_{t = 1}^j g^2_{t,i}
    // adagrad's running sum of squared gradients
    Vector g2RunningSum(x0.size(), 1e-5); // for feature i at iteration j, \sum_{t = 1}^j g^2_{t,i}
    while ((gnorm >= TOL) && (epoch < maxEval) )
      {
	f0 = 0.0; // calculate average reduction in objective value
	gnorm = 0.0; // calculate average reduction in the gradient
	for(int i = 0; i < l - 1; i++){
	  g2 = g;
	  c.evalStochastic(x, f, g, 
			   allIndices[i]);

	  gRunningSum += g;
	  // update running sum of squares of encountered gradients
	  g2RunningSum += elementMultiplication(g,g);
	  regularizedDualAverageUpdate(gRunningSum, g2RunningSum, miniBatchEval, alpha, lambda, x);
	  miniBatchEval++;

	  f0 += f / denom;
	  gnorm += norm(g2-g, gnormType) / denom;

	  if (verbosity > 2)
	    printf("Epoch %d, minibatch %d, alpha: %e, ObjVal: %e, OptCond: %e\n", epoch, i, alpha, f, norm(g));
	}
	if (verbosity > 1){
	  // Evaluate total objective function with learned parameters
	  c2.eval(x, f, g);
	  gnorm = norm(g);
	  printf("Epoch: %d, alpha: %e, ObjVal: %e, OptCond: %e\n", epoch, alpha, f, gnorm);
	}else{
	  printf("Epoch: %d, alpha: %e, Avg. ObjVal Reduction: %e, Avg. Grad. Reduction: %e\n", epoch, alpha, f0, gnorm);
	}
	epoch++;
      }
    if (verbosity > 0){
      // Evaluate total objective function with learned parameters
      c2.eval(x, f, g);
      gnorm = norm(g);
      printf("Epoch: %d, alpha: %e, ObjVal: %e, OptCond: %e\n", epoch, alpha, f, gnorm);
    } else {
      printf("Epoch: %d, alpha: %e, Avg. ObjVal Reduction: %e, Avg. Grad. Reduction: %e\n", epoch, alpha, f0, gnorm);
    }
    return x;
  }		
}
