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

#include "sgdRegularizedDualAveraging.h"

namespace jensen {

  inline void regularizedDualAverageUpdate(const Vector& g,
					   const int numIter, const double stepSize, 
					   const double l1Tolerance, const double decayRate,
					   Vector& x)
{
  double absG = 0;

  x = Vector(g.size(), 0);
  for (int i = 0; i < g.size(); i++){
    absG = std::abs(g[i]);
    if (absG <= l1Tolerance)
      x[i] = 0;
    else
      x[i] = -sign(g[i]) * pow(double(numIter), decayRate) * stepSize * (absG / numIter - l1Tolerance);
      // x[i] = -stepSize * numIter / pow(g2[i], 0.5) * (g[i] / double(numIter) - sign(g[i]) * l1Tolerance);
  }
}

  Vector sgdRegularizedDualAveraging(const ContinuousFunctions& c, const ContinuousFunctions& c2,
				     const Vector& x0, const int numSamples,
				     const double alpha, const double lambda,
				     const int miniBatchSize, 
				     const double TOL, const int maxEval, 
				     const double decayRate, const int verbosity){
	cout<<"Started Stochastic Gradient Descent with AdaGrad\n";
	Vector x = Vector(x0.size(), 0); // implicit assumption of RDA algorithm
	double f = 1e30;
	double f0 = 1e30;
	Vector g;
	double gnorm;
	int epoch = 1;
	int startInd = 0;
	int endInd = 0;
	int miniBatchEval = 1;
	// number of minibatches
	int l = numSamples / miniBatchSize;
	
	// create vector of indices and randomly permute
	std::vector<int> indices;
	for(int i = 0; i < numSamples; i++){
	  indices.push_back(i);
	}
	std::random_shuffle( indices.begin(), indices.end() );
	gnorm = 1e2;

	// dual-averaging average sum of gradients
	Vector gRunningSum(x0.size(), 0); // for feature i at iteration j, \sum_{t = 1}^j g^2_{t,i}
	while ((gnorm >= TOL) && (epoch < maxEval) )
	{
		for(int i = 0; i < l - 1; i++){
		  // create starting and ending indices to take a subvector of indices
		  startInd = i * miniBatchSize;
		  endInd = min((i+1) * miniBatchSize - 1, numSamples-1);
		  std::vector<int> currIndices(indices.begin() + startInd, 
					       indices.begin() + endInd);
		  f0 = f;
		  c.evalStochastic(x, f, g, 
				   currIndices);

		  gRunningSum += g;
		  regularizedDualAverageUpdate(gRunningSum, miniBatchEval, alpha, lambda, decayRate, x);
		  miniBatchEval++;
		  if (verbosity > 1)
		    printf("Epoch %d, minibatch %d, alpha: %f, ObjVal: %f, OptCond: %f\n", epoch, i, alpha, f, gnorm);
		}

		if (verbosity > 1){
		  // Evaluate total objective function with learned parameters
		  c2.eval(x, f, g);
		  gnorm = norm(g);
		  printf("Epoch: %d, alpha: %f, ObjVal: %f, OptCond: %f\n", epoch, alpha, f, gnorm);
		}else{
		  gnorm = fabs(f0-f);
		  printf("Epoch: %d, alpha: %f, ObjVal: %f, OptCond: %f\n", epoch, alpha, f, gnorm);
		}
		
		epoch++;
	}
	if (verbosity > 0){
	  // Evaluate total objective function with learned parameters
	  c2.eval(x, f, g);
	  gnorm = norm(g);
	  printf("Epoch: %d, alpha: %f, ObjVal: %f, OptCond: %f\n", epoch, alpha, f, gnorm);
	}

	return x;
}		
}
