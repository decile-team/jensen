// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*


 *	Stochastic gradient descent with a decaying learning rate.  The rate of decay is
 (t) ^ (-decayRate), where t is the number of considered minibatches.
 Solves the problem \min_x \phi(x), where \phi is a convex (or continuous) function.
 Anthor: John Halloran
 *
 Input: 	Continuous Function: c
 Initial starting point x0
 Number of training/data instances/samples numSamples
 step-size parameter (alpha)
 Number of samples to compute the gradient within an epoch miniBatchSize
 Tolerance (TOL)
 max number of epochs (maxEval)
 power dictating learning rate's decay decayRate
 Verbosity

 Output: Output on convergence (x)
*/

#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include "../../utils/utils.h"
using namespace std;

#include "sgdDecayingLearningRate.h"

namespace jensen {

  Vector sgdDecayingLearningRate(const ContinuousFunctions& c, const Vector& x0, const int numSamples,
				 const double alpha, const int miniBatchSize, 
				 const double TOL, const int maxEval, 
				 const double decayRate, const int verbosity){
    cout<<"Started Stochastic Gradient Descent with Decaying Learning Rate\n";
    Vector x(x0);
    double f = 1e30;
    double f0 = 1e30;
    Vector g;
    double gnorm;
    double learningRate;
    int epoch = 1;
    int miniBatchEval = 1;
    int startInd = 0;
    int endInd = 0;
    // number of minibatches
    int l = numSamples / miniBatchSize;
	
    // create vector of indices and randomly permute
    std::vector<int> indices;
    for(int i = 0; i < numSamples; i++){
      indices.push_back(i);
    }
    std::random_shuffle( indices.begin(), indices.end() );
    gnorm = 1e2;

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
	  // learningRate = alpha / pow(miniBatchEval, decayRate);
	  // learningRate = alpha / pow(miniBatchEval, decayRate);
	  learningRate = alpha / (1 + alpha * miniBatchEval);
	  x = x - learningRate * g;
	  if (verbosity > 1)
	    printf("Epoch %d, minibatch %d, alpha: %f, ObjVal: %f, OptCond: %f\n", epoch, i, alpha, f, gnorm);
	  miniBatchEval++;
	}
	if (verbosity > 1){
	  // Evaluate total objective function with learned parameters
	  c.eval(x, f, g);
	  gnorm = norm(g);
	  printf("Epoch: %d, alpha: %f, ObjVal: %f, OptCond: %f\n", epoch, alpha, f, gnorm);
	}else{
	  gnorm = fabs(f0-f);
	  printf("Epoch: %d, alpha: %f, ObjVal: %f, OptCond: %f\n", epoch, alpha, f, gnorm);
	}
	epoch++;
      }
    if (verbosity > 1){
      // Evaluate total objective function with learned parameters
      c.eval(x, f, g);
      gnorm = norm(g);
      printf("Epoch: %d, alpha: %f, ObjVal: %f, OptCond: %f\n", epoch, alpha, f, gnorm);
    }
    return x;
  }		
}
