// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*


 *	Stochastic Gradient Descent with a Decaying Learning Rate
        for Unconstrained Convex Minimization with constant step size
	Solves the problem \min_x \phi(x), where \phi is a convex (or continuous) function.
	Anthor: Rishabh Iyer
 *
	Input: 	Continuous Function: c
		   	Initial starting point x0
			step-size parameter (alpha)
			max number of iterations (maxiter)
			Tolerance (TOL)
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

#include "sgdLineSearch.h"

namespace jensen {

Vector sgdLineSearch(const ContinuousFunctions& c, const Vector& x0, const int numSamples,
		     double alpha, const int miniBatchSize, 
		     const double TOL, const int maxEval, 
		     const double gamma, const bool resetAlpha, const bool useinputAlpha,
		     const int verbosity){
	cout<<"Started Stochastic Gradient Descent with Line Search\n";
	Vector x(x0);
	double f;
	Vector g;
	Vector xnew;
	double fnew;
	Vector gnew;
	double gnorm;
	double learningRate;
	double gg;
	int epoch = 1;
	int miniBatchEval = 0;
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

	// consider first minibatch for alpha initialization and first value of xnew
	endInd = min(miniBatchSize - 1, numSamples-1);
	std::vector<int> currIndices(indices.begin(), 
				     indices.begin() + endInd);
	// todo: consider doing one entire function evaluation if !useinputAlpha, instead of just
	// using one minibatch as an estimate of norm(g)
	c.evalStochastic(x, f, g, 
			 currIndices);
	gnorm = norm(g); // check that gradient is non-zero since g is computed only over a minibatch
	if (!useinputAlpha){
	  if (gnorm){
	    alpha = 1/gnorm;
	  }
	  else {
	    printf("Warning: trying to initialize alpha = 1 / || gradient of first miniBatch || but gradient is zero.\n");
	    printf("Using supplied initial value of alpha = %f", alpha);
	    
	  }
	}

	gnorm = 1e2;	
	while ((gnorm >= TOL) && (epoch < maxEval) )
	{
		for(int i = 0; i < l - 1; i++){
		  xnew = x - alpha*g;
		  // create starting and ending indices to take a subvector of indices
		  startInd = i * miniBatchSize;
		  endInd = min((i+1) * miniBatchSize - 1, numSamples-1);
		  std::vector<int> currIndices(indices.begin() + startInd, 
					       indices.begin() + endInd);
		  c.evalStochastic(xnew, fnew, gnew, 
				   currIndices);
		  miniBatchEval++;
		  if (verbosity > 0)
		    printf("Epoch %d, minibatch %d, alpha: %f, ObjVal: %f, ObjValNew: %f, OptCond: %f\n", epoch, i, alpha, f, fnew, gnorm);

		  gg = g*g;
		  // Backtracking line search
		  while (fnew > f - gamma*alpha*gg){
		    alpha = alpha*alpha*gg/(2*(fnew + gg*alpha - f));
		    xnew = x - alpha*g;
		    c.evalStochastic(xnew, fnew, gnew,
				     currIndices);
		    miniBatchEval++;
		  }
		  if (resetAlpha)
		    alpha = min(1, 2*(f - fnew)/gg);

		  x = xnew;
		  f = fnew;
		  g = gnew;
		  gnorm = norm(g);
		  if (verbosity > 0)
		    printf("Epoch %d, minibatch %d, alpha: %f, ObjVal: %f, OptCond: %f\n", epoch, i, alpha, f, gnorm);
		}
		
		if (verbosity > 0){
		  // Evaluate total objective function with learned parameters
		  c.eval(x, f, g);
		  printf("Epoch: %d, alpha: %f, ObjVal: %f, OptCond: %f\n", epoch, alpha, f, gnorm);
		}
		epoch++;
	}
	return x;
}		
}
