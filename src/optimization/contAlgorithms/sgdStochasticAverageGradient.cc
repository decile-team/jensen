// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*


 *	Stochastic gradient descent using the stochastic average gradient algorithm (Le Roux et al, 2012).
	Solves the problem \min_x L(x), where L is assumed to be strongly convex.
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

#include "sgdStochasticAverageGradient.h"

namespace jensen {

  Vector sgdStochasticAverageGradient(const ContinuousFunctions& c, const Vector& x0, const int numSamples,
				      const double alpha, const double lambda,
				      const int miniBatchSize, 
				      const double TOL, const int maxEval, 
				      const double decayRate, const int verbosity){
	cout<<"Started Stochastic Gradient Descent with AdaGrad\n";
	Vector x = Vector(x0); // implicit assumption of RDA algorithm
	std::vector<Vector> grads(numSamples);
	int randSample;
	double f;
	Vector g;
	Vector gRunningSum(x0.size(), 0);
	double gnorm;
	double learningRate = 0.0;
	double lipConst = 0.0;
	int epoch = 1;
	int startInd = 0;
	int endInd = 0;
	int miniBatchEval = 1;
	// number of minibatches
	int l = numSamples / miniBatchSize;
	
	// create vector of indices and randomly permute
	std::vector<int> indices;
	for(int i = 0; i < numSamples; i++){
	  grads[i] = Vector(x0.size(),0);
	  indices.push_back(i);
	}
	std::random_shuffle( indices.begin(), indices.end() );

	// initialize gradients and running sum
	for(int i = 0; i < l - 1; i++){
	  // create starting and ending indices to take a subvector of indices
	  startInd = i * miniBatchSize;
	  endInd = min((i+1) * miniBatchSize - 1, numSamples-1);
	  std::vector<int> currIndices(indices.begin() + startInd, 
				       indices.begin() + endInd);
	  c.evalStochastic(x, f, g, 
			   currIndices);
	  gnorm = pow(norm(g), 2);
	  if(gnorm > lipConst)
	    lipConst = gnorm;
	}
	learningRate = 1/(0.25 * lipConst + lambda);
	// learningRate =  1/ (16 * alpha);

	gnorm = 1e2;
	while ((gnorm >= TOL) && (epoch < maxEval) )
	{
		// for(int i = 0; i < l - 1; i++){
		  randSample = (rand() % (l-1));
		  // create starting and ending indices to take a subvector of indices
		  startInd = randSample * miniBatchSize;
		  endInd = min((randSample+1) * miniBatchSize - 1, numSamples-1);
		  std::vector<int> currIndices(indices.begin() + startInd, 
					       indices.begin() + endInd);

		  c.evalStochastic(x, f, g, 
				   currIndices);

		  gRunningSum -= (grads[randSample] - g);

		  // learningRate = alpha / pow(miniBatchEval, decayRate);
		  // learningRate = alpha;

		  x = x - (learningRate / l) * gRunningSum;

		  grads[startInd] = g;

		//   miniBatchEval++;
		//   if (verbosity > 1)
		//     printf("Epoch %d, minibatch %d, alpha: %f, ObjVal: %f, OptCond: %f\n", epoch, i, alpha, f, gnorm);
		// }

		// Evaluate total objective function with learned parameters
		if (verbosity > 0){
		  c.eval(x, f, g);
		  gnorm = norm(g);
		  printf("Epoch: %d, alpha: %f, ObjVal: %f, OptCond: %f\n", epoch, alpha, f, gnorm);
		}
		epoch++;
	}
	return x;
}		
}
