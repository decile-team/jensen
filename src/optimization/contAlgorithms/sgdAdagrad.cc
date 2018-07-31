// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*


 *	Stochastic gradient descent with automatic step-size adjustment using AdaGrad (Duchi et al, 2010)
   Solves the problem \min_x \phi(x), where \phi is a convex (or continuous) function.
   Anthor: John Halloran
 *
   Input:       Continuous Function: c
   Initial starting point x0
   Number of training/data instances/samples numSamples
   step-size parameter (alpha)
   Number of samples to compute the gradient within an epoch miniBatchSize
   max number of epochs (maxEval)
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

#include "sgdAdagrad.h"

namespace jensen {

Vector sgdAdagrad(const ContinuousFunctions& c, const Vector& x0, const int numSamples,
                  const double alpha, const int miniBatchSize,
                  const double TOL, const int maxEval, const int verbosity){
	cout<<"Started Stochastic Gradient Descent with AdaGrad\n";
	Vector x(x0);
	double f = 1e30;
	double f0 = 1e30;
	Vector g(x.size(), 0.0), g2(x.size(), 0.0);
	int gnormType = 1; // use L1 norm, set to 2 if L2 is desired
	double gnorm = 1e2;
	int epoch = 1;
	int startInd = 0;
	int endInd = 0;
	// number of minibatches
	int l = numSamples / miniBatchSize;
	double denom = miniBatchSize * numSamples;

	// create vector of indices and randomly permute
	std::vector<int> indices;
	for(int i = 0; i < numSamples; i++) {
		indices.push_back(i);
	}
	std::random_shuffle( indices.begin(), indices.end() );

	std::vector <std::vector<int> > allIndices = std::vector <std::vector<int> >(l-1);
	for (int i = 0; i < l-1; i++) {
		startInd = i * miniBatchSize;
		endInd = min((i+1) * miniBatchSize - 1, numSamples-1);
		std::vector<int> currIndices(indices.begin() + startInd, indices.begin() + endInd);
		allIndices[i] = currIndices;
	}

	// adagrad's running some of squared gradients
	Vector g2RunningSum(x0.size(), 1e-5); // for feature i at iteration j, \sum_{t = 1}^j g^2_{t,i}
	while ((gnorm >= TOL) && (epoch < maxEval) )
	{
		f0 = 0.0; // calculate average reduction in objective value
		gnorm = 0.0; // calculate average reduction in the gradient
		for(int i = 0; i < l - 1; i++) {
			g2 = g;
			c.evalStochastic(x, f, g,
			                 allIndices[i]);

			// // update running sum of squares of encountered gradients
			g2RunningSum += elementMultiplication(g,g);
			x = x - alpha * elementMultiplication(elementPower(g2RunningSum, -0.5), g); // adagrad, x = x - alpha * (g ./ (g2RunningSum .^ 1/2))
			f0 += f / denom;
			gnorm += norm(g2-g, gnormType) / denom;
			if (verbosity > 2)
				printf("Epoch %d, minibatch %d, alpha: %e, ObjVal: %e, OptCond: %e\n", epoch, i, alpha, f, gnorm);
		}

		if (verbosity > 1) {
			// Evaluate total objective function with learned parameters
			c.eval(x, f, g);
			gnorm = norm(g);
			printf("Epoch: %d, alpha: %e, ObjVal: %e, OptCond: %e\n", epoch, alpha, f, gnorm);
		}else{
			printf("Epoch: %d, alpha: %e, Avg. ObjVal Reduction: %e, Avg. Grad. Reduction: %e\n", epoch, alpha, f0, gnorm);
		}
		epoch++;
	}
	if (verbosity > 1) {
		// Evaluate total objective function with learned parameters
		c.eval(x, f, g);
		gnorm = norm(g);
		printf("Epoch: %d, alpha: %e, ObjVal: %e, OptCond: %e\n", epoch, alpha, f, gnorm);
	} else {
		printf("Epoch: %d, alpha: %e, Avg. ObjVal Reduction: %e, Avg. Grad. Reduction: %e\n", epoch, alpha, f0, gnorm);
	}

	return x;
}
}
