// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*


 *	Online optimization with automatic step-size adjustment using AdaGrad (Duchi et al, 2010)
	Solves the problem \min_x \phi(x), where \phi is a convex (or continuous) function.
	Anthor: John Halloran
 *
	Input: 	Continuous Function: c
		   	Initial starting point x0
			Number of training/data instances/samples numSamples
			step-size parameter (alpha)
			Number of samples to compute the gradient within an epoch miniBatchSize
			max number of epochs (maxEval)
			Tolerance (TOL)
			Verbosity

	Output: Output on convergence (x)
 */

#ifndef CA_SGD_RDA_ADAGRAD
#define CA_SGD_RDA_ADAGRAD

#include "../contFunctions/ContinuousFunctions.h"
#include "../../representation/Vector.h"
#include "../../representation/VectorOperations.h"

namespace jensen {
	
Vector sgdRegularizedDualAveragingAdagrad(const ContinuousFunctions& c, const ContinuousFunctions& c2, 
					  const Vector& x0, const int numSamples,
					  const double alpha = 0.1, const double lambda = 1.0,
					  const int miniBatchSize = 1,
					  const double TOL = 1e-3, const int maxEval = 1000, const int verbosity = 1);

}
#endif
