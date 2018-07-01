// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*
 *	Gradient Descent for Unconstrained Convex Minimization with constant step size
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

#ifndef CA_L1_LR_PRIMAL
#define CA_L1_LR_PRIMAL

#include "../../../representation/Vector.h"
#include "../../../representation/SparseFeature.h"

namespace jensen {
	
void L1LRPrimal(std::vector<SparseFeature>& features, Vector& y, Vector& x, double C, double eps);

}
#endif
