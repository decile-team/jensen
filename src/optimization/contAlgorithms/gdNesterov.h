// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*


 *	Gradient Descent for Unconstrained Convex Minimization with Nesterov
	Solves the problem \min_x \phi(x), where \phi is a convex (or continuous) function.
	Anthor: Rishabh Iyer
 *
	Input: 	Continuous Function: c
		   	Initial starting point x0
			Initial step-size (alpha)
			back-tracking parameter (gamma)
			max number of function evaluations (maxEvals)
			Tolerance (TOL)
			resetAlpha (whether to reset alpha at every iteration or not)
			verbosity

	Output: Output on convergence (x)
 */

#ifndef CA_GD_NESTEROV
#define CA_GD_NESTEROV

#include "../contFunctions/ContinuousFunctions.h"
#include "../../representation/Vector.h"
#include "../../representation/VectorOperations.h"

namespace jensen {
	
Vector gdNesterov(const ContinuousFunctions& c, const Vector& x0, double alpha = 1, const double gamma = 1e-4, 
const int maxEval = 1000, const double TOL = 1e-3, bool resetAlpha = true, bool useinputAlpha = false, int verbosity = 1);

}
#endif
