// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*

   Anthor: Rishabh Iyer, John Halloran and Kai Wei
 *
        Input:  Continuous Function: c
                        Initial starting point x0
                        step-size parameter (alpha)
                        max number of iterations (maxiter)
                        Tolerance (TOL)
                        Verbosity

        Output: Output on convergence (x)
 */

#ifndef CA_GD
#define CA_GD

#include "../contFunctions/ContinuousFunctions.h"
#include "../../representation/Vector.h"
#include "../../representation/VectorOperations.h"

namespace jensen {

Vector gd(const ContinuousFunctions& c, const Vector& x0, const double alpha = 0.1,
          const int maxEval = 1000, const double TOL = 1e-3, const int verbosity = 1);

}
#endif
