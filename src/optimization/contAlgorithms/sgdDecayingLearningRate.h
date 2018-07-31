// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*


 *	Stochastic gradient descent with a decaying learning rate.  The rate of decay is
        (t) ^ (-decayRate), where t is the number of considered minibatches.
        Solves the problem \min_x \phi(x), where \phi is a convex (or continuous) function.
        Anthor: John Halloran
 *
        Input:  Continuous Function: c
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

#ifndef CA_SGD_DECAYING_LEARNING_RATE
#define CA_SGD_DECAYING_LEARNING_RATE

#include "../contFunctions/ContinuousFunctions.h"
#include "../../representation/Vector.h"
#include "../../representation/VectorOperations.h"

namespace jensen {

Vector sgdDecayingLearningRate(const ContinuousFunctions& c, const Vector& x0, const int numSamples,
                               const double alpha = 0.1, const int miniBatchSize = 1,
                               const double TOL = 1e-3, const int maxEval = 1000,
                               const double decayRate = 0.5, const int verbosity = 1);
}
#endif
