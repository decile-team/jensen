// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*


        Jensen: A Convex Optimization And Machine Learning ToolKit
 *	Abstract base class for Continuous Functions
        Author: Rishabh Iyer
 *
 */

#ifndef CONTINUOUS_FUNCTIONS_H
#define CONTINUOUS_FUNCTIONS_H

#include "../../representation/Vector.h"
#include "../../representation/Matrix.h"
#include "../../representation/VectorOperations.h"
#include "../../representation/MatrixOperations.h"

namespace jensen {

class ContinuousFunctions {
protected:
int n;                  // The number of convex functions added together, i.e if g(X) = \sum_{i = 1}^n f_i(x)
int m;                 // Dimension of vectors or features (i.e. size of x in f(x))
public:
bool isSmooth;
ContinuousFunctions(bool isSmooth);
ContinuousFunctions(bool isSmooth, int m, int n);
ContinuousFunctions(const ContinuousFunctions& c);         // copy constructor

virtual ~ContinuousFunctions();

virtual double eval(const Vector& x) const;                 // functionEval
virtual Vector evalGradient(const Vector& x) const;                 // gradientEval
virtual void eval(const Vector& x, double& f, Vector& gradient) const;                 // combined function and gradient eval
virtual Vector evalStochasticGradient(const Vector& x, std::vector<int>& batch) const;                 // stochastic gradient
virtual void evalStochastic(const Vector& x, double& f, Vector& g, std::vector<int>& miniBatch) const;                 // stochastic combined evaluation
virtual Matrix evalHessian(const Vector& x) const;                      // hessianEval
virtual void evalHessianVectorProduct(const Vector& x, const Vector& v, Vector& Hxv) const;                 // evaluate a product between a hessian and a vector
double operator()(const Vector& x) const;

int size() const;                 // number of features or dimension size (m)
int length() const;                 // number of convex functions adding up (n)
};

}
#endif
