// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*
 *	Abstract base class for Continuous Functions
        Author: Rishabh Iyer
 *
 */

#include <iostream>
using namespace std;

#include "SumContinuousFunctions.h"
#include <assert.h>
#define EPSILON 1e-6
namespace jensen {
SumContinuousFunctions::SumContinuousFunctions(ContinuousFunctions& c1, ContinuousFunctions& c2, double lambda) : c1(c1), c2(c2),
	ContinuousFunctions(c1.isSmooth && c2.isSmooth, c1.size(), c1.length() + c2.length()), lambda(lambda)
{
	assert(c1.size() == c2.size());         // make sure the feature dimensions of f and g are the same.
	assert(lambda >= 0);
}
SumContinuousFunctions::SumContinuousFunctions(const SumContinuousFunctions& sc) : ContinuousFunctions(sc), c1(sc.c1), c2(sc.c2), lambda(sc.lambda) {
}

SumContinuousFunctions::~SumContinuousFunctions(){
}

double SumContinuousFunctions::eval(const Vector& x) const {
	return c1.eval(x) + lambda*c2.eval(x);
}

Vector SumContinuousFunctions::evalGradient(const Vector& x) const {        // in case the function is non-differentiable, this is the subgradient
	c1.evalGradient(x) + lambda*c2.evalGradient(x);
}

void SumContinuousFunctions::eval(const Vector& x, double& f, Vector& g) const {
	double f1, f2;
	Vector g1, g2;
	c1.eval(x, f1, g1);
	c2.eval(x, f2, g2);
	f = f1 + lambda*f2;
	g = g1 + lambda*g2;
}

Vector SumContinuousFunctions::evalStochasticGradient(const Vector& x, vector<int> miniBatch) const {
	return c1.evalStochasticGradient(x, miniBatch) + c2.evalStochasticGradient(x, miniBatch);
}

void SumContinuousFunctions::evalStochastic(const Vector& x, double& f, Vector& g, std::vector<int>& miniBatch) const {
	double f1, f2;
	Vector g1, g2;
	c1.evalStochastic(x, f1, g1, miniBatch);
	c2.evalStochastic(x, f2, g2, miniBatch);
	f = f1 + lambda*f2;
	g = g1 + lambda*g2;
	return;
}
}
