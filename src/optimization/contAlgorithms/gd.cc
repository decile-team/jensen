// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*

Anthor: Rishabh Iyer, John Halloran and Kai Wei

 *	Gradient Descent for Unconstrained Convex Minimization with backtracking line search
	Solves the problem \min_x \phi(x), where \phi is a convex (or continuous) function.
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
using namespace std;

#include "gd.h"

namespace jensen {

Vector gd(const ContinuousFunctions& c, const Vector& x0, const double alpha,
const int maxEval, const double TOL, const int verbosity){
	cout<<"Started Gradient Descent\n";
	Vector x(x0);
	double f;
	Vector g;
	c.eval(x, f, g);
	double gnorm = norm(g);
	int funcEval = 1;
	while ((gnorm >= TOL) && (funcEval < maxEval) )
	{
		multiplyAccumulate(x, alpha, g);
		c.eval(x, f, g);
		funcEval++;
		gnorm = norm(g);
		if (verbosity > 0)
			printf("numIter: %d, alpha: %e, ObjVal: %e, OptCond: %e\n", funcEval, alpha, f, gnorm);
	}
	return x;
}
}
