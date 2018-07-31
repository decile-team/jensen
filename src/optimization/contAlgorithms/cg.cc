// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*

   Anthor: Rishabh Iyer, John Halloran and Kai Wei

 *	Gradient Descent for Unconstrained Convex Minimization with conjugate gradient descent
        Solves the problem \min_x \phi(x), where \phi is a convex (or continuous) function.
 *
        Input:  Continuous Function: c
                        Initial starting point x0
                        Initial step-size (alpha)
                        back-tracking parameter (gamma)
                        max number of function evaluations (maxEvals)
                        Tolerance (TOL)
                        resetAlpha (whether to reset alpha at every iteration or not)
                        verbosity

        Output: Output on convergence (x)
 */

#include <stdio.h>
#include <algorithm>
#include <iostream>
using namespace std;

#include "cg.h"
#include "../../utils/utils.h"
namespace jensen {

Vector cg(const ContinuousFunctions& c, const Vector& x0, double alpha, const double gamma,
          const int maxEval, const double TOL, bool resetAlpha, bool useinputAlpha, int verbosity){
	Vector x(x0);
	Vector g;
	double f;
	c.eval(x, f, g);
	Vector d = g; // conjugate direction
	double fnew;
	Vector gnew;
	Vector xnew;
	double beta;
	double gnorm = norm(g);
	int funcEval = 1;
	if (!useinputAlpha)
		alpha = 1/norm(g);
	while ((gnorm >= TOL) && (funcEval < maxEval) )
	{
		xnew = x - alpha*d;
		c.eval(xnew, fnew, gnew);
		funcEval++;
		double gg = g*g;
		double gd = g*d;
		// double fgoal = f - gamma*alpha*gd;
		// Backtracking line search
		while (fnew > f - gamma*alpha*gd) {
			alpha = alpha*alpha*gd/(2*(fnew + gd*alpha - f));
			xnew = x - alpha*d;
			c.eval(xnew, fnew, gnew);
			funcEval++;
		}
		double ggnew = gnew*gnew;
		if (resetAlpha)
			alpha = min(1, 2*(f - fnew)/ggnew);
		beta = ggnew/gg;
		d = gnew + beta*d;
		x = xnew;
		f = fnew;
		g = gnew;
		gnorm = norm(g);
		if (verbosity > 0)
			printf("numIter: %d, alpha: %e, ObjVal: %e, OptCond: %e\n", funcEval, alpha, f, gnorm);
	}
	return x;
}
}
