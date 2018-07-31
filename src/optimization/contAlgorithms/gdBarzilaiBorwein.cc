// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*


 *	Gradient Descent for Unconstrained Convex Minimization with backtracking line search and Barzilai-Borwein
        step length

        Solves the problem \min_x \phi(x), where \phi is a convex (or continuous) function.
        Anthor: Rishabh Iyer, John Halloran and Kai Wei
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

#include "gdBarzilaiBorwein.h"
namespace jensen {

Vector gdBarzilaiBorwein(const ContinuousFunctions& c, const Vector& x0, double alpha, const double gamma,
                         const int maxEval, const double TOL, bool useinputAlpha, int verbosity){
	Vector x(x0);
	Vector g;
	double f;
	c.eval(x, f, g);
	Vector xnew;
	double fnew;
	Vector gnew;
	double gnorm = norm(g);
	int funcEval = 1;
	if (!useinputAlpha)
		alpha = 1/norm(g);
	while ((gnorm >= TOL) && (funcEval < maxEval) )
	{
		double gg = g*g;
		xnew = x - alpha*g;
		c.eval(xnew, fnew, gnew);
		funcEval++;
		// double fgoal = f - gamma*alpha*gg;
		// Backtracking line search
		while (fnew > f - gamma*alpha*gg) {
			alpha = alpha*alpha*gg/(2*(fnew + gg*alpha - f));
			// printf("alpha: %e, fnew = %e, fgoal = %e\n", alpha, fnew, fgoal);
			xnew = x - alpha*g;
			c.eval(xnew, fnew, gnew);
			funcEval++;
		}
		Vector gdiff = gnew - g;
		// Barzilai-Borwein update
		alpha = -alpha*(g*gdiff)/(gdiff*gdiff);
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
