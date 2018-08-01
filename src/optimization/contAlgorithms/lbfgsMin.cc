// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*


 *	Gradient Descent for Unconstrained Convex Minimization with backtracking line search
        Solves the problem \min_x \phi(x), where \phi is a convex (or continuous) function.
        Author: Rishabh Iyer
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

#include "lbfgsMin.h"
#include "../../utils/utils.h"
namespace jensen {

inline void lbfgsUpdate(Matrix& S, Matrix& Y, const Vector& s, const Vector& y, int memory){
	if (Y.numRows() < memory) {
		S.push_back(s);
		Y.push_back(y);
	}
	else {
		S.remove(0);
		Y.remove(0);
		S.push_back(s);
		Y.push_back(y);
	}
}

inline void lbfgsDirection(const Vector& g, const Matrix& S, const Matrix& Y, const double h, Vector& d){
	int k = S.numRows();
	int p = S.numColumns();
	Vector ro;
	for (int i = 0; i < k; i++)
		ro.push_back(1/(Y[i]*S[i]));
	Matrix Q(k+1, p);
	Matrix R(k+1, p);
	Vector al(k, 0);
	Vector be(k, 0);
	Q[k] = g;
	for (int i = k-1; i >= 0; i--) {
		al[i] = ro[i]*(S[i]*Q[i+1]);
		// Q[i] = Q[i+1] - al[i]*Y[i];
		multiplyAccumulate(Q[i], Q[i+1], al[i], Y[i]);
	}
	R[0] = h*Q[1];
	for (int i = 0; i < k; i++) {
		be[i] = ro[i]*(Y[i]*R[i]);
		// R[i+1] = R[i] + S[i]*(al[i] - be[i]);
		multiplyAccumulate(R[i+1], R[i], be[i] - al[i], S[i]);
	}
	d = R[k];
	return;
}

Vector lbfgsMin(const ContinuousFunctions& c, const Vector& x0, double alpha, const double gamma,
                const int maxEval, const int memory, const double TOL, bool resetAlpha, bool useinputAlpha, int verbosity){
	Vector x(x0);
	Vector g;
	double f;
	c.eval(x, f, g);
	Vector xnew;
	double fnew;
	Vector gnew;
	double gnorm = norm(g);
	int funcEval = 1;
	Vector d = g; // lbfgs direction
	Matrix S;
	Matrix Y;
	double h = 1;
	if (!useinputAlpha)
		alpha = 1/norm(g);
	while ((gnorm >= TOL) && (funcEval < maxEval) )
	{
		// xnew = x - alpha*d;
		multiplyAccumulate(xnew, x, alpha, d);
		c.eval(xnew, fnew, gnew);
		funcEval++;

		double gd = g*d;
		// double fgoal = f - gamma*alpha*gd;
		// Backtracking line search
		while (fnew > f - gamma*alpha*gd) {
			alpha = alpha*alpha*gd/(2*(fnew + gd*alpha - f));
			// xnew = x - alpha*d;
			multiplyAccumulate(xnew, x, alpha, d);
			c.eval(xnew, fnew, gnew);
			funcEval++;
		}

		Vector gdiff = gnew - g;
		lbfgsUpdate(S, Y, -alpha*d, gdiff, memory);
		h = -alpha*(d*gdiff)/(gdiff*gdiff);
		x = xnew;
		f = fnew;
		g = gnew;
		lbfgsDirection(g, S, Y, h, d);
		gnorm = norm(g);
		if (verbosity > 0)
			printf("numIter: %d, alpha: %e, ObjVal: %e, OptCond: %e\n", funcEval, alpha, f, gnorm);
		if (resetAlpha)
			alpha = 1;

	}
	return x;
}

}
