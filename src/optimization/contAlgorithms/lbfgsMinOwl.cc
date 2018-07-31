// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*


 *	LBFGS with orthant-wise projection (Algorithm from Andrew and Gao, 2007)
        Solves the problem \min_x L(x) + |x|_1, i.e L1 regularized optimization problems.
        This algorithm directly encourages sparsity.
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
#include <math.h>
#include <algorithm>
#include <iostream>
using namespace std;

#include "lbfgsMinOwl.h"
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
		Q[i] = Q[i+1] - al[i]*Y[i];
	}
	R[0] = h*Q[1];
	for (int i = 0; i < k; i++) {
		be[i] = ro[i]*(Y[i]*R[i]);
		R[i+1] = R[i] + S[i]*(al[i] - be[i]);
	}
	d = R[k];
	return;
}

inline void orthantProject(const Vector& x, const Vector& xi, Vector& xp)
{
	xp = Vector(x.size(), 0);
	for (int i = 0; i < x.size(); i++) {
		if (sign(x[i]) != xi[i])
			xp[i] = 0;
		else
			xp[i] = x[i];
	}
}

Vector lbfgsMinOwl(const ContinuousFunctions& c, const Vector& x0, double alpha, const double gamma,
                   const int maxEval, const int memory, const double TOL, bool resetAlpha, bool useinputAlpha, int verbosity){
	Vector x(x0);
	Vector g;
	double f;
	c.eval(x, f, g);
	double gnorm = norm(g);
	double fnorm = 1e30;
	int funcEval = 1;
	Vector xnew;
	double fnew;
	Vector gnew;
	Vector xold;
	double fold;
	Vector gold;

	Vector d = g; // lbfgs direction
	Matrix S;
	Matrix Y;
	double h = 1;
	if (!useinputAlpha)
		alpha = 1/norm(g);
	while ( (fnorm >= TOL) && (funcEval < maxEval) )
	{
		if (funcEval > 1) {
			Vector y = g - gold;
			Vector s = x - xold;
			lbfgsUpdate(S, Y, s, y, memory);
			h = (y*s)/(y*y);
			lbfgsDirection(g, S, Y, h, d);
			alpha = min(1, 2*(fold - f)/(g*d));
		}
		fold = f;
		gold = g;
		xold = x;
		Vector xi = sign(x);
		for (int i = 0; i < x.size(); i++) {
			if (sign(d[i]) != sign(g[i]))
				d[i] = 0;
			if (x[i] == 0)
				xi[i] = sign(-g[i]);
		}
		orthantProject(x - alpha*d, xi, xnew);
		c.eval(xnew, fnew, gnew);
		funcEval++;

		double gd = g*d;
		// double fgoal = f - gamma*alpha*gd;
		// Backtracking line search
		while (fnew > f - gamma*g*(xnew - x)) {
			alpha = alpha*alpha*gd/(2*(fnew + gd*alpha - f));
			orthantProject(x - alpha*d, xi, xnew);
			c.eval(xnew, fnew, gnew);
			funcEval++;
		}
		fnorm = fabs(f - fnew);
		x = xnew;
		f = fnew;
		g = gnew;
		gnorm = norm(g);
		if (verbosity > 0)
			printf("numIter: %d, alpha: %e, ObjVal: %e, OptCond: %e\n", funcEval, alpha, f, fnorm);
		if (resetAlpha)
			alpha = 1;
	}
	return x;
}

}
