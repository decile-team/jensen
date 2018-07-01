// Copyright (c) 2007-2015 The LIBLINEAR Project.
// Modified for use in Jensen by Rishabh Iyer
/*


 *	Trust Region Newton Method, using Conjugate Gradient at every iteration
	Solves the problem \min_x L(x) + |x|_1, i.e L1 regularized optimization problems.
	This algorithm directly encourages sparsity.
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

#include <stdio.h>
#include <algorithm>
#include <iostream>
using namespace std;

#include "tron.h"
#include "../../utils/utils.h"
#include <math.h>
namespace jensen {

  int trcg(double delta, const ContinuousFunctions& c, const Vector& g, const Vector& x, Vector& s, Vector &r, bool* reach_boundary, double eps_cg = 0.1)
  {
    int i;
    int m = c.size();
    Vector d(m, 0);
    Vector Hd(m, 0);
    double rTr, rnewTrnew, alpha, beta, cgtol;

    *reach_boundary = false;
    for (i=0; i<m; i++)
      {
	s[i] = 0;
	r[i] = -g[i];
	d[i] = r[i];
      }
    cgtol = eps_cg*norm(g);
    int cg_iter = 0;
    rTr = r*r;
    while (1)
      {
	if (norm(r) <= cgtol)
	  break;
	cg_iter++;
	c.evalHessianVectorProduct(x, d, Hd);

	alpha = rTr/(d*Hd);
	s += alpha*d;
	if (norm(s) > delta)
	  {
	    cout << "cg reaches trust region boundary\n";
	    *reach_boundary = true;
	    alpha = -alpha;
	    s += alpha*d;

	    double std = s*d;
	    double sts = s*s;
	    double dtd = d*d;
	    double dsq = delta*delta;
	    double rad = sqrt(std*std + dtd*(dsq-sts));
	    if (std >= 0)
	      alpha = (dsq - sts)/(std + rad);
	    else
	      alpha = (rad - std)/dtd;
	    s += d*alpha;
	    alpha = -alpha;
	    r += Hd*alpha;
	    break;
	  }
	alpha = -alpha;
	r += Hd*alpha;
	rnewTrnew = r*r;
	beta = rnewTrnew/rTr;
	d *= beta;
	d += r;
	rTr = rnewTrnew;
      }
    return cg_iter;
  }


  Vector tron(const ContinuousFunctions& c, const Vector& x0, const int maxEval, const double TOL, int verbosity){
    double eta0 = 1e-4, eta1 = 0.25, eta2 = 0.75;

    // Parameters for updating the trust region size delta.
    double sigma1 = 0.25, sigma2 = 0.5, sigma3 = 4;

    int m = c.size();
    int i, cg_iter;
    double delta, snorm;
    double alpha, f, fnew, prered, actred, gs;
    Vector s(m, 0);
    Vector r(m, 0);
    Vector g(m, 0);

    // calculate gradient norm at w=0 for stopping condition.
    c.eval(r, f, g);
    double gnorm0 = norm(g);

    Vector x(x0);
    c.eval(x, f, g);
    double gnorm = norm(g);
    delta = gnorm;
    int funcEval = 1;
    Vector xnew;
    Vector gnew;
    bool reach_boundary;
    bool search = true;
    if (gnorm <= TOL*gnorm0)
      search = false;

    // while ((gnorm >= TOL) && (funcEval < maxEval) )
    while ((funcEval <= maxEval) && search)
      {
	cg_iter = trcg(delta, c, g, x, s, r, &reach_boundary);

	xnew = x + s;

	gs = g*s;
	prered = -0.5*(gs-s*r);
	c.eval(xnew, fnew, gnew);
	// Compute the actual reduction.
	actred = f - fnew;

	// On the first iteration, adjust the initial step bound.
	snorm = norm(s);
	if (funcEval == 1)
	  delta = min(delta, snorm);

	// Compute prediction alpha*snorm of the step.
	if (fnew - f - gs <= 0)
	  alpha = sigma3;
	else
	  alpha = max(sigma1, -0.5*(gs/(fnew - f - gs)));

	// Update the trust region bound according to the ratio of actual to predicted reduction.
	if (actred < eta0*prered)
	  delta = min(max(alpha, sigma1)*snorm, sigma2*delta);
	else if (actred < eta1*prered)
	  delta = max(sigma1*delta, min(alpha*snorm, sigma2*delta));
	else if (actred < eta2*prered)
	  delta = max(sigma1*delta, min(alpha*snorm, sigma3*delta));
	else {
	  if (reach_boundary)
	    delta = sigma3*delta;
	  else
	    delta = max(delta, min(alpha*snorm, sigma3*delta));
	}

	if (verbosity > 0)
	  printf("numIter: %d, act: %e pre: %e delta: %e, ObjVal: %e, OptCond: %e\n", funcEval, actred, prered, delta, f, gnorm);

	if (actred > eta0*prered)
	  {
	    funcEval++;
	    x = xnew;
	    f = fnew;
	    g = gnew;

	    gnorm = norm(g);
	    if (gnorm <= TOL*gnorm0)
	      break;
	  }
	if (f < -1.0e+32)
	  {
	    printf("WARNING: f < -1.0e+32\n");
	    break;
	  }
	// if (fabs(actred) <= 0 && prered <= 0)
	// {
	// 	printf("WARNING: actred and prered <= 0\n");
	// 	break;
	// }
	if (prered <= 0)
	  {
	    printf("WARNING: prered <= 0\n");
	    break;
	  }
	if (fabs(actred) <= 1.0e-12*fabs(f) &&
	    fabs(prered) <= 1.0e-12*fabs(f))
	  {
	    printf("WARNING: actred and prered too small\n");
	    break;
	  }
      }
    return x;
  }
	
}
