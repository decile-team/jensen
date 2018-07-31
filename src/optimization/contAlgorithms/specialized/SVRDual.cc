// Copyright (c) 2007-2015 The LIBLINEAR Project.
// Modified for use in Jensen by Rishabh Iyer
/*
 *	Gradient Descent for Unconstrained Convex Minimization with constant step size
        Solves the problem \min_x \phi(x), where \phi is a convex (or continuous) function.
        Anthor: Rishabh Iyer
 *
        Input:  Continuous Function: c
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
#include <cmath>
using namespace std;

#include "SVRDual.h"
#include "../../contFunctions/ContinuousFunctions.h"
#include "../../contFunctions/L2SmoothSVRLoss.h"
#include "../../contFunctions/L2HingeSVRLoss.h"
#define L2R_L1LOSS_SVR_DUAL 1
#define L2R_L2LOSS_SVR_DUAL 2
#define INF HUGE_VAL

namespace jensen {

Vector SVRDual(vector<SparseFeature>& features, Vector& y, int solver_type, double lambda, double p, double eps, int max_iter, const int verbosity)

{
	int l = features.size();
	double C = 1/lambda;
	int w_size = features[0].numFeatures;
	int i, s, iter = 0;
	int active_size = l;
	vector<int> index(l);
	Vector w = Vector(w_size, 0);
	double d, G, H;
	double pval;
	double Gmax_old = INF;
	double Gmax_new, Gnorm1_new;
	double Gnorm1_init = -1.0;         // Gnorm1_init is initialized at the first iteration
	Vector beta(l);
	Vector QD(l);

	// L2R_L2LOSS_SVR_DUAL
	double upper_bound = INF;
	ContinuousFunctions *c;
	if(solver_type == L2R_L1LOSS_SVR_DUAL)
	{
		lambda = 0;
		upper_bound = C;
		c = new L2HingeSVRLoss<SparseFeature>(w_size, features, y, lambda, p);

	}
	else if (solver_type == L2R_L2LOSS_SVR_DUAL) {
		lambda = 0.5*lambda;
		upper_bound = INF;
		c = new L2SmoothSVRLoss<SparseFeature>(w_size, features, y, lambda, p);
	}
	// Initial beta can be set here. Note that
	// -upper_bound <= beta[i] <= upper_bound
	for(i=0; i<l; i++)
		beta[i] = 0;

	for(i=0; i<w_size; i++)
		w[i] = 0;
	for(i=0; i<l; i++)
	{
		QD[i] = 0;
		for (int j = 0; j < features[i].featureIndex.size(); j++) {
			double val = features[i].featureVec[j];
			int ind = features[i].featureIndex[j];
			QD[i] += val*val;
			w[ind] += beta[i]*val;
		}

		index[i] = i;
	}


	while(iter < max_iter)
	{
		Gmax_new = 0;
		Gnorm1_new = 0;

		for(i=0; i<active_size; i++)
		{
			int j = i+rand()%(active_size-i);
			swap(index[i], index[j]);
		}

		for(s=0; s<active_size; s++)
		{
			i = index[s];
			G = -y[i] + lambda*beta[i];
			H = QD[i] + lambda;

			for (int j = 0; j < features[i].featureIndex.size(); j++) {
				double val = features[i].featureVec[j];
				int ind = features[i].featureIndex[j];
				G += val*w[ind];
			}

			double Gp = G+p;
			double Gn = G-p;
			double violation = 0;
			if(beta[i] == 0)
			{
				if(Gp < 0)
					violation = -Gp;
				else if(Gn > 0)
					violation = Gn;
				else if(Gp>Gmax_old && Gn<-Gmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(beta[i] >= upper_bound)
			{
				if(Gp > 0)
					violation = Gp;
				else if(Gp < -Gmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(beta[i] <= -upper_bound)
			{
				if(Gn < 0)
					violation = -Gn;
				else if(Gn > Gmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(beta[i] > 0)
				violation = fabs(Gp);
			else
				violation = fabs(Gn);

			Gmax_new = max(Gmax_new, violation);
			Gnorm1_new += violation;

			// obtain Newton direction d
			if(Gp < H*beta[i])
				d = -Gp/H;
			else if(Gn > H*beta[i])
				d = -Gn/H;
			else
				d = -beta[i];

			if(fabs(d) < 1.0e-12)
				continue;

			double beta_old = beta[i];
			beta[i] = min(max(beta[i]+d, -upper_bound), upper_bound);
			d = beta[i]-beta_old;

			if(d != 0)
			{
				for (int j = 0; j < features[i].featureIndex.size(); j++) {
					double val = features[i].featureVec[j];
					int ind = features[i].featureIndex[j];
					w[ind] += d*val;
				}
			}
		}

		if(iter == 0)
			Gnorm1_init = Gnorm1_new;
		iter++;
		if(iter % 10 == 0)
			// printf(".");

			if(Gnorm1_new <= eps*Gnorm1_init)
			{
				if(active_size == l)
					break;
				else
				{
					active_size = l;
					// printf("*");
					Gmax_old = INF;
					continue;
				}
			}

		Gmax_old = Gmax_new;
		double v = 0;
		int nSV = 0;
		for(i=0; i<w_size; i++)
			v += w[i]*w[i];
		v = 0.5*v;
		for(i=0; i<l; i++)
		{
			v += p*fabs(beta[i]) - y[i]*beta[i] + 0.5*lambda*beta[i]*beta[i];
			if(beta[i] != 0)
				nSV++;
		}
		pval = c->eval(w);
		// printf("Dual Objective value = %lf, Primal Objective Value = %lf, nSV = %d\n",v/2, pval, nSV);
		if(verbosity > 0)
			printf("numIter: %d, ObjVal: %e, Dual ObjVal: %e, nSV: %e\n", iter, pval, v/2, nSV);

	}

	// printf("\noptimization finished, #iter = %d\n", iter);
	if(iter >= max_iter)
		printf("\nWARNING: reaching max number of iterations\nUsing -s 11 may be faster\n\n");

	// calculate objective value
	double v = 0;
	int nSV = 0;
	for(i=0; i<w_size; i++)
		v += w[i]*w[i];
	v = 0.5*v;
	for(i=0; i<l; i++)
	{
		v += p*fabs(beta[i]) - y[i]*beta[i] + 0.5*lambda*beta[i]*beta[i];
		if(beta[i] != 0)
			nSV++;
	}

	// printf("Objective value = %lf\n", v);
	// printf("nSV = %d\n",nSV);
	delete c;
	return w;
}
}
