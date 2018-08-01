// Copyright (c) 2007-2015 The LIBLINEAR Project.
// Modified for use in Jensen by Rishabh Iyer
/*
 *	Co-ordinate Descent Algorithm for L2 Regularized Support vector machine Classification
        Author: Rishabh Iyer
 *
        Input:  vector<SparseFeatures> features: Features for training
                        Vector y: Labels (binary)
                        Vector w: Output weights
                        double C: 1/lambda (regularization)
                        double eps: tolerance

        Output: Output on convergence (x)
 */

#include <stdio.h>
#include <cmath>
#include <algorithm>
#include <iostream>
using namespace std;

#include "SVCDual.h"
#include "../../contFunctions/ContinuousFunctions.h"
#include "../../contFunctions/L2SmoothSVMLoss.h"
#include "../../contFunctions/L2HingeSVMLoss.h"
#define L2R_L1LOSS_SVC_DUAL 1
#define L2R_L2LOSS_SVC_DUAL 2
#define INF HUGE_VAL

//template <class T> static inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
//#ifndef min
//template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
//#endif
//#ifndef max
//template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
//#endif

namespace jensen {
Vector SVCDual(vector<SparseFeature>& features, Vector& y, int solver_type, double lambda, double eps, int max_iter, const int verbosity)
{
	int l = features.size(); // number of training examples
	int w_size = features[0].numFeatures; // dimension of the features
	Vector w = Vector(w_size, 0);
	int i, s, iter = 0;
	double C, d, G;
	Vector QD(l);
	vector<int> index(l);
	Vector alpha(l);
	int active_size = l;
	C = 1/lambda;
	// PG: projected gradient, for shrinking and stopping
	double PG;
	double PGmax_old = INF;
	double PGmin_old = -INF;
	double PGmax_new, PGmin_new;
	double pval;

	// default solver_type: L2R_L2LOSS_SVC_DUAL
	double diag;
	double upper_bound;
	ContinuousFunctions *c;
	if(solver_type == L2R_L1LOSS_SVC_DUAL)
	{
		diag = 0;
		upper_bound = C;
		c = new L2HingeSVMLoss<SparseFeature>(w_size, features, y, lambda);
	}
	else if (solver_type == L2R_L2LOSS_SVC_DUAL) {
		diag = 0.5*lambda;
		upper_bound = INF;
		c = new L2SmoothSVMLoss<SparseFeature>(w_size, features, y, lambda);
	}
	// Initial alpha can be set here. Note that
	// 0 <= alpha[i] <= upper_bound
	for(i=0; i<l; i++)
		alpha[i] = 0;

	for(i=0; i<w_size; i++)
		w[i] = 0;
	for(i=0; i<l; i++)
	{
		QD[i] = 0.5/C;

		for (int j = 0; j < features[i].featureIndex.size(); j++) {
			double val = features[i].featureVec[j];
			int ind = features[i].featureIndex[j];
			QD[i] += val*val;
			w[ind] += y[i]*alpha[i]*val;
		}
		index[i] = i;
	}

	while (iter < max_iter)
	{
		PGmax_new = -INF;
		PGmin_new = INF;

		for (i=0; i<active_size; i++)
		{
			int j = i+rand()%(active_size-i);
			swap(index[i], index[j]);
		}

		for (s=0; s<active_size; s++)
		{
			i = index[s];
			G = 0;

			for (int j = 0; j < features[i].featureIndex.size(); j++) {
				double val = features[i].featureVec[j];
				int ind = features[i].featureIndex[j];
				G += w[ind]*val;
			}
			G = G*y[i]-1;

			C = upper_bound;
			G += alpha[i]*diag;

			PG = 0;
			if (alpha[i] == 0)
			{
				if (G > PGmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
				else if (G < 0)
					PG = G;
			}
			else if (alpha[i] == C)
			{
				if (G < PGmin_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
				else if (G > 0)
					PG = G;
			}
			else
				PG = G;

			PGmax_new = max(PGmax_new, PG);
			PGmin_new = min(PGmin_new, PG);

			if(fabs(PG) > 1.0e-12)
			{
				double alpha_old = alpha[i];
				alpha[i] = min(max(alpha[i] - G/QD[i], 0.0), C);
				d = (alpha[i] - alpha_old)*y[i];
				for (int j = 0; j < features[i].featureIndex.size(); j++) {
					double val = features[i].featureVec[j];
					int ind = features[i].featureIndex[j];
					w[ind] += d*val;
				}
			}
		}

		iter++;
		if(iter % 10 == 0)
			// printf(".");

			if(PGmax_new - PGmin_new <= eps)
			{
				if(active_size == l)
					break;
				else
				{
					active_size = l;
					// printf("*");
					PGmax_old = INF;
					PGmin_old = -INF;
					continue;
				}
			}
		PGmax_old = PGmax_new;
		PGmin_old = PGmin_new;
		if (PGmax_old <= 0)
			PGmax_old = INF;
		if (PGmin_old >= 0)
			PGmin_old = -INF;

		// calculate objective value

		double v = 0;
		int nSV = 0;
		for(i=0; i<w_size; i++)
			v += w[i]*w[i];
		for(i=0; i<l; i++)
		{
			v += alpha[i]*(alpha[i]*diag - 2);
			if(alpha[i] > 0)
				++nSV;
		}
		pval = c->eval(w);
		// printf("Dual Objective value = %lf, Primal Objective Value = %lf, nSV = %d\n",v/2, pval, nSV);
		if(verbosity > 0)
			printf("numIter: %d, ObjVal: %e, Dual ObjVal: %e, nSV: %e\n", iter, pval, v/2, nSV);
	}

	// printf("\noptimization finished, #iter = %d\n",iter);
	if (iter >= max_iter)
		printf("\nWARNING: reaching max number of iterations\nUsing -s 2 may be faster (also see FAQ)\n\n");

	// calculate objective value

	double v = 0;
	int nSV = 0;
	for(i=0; i<w_size; i++)
		v += w[i]*w[i];
	for(i=0; i<l; i++)
	{
		v += alpha[i]*(alpha[i]*diag - 2);
		if(alpha[i] > 0)
			++nSV;
	}
	// printf("Objective value = %lf\n",v/2);
	// printf("nSV = %d\n",nSV);
	// printf("\noptimization finished, #iter = %d, Dual ObjVal = %e, nSV = %d\n",iter, v/2, nSV);
	delete c;
	return w;
}
}
