// Copyright (c) 2007-2015 The LIBLINEAR Project.
// Modified for use in Jensen by Rishabh Iyer
// A coordinate descent algorithm for 
// L1-regularized logistic regression problems
//
//  min_w \sum |wj| + C \sum log(1+exp(-yi w^T xi)),
//
// Given: 
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Yuan et al. (2011) and appendix of LIBLINEAR paper, Fan et al. (2008)

#include <stdio.h>
#include <algorithm>
#include <vector>
#include <iostream>
#include <cmath>
using namespace std;

#include "../../../representation/VectorOperations.h"
#include "L1LRPrimal.h"

namespace jensen {
	
#define INF HUGE_VAL
template <class T> static inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif

// transpose matrix X from row format to column format
static void transpose(const vector<SparseFeature>& features, vector<SparseFeature>& invfeatures)
{
	int n = features.size(); // number of data items
	int m = features[0].numFeatures; // number of features
	invfeatures = vector<SparseFeature>(m);
	for(int i=0; i < n; i++)
	{
		for (int j = 0; j < features[i].featureIndex.size(); j++){
			invfeatures[features[i].featureIndex[j]].featureIndex.push_back(i);
			invfeatures[features[i].featureIndex[j]].featureVec.push_back(features[i].featureVec[j]);
		}
	}
}

  void L1LRPrimal(vector<SparseFeature>& features, Vector& y, Vector& x, double C, double eps, const int max_newton_iter,  const int verbosity){
	vector<SparseFeature> invfeatures;
	transpose(features, invfeatures);
	int n = features.size();
	int m = invfeatures.size();
	x = Vector(m, 0);
	int s, newton_iter=0, iter=0;
	// int max_newton_iter = 100;
	int max_iter = 1000;
	int max_num_linesearch = 20;
	int active_size;
	int QP_active_size;

	double nu = 1e-12;
	double inner_eps = 1;
	double sigma = 0.01;
	double x_norm, x_norm_new;
	double z, G, H;
	double Gnorm1_init = -1.0; // Gnorm1_init is initialized at the first iteration
	double Gmax_old = 1e30;
	double Gmax_new, Gnorm1_new;
	double QP_Gmax_old = 1e30;
	double QP_Gmax_new, QP_Gnorm1_new;
	double delta, negsum_xTd, cond;

	vector<int> index(m);
	Vector Hdiag(m);
	Vector Grad(m);
	Vector xpd(m);
	Vector xjneg_sum(m);
	Vector xTd(n);
	Vector exp_wTx(n, 0);
	Vector exp_wTx_new(n);
	Vector tau(n);
	Vector D(n);
	// feature_node *x;
	for(int j=0; j<n; j++)
	{
		exp_wTx[j] = 0;
	}

	x_norm = 0;
	for(int j=0; j < m; j++)
	{
		x_norm += fabs(x[j]);
		xpd[j] = x[j];
		index[j] = j;
		xjneg_sum[j] = 0;
		for (int i = 0; i < invfeatures[j].featureIndex.size(); i++)
		{
			int ind = invfeatures[j].featureIndex[i];
			double fval = invfeatures[j].featureVec[i];
			exp_wTx[ind] += x[j]*fval;
			if(y[ind] == -1)
				xjneg_sum[j] += C*fval;
		}
	}
	for(int j=0; j < n; j++)
	{
		exp_wTx[j] = exp(exp_wTx[j]);
		double tau_tmp = 1/(1+exp_wTx[j]);
		tau[j] = C*tau_tmp;
		D[j] = C*exp_wTx[j]*tau_tmp*tau_tmp;
	}

	while(newton_iter < max_newton_iter)
	{
		Gmax_new = 0;
		Gnorm1_new = 0;
		active_size = m;

		for(s=0; s<active_size; s++)
		{
			int j = index[s];
			Hdiag[j] = nu;
			Grad[j] = 0;

			double tmp = 0;
			for (int i = 0; i < invfeatures[j].featureIndex.size(); i++)
			{
				int ind = invfeatures[j].featureIndex[i];
				double fval = invfeatures[j].featureVec[i];
				Hdiag[j] += fval*fval*D[ind];
				tmp += fval*tau[ind];
			}
			Grad[j] = -tmp + xjneg_sum[j];

			double Gp = Grad[j]+1;
			double Gn = Grad[j]-1;
			double violation = 0;
			if(x[j] == 0)
			{
				if(Gp < 0)
					violation = -Gp;
				else if(Gn > 0)
					violation = Gn;
				//outer-level shrinking
				else if(Gp>Gmax_old/n && Gn<-Gmax_old/n)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(x[j] > 0)
				violation = fabs(Gp);
			else
				violation = fabs(Gn);

			Gmax_new = max(Gmax_new, violation);
			Gnorm1_new += violation;
		}

		if(newton_iter == 0)
			Gnorm1_init = Gnorm1_new;

		if(Gnorm1_new <= eps*Gnorm1_init)
			break;

		iter = 0;
		QP_Gmax_old = 1e30;
		QP_active_size = active_size;

		for(int i=0; i<n; i++)
			xTd[i] = 0;

		// optimize QP over xpd
		while(iter < max_iter)
		{
			QP_Gmax_new = 0;
			QP_Gnorm1_new = 0;

			for(int j=0; j<QP_active_size; j++)
			{
				int i = j+rand()%(QP_active_size-j);
				swap(index[i], index[j]);
			}

			for(s=0; s<QP_active_size; s++)
			{
				int j = index[s];
				H = Hdiag[j];

				G = Grad[j] + (xpd[j]-x[j])*nu;
				for (int i = 0; i < invfeatures[j].featureIndex.size(); i++)
				{
					int ind = invfeatures[j].featureIndex[i];
					double fval = invfeatures[j].featureVec[i];
					G += fval*D[ind]*xTd[ind];
				}

				double Gp = G+1;
				double Gn = G-1;
				double violation = 0;
				if(xpd[j] == 0)
				{
					if(Gp < 0)
						violation = -Gp;
					else if(Gn > 0)
						violation = Gn;
					//inner-level shrinking
					else if(Gp>QP_Gmax_old/n && Gn<-QP_Gmax_old/n)
					{
						QP_active_size--;
						swap(index[s], index[QP_active_size]);
						s--;
						continue;
					}
				}
				else if(xpd[j] > 0)
					violation = fabs(Gp);
				else
					violation = fabs(Gn);

				QP_Gmax_new = max(QP_Gmax_new, violation);
				QP_Gnorm1_new += violation;

				// obtain solution of one-variable problem
				if(Gp < H*xpd[j])
					z = -Gp/H;
				else if(Gn > H*xpd[j])
					z = -Gn/H;
				else
					z = -xpd[j];

				if(fabs(z) < 1.0e-12)
					continue;
				z = min(max(z,-10.0),10.0);

				xpd[j] += z;

				for (int i = 0; i < invfeatures[j].featureIndex.size(); i++)
				{
					int ind = invfeatures[j].featureIndex[i];
					double fval = invfeatures[j].featureVec[i];
					xTd[ind] += fval*z;
				}
			}

			iter++;

			if(QP_Gnorm1_new <= inner_eps*Gnorm1_init)
			{
				//inner stopping
				if(QP_active_size == active_size)
					break;
				//active set reactivation
				else
				{
					QP_active_size = active_size;
					QP_Gmax_old = 1e30;
					continue;
				}
			}

			QP_Gmax_old = QP_Gmax_new;
		}

		if(iter >= max_iter)
			printf("WARNING: reaching max number of inner iterations\n");

		delta = 0;
		x_norm_new = 0;
		for(int j=0; j < m; j++)
		{
			delta += Grad[j]*(xpd[j]-x[j]);
			if(xpd[j] != 0)
				x_norm_new += fabs(xpd[j]);
		}
		delta += (x_norm_new-x_norm);

		negsum_xTd = 0;
		for(int i = 0; i < n; i++)
			if(y[i] == -1)
				negsum_xTd += C*xTd[i];

		int num_linesearch;
		for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)
		{
			cond = x_norm_new - x_norm + negsum_xTd - sigma*delta;

			for(int i=0; i<n; i++)
			{
				double exp_xTd = exp(xTd[i]);
				exp_wTx_new[i] = exp_wTx[i]*exp_xTd;
				cond += C*log((1+exp_wTx_new[i])/(exp_xTd+exp_wTx_new[i]));
			}
			if(cond <= 0)
			{
				x_norm = x_norm_new;
				for(int j=0; j<m; j++)
					x[j] = xpd[j];
				for(int i=0; i<n; i++)
				{
					exp_wTx[i] = exp_wTx_new[i];
					double tau_tmp = 1/(1+exp_wTx[i]);
					tau[i] = C*tau_tmp;
					D[i] = C*exp_wTx[i]*tau_tmp*tau_tmp;
				}
				break;
			}
			else
			{
				x_norm_new = 0;
				for(int j=0; j<m; j++)
				{
					xpd[j] = (x[j]+xpd[j])*0.5;
					if(xpd[j] != 0)
						x_norm_new += fabs(xpd[j]);
				}
				delta *= 0.5;
				negsum_xTd *= 0.5;
				for(int i=0; i<n ; i++)
					xTd[i] *= 0.5;
			}
		}

		// Recompute some info due to too many line search steps
		if(num_linesearch >= max_num_linesearch)
		{
			for(int i=0; i < n; i++)
				exp_wTx[i] = 0;

			for(int j=0; j<m; j++)
			{
				if(x[j]==0) continue;
				for (int i = 0; i < invfeatures[j].featureIndex.size(); i++)
				{
					int ind = invfeatures[j].featureIndex[i];
					double fval = invfeatures[j].featureVec[i];
					exp_wTx[ind] += x[j]*fval;
				}
			}

			for(int i=0; i<n; i++)
				exp_wTx[i] = exp(exp_wTx[i]);
		}

		if(iter == 1)
			inner_eps *= 0.25;

		newton_iter++;
		Gmax_old = Gmax_new;

		// printf("iter %3d  #CD cycles %d\n", newton_iter, iter);
		double v = 0;
		int nnz = 0;
		for(int j=0; j<m; j++)
			if(x[j] != 0)
			{
				v += fabs(x[j]);
				nnz++;
			}
		for(int j=0; j<=n; j++)
			if(y[j] == 1)
				v += C*log(1+1/exp_wTx[j]);
			else
				v += C*log(1+exp_wTx[j]);

		// printf("Objective value = %lf\n", v/C);
		// printf("#nonzeros/#features = %d/%d\n", nnz, n);
		if (verbosity > 0)
		  printf("numIter: %d, #CD cycles: %d, ObjVal: %e, #nonzeros/#features: %d/%d\n", newton_iter, iter, v/C, nnz,n);
	}

	double v = 0;
	int nnz = 0;
	for(int j=0; j<m; j++)
		if(x[j] != 0)
		{
			v += fabs(x[j]);
			nnz++;
		}
	for(int j=0; j<=n; j++)
		if(y[j] == 1)
			v += C*log(1+1/exp_wTx[j]);
		else
			v += C*log(1+exp_wTx[j]);

	if (verbosity > 0)
	  printf("numIter: %d, #CD cycles: %d, ObjVal: %e, #nonzeros/#features: %d/%d\n", newton_iter, iter, v/C, nnz,n);
	// printf("Objective value = %lf\n", v/C);
	// printf("#nonzeros/#features = %d/%d\n", nnz, n);
	printf("=========================\n");
	printf("optimization finished, #iter = %d\n", newton_iter);
	if(newton_iter >= max_newton_iter)
		printf("WARNING: reaching max number of iterations\n");

	// calculate objective value
	
}
		
}
