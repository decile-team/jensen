// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*


	Jensen: A Convex Optimization And Machine Learning ToolKit
 *	Least Squares Loss with L1 regularization
	Author: Rishabh Iyer
 *
*/

#include<iostream>
#include <math.h>
using namespace std;

#include "L1LeastSquaresLoss.h"
#include "../../representation/VectorOperations.h"
#include <assert.h>
#define EPSILON 1e-6
#define MAX 1e2
namespace jensen {	
  template <class Feature>
	L1LeastSquaresLoss<Feature>::L1LeastSquaresLoss(int m, std::vector<Feature>& features, Vector& y, double lambda): 
	ContinuousFunctions(true, m, features.size()), features(features), y(y), lambda(lambda)
	{
		if (n > 0)
			assert(features[0].numFeatures == m);
		assert(features.size() == y.size());
	}
	
  template <class Feature>
	L1LeastSquaresLoss<Feature>::L1LeastSquaresLoss(const L1LeastSquaresLoss& l) : 
	ContinuousFunctions(true, l.m, l.n), features(l.features), y(l.y), lambda(l.lambda) {}

  template <class Feature>
    L1LeastSquaresLoss<Feature>::~L1LeastSquaresLoss(){}
	
  template <class Feature>
	double L1LeastSquaresLoss<Feature>::eval(const Vector& x) const{
		assert(x.size() == m);
		double sum = lambda * norm(x, 1);
		for (int i = 0; i < n; i++){
			sum += (y[i] - (x*features[i]))*(y[i] - (x*features[i]));
		}
		return sum;
	}
	
  template <class Feature>
	Vector L1LeastSquaresLoss<Feature>::evalGradient(const Vector& x) const{
		assert(x.size() == m);
		Vector g = Vector(m, 0);
		for (int i = 0; i < n; i++){
			g -= 2*(y[i] - (x*features[i]))*features[i];
		}
		for (int i = 0; i < m; i++)
		{
			if (x[i] != 0)
			{
				g[i] += lambda*sign(x[i]);
			}
			else
			{
				if (g[i] > lambda)
					g[i] -= lambda;
				else if (g[i] < -lambda)
					g[i] += lambda;
			}
		}
		return g;
	}

  template <class Feature>
	void L1LeastSquaresLoss<Feature>::eval(const Vector& x, double& f, Vector& g) const{
		assert(x.size() == m);
		g = Vector(m, 0);
		f = lambda*norm(x, 1);
		double val;
		for (int i = 0; i < n; i++){
			double val = y[i] - (x*features[i]);
			f += val*val;
			g -= 2*val*features[i];
		}
		for (int i = 0; i < m; i++)
		{
			if (x[i] != 0)
			{
				g[i] += lambda*sign(x[i]);
			}
			else
			{
				if (g[i] > lambda)
					g[i] -= lambda;
				else if (g[i] < -lambda)
					g[i] += lambda;
			}
		}
		return;
	}
	
  template <class Feature>
	Vector L1LeastSquaresLoss<Feature>::evalStochasticGradient(const Vector& x, std::vector<int>& miniBatch) const{
		assert(x.size() == m);
		Vector g = Vector(m, 0);
		double val;
		for (vector<int>::iterator it = miniBatch.begin(); it != miniBatch.end(); it++){
			double val = y[*it] - (x*features[*it]);
			g -= 2*val*features[*it];
		}
		for (int i = 0; i < m; i++)
		{
			if (x[i] != 0)
			{
				g[i] += lambda*sign(x[i]);
			}
			else
			{
				if (g[i] > lambda)
					g[i] -= lambda;
				else if (g[i] < -lambda)
					g[i] += lambda;
			}
		}
		return g;
	}

  template <class Feature>
	void L1LeastSquaresLoss<Feature>::evalStochastic(const Vector& x, double& f, Vector& g, std::vector<int>& miniBatch) const{
		assert(x.size() == m);
		g = Vector(m, 0);
		f = lambda*norm(x, 1);
		for (vector<int>::iterator it = miniBatch.begin(); it != miniBatch.end(); it++){
			for (vector<int>::iterator it = miniBatch.begin(); it != miniBatch.end(); it++){
				double val = y[*it] - (x*features[*it]);
				f += val*val;
				g -= 2*val*features[*it];
			}
		}
		for (int i = 0; i < m; i++)
		{
			if (x[i] != 0)
			{
				g[i] += lambda*sign(x[i]);
			}
			else
			{
				if (g[i] > lambda)
					g[i] -= lambda;
				else if (g[i] < -lambda)
					g[i] += lambda;
			}
		}
		return;
	}
  template class L1LeastSquaresLoss<SparseFeature>;
  template class L1LeastSquaresLoss<DenseFeature>;
		
}
