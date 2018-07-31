// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*
        Jensen: A Convex Optimization And Machine Learning ToolKit
 *	Probit Loss with L2 regularization
        Author: Rishabh Iyer
 *
 */

#include <iostream>
#include <math.h>
using namespace std;

#include "L1ProbitLoss.h"
#include "../../representation/VectorOperations.h"
#include <assert.h>
#define EPSILON 1e-16
#define MAX 1e2
#define _USE_MATH_DEFINES

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

namespace jensen {
template <class Feature>
L1ProbitLoss<Feature>::L1ProbitLoss(int m, std::vector<Feature>& features, Vector& y, double lambda) :
	ContinuousFunctions(true, m, features.size()), features(features), y(y), lambda(lambda)
{
	if (n > 0)
		assert(features[0].numFeatures == m);
	assert(features.size() == y.size());
}

template <class Feature>
L1ProbitLoss<Feature>::L1ProbitLoss(const L1ProbitLoss& l) :
	ContinuousFunctions(true, l.m, l.n), features(l.features), y(l.y), lambda(l.lambda) {
}

template <class Feature>
L1ProbitLoss<Feature>::~L1ProbitLoss(){
}

template <class Feature>
double L1ProbitLoss<Feature>::eval(const Vector& x) const {
	assert(x.size() == m);
	double sum = lambda*norm(x, 1);
	for (int i = 0; i < n; i++) {
		double val = y[i]*(x*features[i])/sqrt(2);
		double probitval = (1/2)*(1 + erf(val))+EPSILON;
		sum-= log(probitval);
	}
	return sum;
}

template <class Feature>
Vector L1ProbitLoss<Feature>::evalGradient(const Vector& x) const {
	assert(x.size() == m);
	Vector g = Vector(m, 0);
	for (int i = 0; i < n; i++) {
		double val = y[i]*(x*features[i])/sqrt(2);
		double normval = (1/sqrt(2*M_PI))*exp(-(val*val));
		double probitval = (1/2)*(1 + erf(val))+EPSILON;
		g -= features[i]*(y[i]*normval/probitval);
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
void L1ProbitLoss<Feature>::eval(const Vector& x, double& f, Vector& g) const {
	assert(x.size() == m);
	g = Vector(m, 0);
	f = lambda*norm(x, 1);
	for (int i = 0; i < n; i++) {
		double val = y[i]*(x*features[i])/sqrt(2);
		double normval = (1/sqrt(2*M_PI))*exp(-(val*val));
		double probitval = 0.5*(1 + erf(val))+EPSILON;
		g -= features[i]*(y[i]*normval/probitval);
		f -= log(probitval);
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
Vector L1ProbitLoss<Feature>::evalStochasticGradient(const Vector& x, std::vector<int>& miniBatch) const {
	assert(x.size() == m);
	Vector g = Vector(m, 0);
	double val;
	for (vector<int>::iterator it = miniBatch.begin(); it != miniBatch.end(); it++) {
		double val = y[*it]*(x*features[*it])/sqrt(2);
		double normval = (1/sqrt(2*M_PI))*exp(-(val*val));
		double probitval = (1/2)*(1 + erf(val))+EPSILON;
		g -= features[*it]*(y[*it]*normval/probitval);
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
void L1ProbitLoss<Feature>::evalStochastic(const Vector& x, double& f, Vector& g, std::vector<int>& miniBatch) const {
	assert(x.size() == m);
	g = Vector(m, 0);
	f = lambda*norm(x, 1);
	for (vector<int>::iterator it = miniBatch.begin(); it != miniBatch.end(); it++) {
		double val = y[*it]*(x*features[*it])/sqrt(2);
		double normval = (1/sqrt(2*M_PI))*exp(-(val*val));
		double probitval = (1/2)*(1 + erf(val))+EPSILON;
		g -= features[*it]*(y[*it]*normval/probitval);
		f -= log(probitval);
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

template class L1ProbitLoss<SparseFeature>;
template class L1ProbitLoss<DenseFeature>;

}
