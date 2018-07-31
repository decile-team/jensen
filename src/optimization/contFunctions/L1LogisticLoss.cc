// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*

        Jensen: A Convex Optimization And Machine Learning ToolKit
 *	Logistic Loss with L1 regularization
        Author: Rishabh Iyer
 *
 */

#include <iostream>
#include <math.h>
using namespace std;

#include "L1LogisticLoss.h"
#include "../../representation/VectorOperations.h"
#include <assert.h>
#define EPSILON 1e-6
#define MAX 1e2
namespace jensen {
template <class Feature>
L1LogisticLoss<Feature>::L1LogisticLoss(int m, std::vector<Feature>& features, Vector& y, double lambda) :
	ContinuousFunctions(true, m, features.size()), features(features), y(y), lambda(lambda)
{
	if (n > 0)
		assert(features[0].numFeatures == m);
	assert(features.size() == y.size());
}

template <class Feature>
L1LogisticLoss<Feature>::L1LogisticLoss(const L1LogisticLoss& l) :
	ContinuousFunctions(true, l.m, l.n), features(l.features), y(l.y), lambda(l.lambda) {
}

template <class Feature>
L1LogisticLoss<Feature>::~L1LogisticLoss(){
}

template <class Feature>
double L1LogisticLoss<Feature>::eval(const Vector& x) const {
	assert(x.size() == m);
	double sum = lambda*norm(x, 1);
	for (int i = 0; i < n; i++) {
		double preval = y[i]*(x*features[i]);
		if (preval > MAX)
			continue;
		else if (preval < -1*MAX)
			sum += preval;
		else
			sum += log(1 + exp(-preval));
	}
	return sum;
}

template <class Feature>
Vector L1LogisticLoss<Feature>::evalGradient(const Vector& x) const {
	assert(x.size() == m);
	Vector g = Vector(m, 0);
	for (int i = 0; i < n; i++) {
		double preval = y[i]*(x*features[i]);
		if (preval > MAX)
			g -= y[i]*features[i];
		else if (preval < -1*MAX)
			continue;
		else
			g -= (y[i]/(1 + exp(-preval)))*features[i];
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
void L1LogisticLoss<Feature>::eval(const Vector& x, double& f, Vector& g) const {
	assert(x.size() == m);
	g = Vector(m, 0);
	f = lambda*norm(x, 1);
	double val;
	for (int i = 0; i < n; i++) {
		double preval = y[i]*(x*features[i]);
		if (preval > MAX) {
			g -= (y[i]/(1 + exp(preval)))*features[i];
		}
		else if (preval < -1*MAX) {
			g -= (y[i]/(1 + exp(preval)))*features[i];
			f-=preval;
		}
		else{
			f += log(1 + exp(-preval));
			g -= (y[i]/(1 + exp(preval)))*features[i];
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

template <class Feature>
Vector L1LogisticLoss<Feature>::evalStochasticGradient(const Vector& x, std::vector<int>& miniBatch) const {
	assert(x.size() == m);
	Vector g = Vector(m, 0);
	for (vector<int>::iterator it = miniBatch.begin(); it != miniBatch.end(); it++) {
		double preval = y[*it]*(x*features[*it]);
		if (preval > MAX)
			g -= y[*it]*features[*it];
		else if (preval < -1*MAX)
			continue;
		else
			g -= (y[*it]/(1 + exp(-preval)))*features[*it];
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
void L1LogisticLoss<Feature>::evalStochastic(const Vector& x, double& f, Vector& g, std::vector<int>& miniBatch) const {
	assert(x.size() == m);
	g = Vector(m, 0);
	f = lambda*norm(x, 1);
	double val;
	for (vector<int>::iterator it = miniBatch.begin(); it != miniBatch.end(); it++) {
		double preval = y[*it]*(x*features[*it]);
		if (preval > MAX) {
			g -= (y[*it]/(1 + exp(preval)))*features[*it];
		}
		else if (preval < -1*MAX) {
			g -= (y[*it]/(1 + exp(preval)))*features[*it];
			f-=preval;
		}
		else{
			f += log(1 + exp(-preval));
			g -= (y[*it]/(1 + exp(preval)))*features[*it];
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
template class L1LogisticLoss<SparseFeature>;
template class L1LogisticLoss<DenseFeature>;
}
