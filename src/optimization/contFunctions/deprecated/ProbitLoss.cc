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

#include "ProbitLoss.h"
#include "../datarep/VectorOperations.h"
#include <assert.h>
#define EPSILON 1e-16
#define MAX 1e2
#define _USE_MATH_DEFINES

namespace jensen {
template <class Feature>
ProbitLoss<Feature>::ProbitLoss(int m, std::vector<Feature>& features, Vector& y) :
	ContinuousFunctions(true, m, features.size()), features(features), y(y)
{
	if (n > 0)
		assert(features[0].numFeatures == m);
	assert(features.size() == y.size());
}

template <class Feature>
ProbitLoss<Feature>::ProbitLoss(const ProbitLoss& l) :
	ContinuousFunctions(true, l.m, l.n), features(l.features), y(l.y) {
}

template <class Feature>
ProbitLoss<Feature>::~ProbitLoss(){
}

template <class Feature>
double ProbitLoss<Feature>::eval(const Vector& x) const {
	assert(x.size() == m);
	double sum = 0;
	double val;
	for (int i = 0; i < n; i++) {
		double val = y[i]*(x*features[i])/sqrt(2);
		double probitval = (1/2)*(1 + erf(val))+EPSILON;
		sum-= log(probitval);
	}
	return sum;
}

template <class Feature>
Vector ProbitLoss<Feature>::evalGradient(const Vector& x) const {
	assert(x.size() == m);
	Vector g = Vector(m, 0);
	double val;
	for (int i = 0; i < n; i++) {
		double val = y[i]*(x*features[i])/sqrt(2);
		double normval = (1/sqrt(2*M_PI))*exp(-(val*val));
		double probitval = (1/2)*(1 + erf(val))+EPSILON;
		g -= features[i]*(y[i]*normval/probitval);
	}
	return g;
}

template <class Feature>
void ProbitLoss<Feature>::eval(const Vector& x, double& f, Vector& g) const {
	assert(x.size() == m);
	g = Vector(m, 0);
	f = 0;
	double val;
	for (int i = 0; i < n; i++) {
		double val = y[i]*(x*features[i])/sqrt(2);
		double normval = (1/sqrt(2*M_PI))*exp(-(val*val));
		double probitval = 0.5*(1 + erf(val))+EPSILON;
		g -= features[i]*(y[i]*normval/probitval);
		f -= log(probitval);
	}
	return;
}

template <class Feature>
Vector ProbitLoss<Feature>::evalStochasticGradient(const Vector& x, std::vector<int>& miniBatch) const {
	assert(x.size() == m);
	Vector g = Vector(m, 0);
	double val;
	for (vector<int>::iterator it = miniBatch.begin(); it != miniBatch.end(); it++) {
		double val = y[*it]*(x*features[*it])/sqrt(2);
		double normval = (1/sqrt(2*M_PI))*exp(-(val*val));
		double probitval = (1/2)*(1 + erf(val))+EPSILON;
		g -= features[*it]*(y[*it]*normval/probitval);
	}
	return g;
}

template <class Feature>
void ProbitLoss<Feature>::evalStochastic(const Vector& x, double& f, Vector& g, std::vector<int>& miniBatch) const {
	assert(x.size() == m);
	g = Vector(m, 0);
	f = 0;
	for (vector<int>::iterator it = miniBatch.begin(); it != miniBatch.end(); it++) {
		double val = y[*it]*(x*features[*it])/sqrt(2);
		double normval = (1/sqrt(2*M_PI))*exp(-(val*val));
		double probitval = (1/2)*(1 + erf(val))+EPSILON;
		g -= features[*it]*(y[*it]*normval/probitval);
		f -= log(probitval);
	}
	return;
}

template class ProbitLoss<SparseFeature>;
template class ProbitLoss<DenseFeature>;


}
