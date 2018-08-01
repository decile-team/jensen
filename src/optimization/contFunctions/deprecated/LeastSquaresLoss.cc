// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*
   Jensen: A Convex Optimization And Machine Learning ToolKit
 *	Least Squares Loss with L2 regularization
   Author: Rishabh Iyer
 *
 */

#include <iostream>
#include <math.h>
using namespace std;

#include "LeastSquaresLoss.h"
#include "../datarep/VectorOperations.h"
#include <assert.h>
#define EPSILON 1e-6
#define MAX 1e2
namespace jensen {
template <class Feature>
LeastSquaresLoss<Feature>::LeastSquaresLoss(int m, std::vector<Feature>& features, Vector& y) :
	ContinuousFunctions(true, m, features.size()), features(features), y(y)
{
	if (n > 0)
		assert(features[0].numFeatures == m);
	assert(features.size() == y.size());
}

template <class Feature>
LeastSquaresLoss<Feature>::LeastSquaresLoss(const LeastSquaresLoss& l) :
	ContinuousFunctions(true, l.m, l.n), features(l.features), y(l.y) {
}

template <class Feature>
LeastSquaresLoss<Feature>::~LeastSquaresLoss(){
}

template <class Feature>
double LeastSquaresLoss<Feature>::eval(const Vector& x) const {
	assert(x.size() == m);
	double sum = 0;
	double val;
	for (int i = 0; i < n; i++) {
		double xf = x*features[i];
		sum += (y[i] - xf)*(y[i] - xf);
	}
	return sum;
}

template <class Feature>
Vector LeastSquaresLoss<Feature>::evalGradient(const Vector& x) const {
	assert(x.size() == m);
	Vector g = Vector(m, 0);
	double val;
	for (int i = 0; i < n; i++) {
		g -= 2*(y[i] - (x*features[i]))*features[i];
	}
	return g;
}

template <class Feature>
void LeastSquaresLoss<Feature>::eval(const Vector& x, double& f, Vector& g) const {
	assert(x.size() == m);
	g = Vector(m, 0);
	f = 0;
	double val;
	for (int i = 0; i < n; i++) {
		double val = y[i] - (x*features[i]);
		f += val*val;
		g -= 2*val*features[i];
	}
	return;
}

template <class Feature>
Vector LeastSquaresLoss<Feature>::evalStochasticGradient(const Vector& x, std::vector<int>& miniBatch) const {
	assert(x.size() == m);
	Vector g = Vector(m, 0);
	double val;
	for (vector<int>::iterator it = miniBatch.begin(); it != miniBatch.end(); it++) {
		double val = y[*it] - (x*features[*it]);
		g -= 2*val*features[*it];
	}
	return g;
}

template <class Feature>
void LeastSquaresLoss<Feature>::evalStochastic(const Vector& x, double& f, Vector& g, std::vector<int>& miniBatch) const {
	assert(x.size() == m);
	g = Vector(m, 0);
	f = 0;
	for (vector<int>::iterator it = miniBatch.begin(); it != miniBatch.end(); it++) {
		for (vector<int>::iterator it = miniBatch.begin(); it != miniBatch.end(); it++) {
			double val = y[*it] - (x*features[*it]);
			f += val*val;
			g -= 2*val*features[*it];
		}
	}
	return;
}

template class LeastSquaresLoss<DenseFeature>;
}
