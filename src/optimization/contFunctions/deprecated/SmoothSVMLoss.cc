// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*
   Jensen: A Convex Optimization And Machine Learning ToolKit
 *	Smooth SVM Loss with L2 regularization
   Author: Rishabh Iyer
 *
 */

#include <iostream>
#include <math.h>
using namespace std;

#include "SmoothSVMLoss.h"
#include "../datarep/VectorOperations.h"
#include <assert.h>
#define EPSILON 1e-6
#define MAX 1e2
namespace jensen {
template <class Feature>
SmoothSVMLoss<Feature>::SmoothSVMLoss(int m, std::vector<Feature>& features, Vector& y) :
	ContinuousFunctions(true, m, features.size()), features(features), y(y)
{
	if (n > 0)
		assert(features[0].numFeatures == m);
	assert(features.size() == y.size());
}

template <class Feature>
SmoothSVMLoss<Feature>::SmoothSVMLoss(const SmoothSVMLoss& l) :
	ContinuousFunctions(true, l.m, l.n), features(l.features), y(l.y) {
}

template <class Feature>
SmoothSVMLoss<Feature>::~SmoothSVMLoss(){
}

template <class Feature>
double SmoothSVMLoss<Feature>::eval(const Vector& x) const {
	assert(x.size() == m);
	double sum = 0;
	for (int i = 0; i < n; i++) {
		double preval = y[i]*(x*features[i]);
		if (1 - preval>= 0) {
			sum += (1 - preval)*(1 - preval);
		}
	}
	return sum;
}

template <class Feature>
Vector SmoothSVMLoss<Feature>::evalGradient(const Vector& x) const {
	assert(x.size() == m);
	Vector g = Vector(m, 0);
	for (int i = 0; i < n; i++) {
		double preval = y[i]*(x*features[i]);
		if (1 - preval>= 0) {
			g -= 2*features[i]*(1 - preval)*y[i];
		}
	}
	return g;
}

template <class Feature>
void SmoothSVMLoss<Feature>::eval(const Vector& x, double& f, Vector& g) const {
	assert(x.size() == m);
	g = Vector(m, 0);
	f = 0;
	for (int i = 0; i < n; i++) {
		double preval = y[i]*(x*features[i]);
		if (1 - preval>= 0) {
			f += (1 - preval)*(1 - preval);
			g -= 2*features[i]*(1 - preval)*y[i];
		}
	}
	return;
}

template <class Feature>
Vector SmoothSVMLoss<Feature>::evalStochasticGradient(const Vector& x, std::vector<int>& miniBatch) const {
	assert(x.size() == m);
	Vector g = Vector(m, 0);
	for (vector<int>::iterator it = miniBatch.begin(); it != miniBatch.end(); it++) {
		double preval = y[*it]*(x*features[*it]);
		if (1 - preval>= 0) {
			g -= 2*features[*it]*(1 - preval)*y[*it];
		}
	}
	return g;
}

template <class Feature>
void SmoothSVMLoss<Feature>::evalStochastic(const Vector& x, double& f, Vector& g, std::vector<int>& miniBatch) const {
	assert(x.size() == m);
	g = Vector(m, 0);
	f = 0;
	double val;
	for (vector<int>::iterator it = miniBatch.begin(); it != miniBatch.end(); it++) {
		double preval = y[*it]*(x*features[*it]);
		if (1 - preval>= 0) {
			f += (1 - preval)*(1 - preval);
			g -= 2*features[*it]*(1 - preval)*y[*it];
		}
	}
	return;
}

template class SmoothSVMLoss<SparseFeature>;
template class SmoothSVMLoss<DenseFeature>;
}
