// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*
   Jensen: A Convex Optimization And Machine Learning ToolKit
 *	Smooth SVR Loss
   Author: Rishabh Iyer
 *
 */

#include <iostream>
#include <cmath>
#include "../utils/utils.h"
using namespace std;

#include "HingeSVRLoss.h"
#include "../datarep/VectorOperations.h"
#include <assert.h>
#define EPSILON 1e-6
#define MAX 1e2
namespace jensen {
template <class Feature>
HingeSVRLoss<Feature>::HingeSVRLoss(int m, std::vector<Feature>& features, Vector& y, double p) :
	ContinuousFunctions(true, m, features.size()), features(features), y(y), p(p)
{
	if (n > 0)
		assert(features[0].numFeatures == m);
	assert(features.size() == y.size());
}

template <class Feature>
HingeSVRLoss<Feature>::HingeSVRLoss(const HingeSVRLoss& l) :
	ContinuousFunctions(true, l.m, l.n), features(l.features), y(l.y), p(l.p) {
}

template <class Feature>
HingeSVRLoss<Feature>::~HingeSVRLoss(){
}

template <class Feature>
double HingeSVRLoss<Feature>::eval(const Vector& x) const {
	assert(x.size() == m);
	double sum = 0;
	for (int i = 0; i < n; i++) {
		double preval = (x*features[i]) - y[i];
		if (fabs(preval) - p > 0) {
			sum += fabs(preval) - p;
		}
	}
	return sum;
}

template <class Feature>
Vector HingeSVRLoss<Feature>::evalGradient(const Vector& x) const {
	assert(x.size() == m);
	Vector g = Vector(m, 0);
	for (int i = 0; i < n; i++) {
		double preval = (x*features[i]) - y[i];
		if (fabs(preval) - p > 0) {
			g -= features[i]*sign(preval);
		}
	}
	return g;
}

template <class Feature>
void HingeSVRLoss<Feature>::eval(const Vector& x, double& f, Vector& g) const {
	assert(x.size() == m);
	g = Vector(m, 0);
	f = 0;
	for (int i = 0; i < n; i++) {
		double preval = (x*features[i]) - y[i];
		if (fabs(preval) - p > 0) {
			f += fabs(preval) - p;
			g -= features[i]*sign(preval);
		}
	}
	return;
}

template <class Feature>
Vector HingeSVRLoss<Feature>::evalStochasticGradient(const Vector& x, std::vector<int>& miniBatch) const {
	assert(x.size() == m);
	Vector g = Vector(m, 0);
	for (vector<int>::iterator it = miniBatch.begin(); it != miniBatch.end(); it++) {
		double preval = (x*features[*it]) - y[*it];
		if (fabs(preval) - p > 0) {
			g -= features[*it]*sign(preval);
		}
	}
	return g;
}

template <class Feature>
void HingeSVRLoss<Feature>::evalStochastic(const Vector& x, double& f, Vector& g, std::vector<int>& miniBatch) const {
	assert(x.size() == m);
	g = Vector(m, 0);
	f = 0;
	for (vector<int>::iterator it = miniBatch.begin(); it != miniBatch.end(); it++) {
		double preval = (x*features[*it]) - y[*it];
		if (fabs(preval) - p > 0) {
			f += fabs(preval) - p;
			g -= features[*it]*sign(preval);
		}
	}
	return;
}

template class HingeSVRLoss<SparseFeature>;
template class HingeSVRLoss<DenseFeature>;
}
