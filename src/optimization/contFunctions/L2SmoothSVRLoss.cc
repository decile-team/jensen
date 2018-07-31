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
#include <math.h>
using namespace std;

#include "L2SmoothSVRLoss.h"
#include "../../representation/VectorOperations.h"
#include <assert.h>
#define EPSILON 1e-6
#define MAX 1e2
namespace jensen {
template <class Feature>
L2SmoothSVRLoss<Feature>::L2SmoothSVRLoss(int m, std::vector<Feature>& features, Vector& y, double lambda, double p) :
	ContinuousFunctions(true, m, features.size()), features(features), y(y), p(p), lambda(lambda)
{
	if (n > 0)
		assert(features[0].numFeatures == m);
	assert(features.size() == y.size());
}

template <class Feature>
L2SmoothSVRLoss<Feature>::L2SmoothSVRLoss(const L2SmoothSVRLoss& l) :
	ContinuousFunctions(true, l.m, l.n), features(l.features), y(l.y), p(l.p), lambda(l.lambda) {
}

template <class Feature>
L2SmoothSVRLoss<Feature>::~L2SmoothSVRLoss(){
}

template <class Feature>
double L2SmoothSVRLoss<Feature>::eval(const Vector& x) const {
	assert(x.size() == m);
	double sum = 0.5*lambda*(x*x);
	for (int i = 0; i < n; i++) {
		double preval = (x*features[i]) - y[i];
		if (preval < -p) {
			sum += (preval + p)*(preval + p);
		}
		else if (preval > p) {
			sum += (preval - p)*(preval - p);
		}
	}
	return sum;
}

template <class Feature>
Vector L2SmoothSVRLoss<Feature>::evalGradient(const Vector& x) const {
	assert(x.size() == m);
	Vector g = lambda*x;
	for (int i = 0; i < n; i++) {
		double preval = (x*features[i]) - y[i];
		if (preval < -p) {
			g += 2*features[i]*(preval + p);
		}
		else if (preval > p) {}
		g += 2*features[i]*(preval - p);
	}
	return g;
}

template <class Feature>
void L2SmoothSVRLoss<Feature>::eval(const Vector& x, double& f, Vector& g) const {
	assert(x.size() == m);
	g = lambda*x;
	f = 0.5*lambda*(x*x);
	for (int i = 0; i < n; i++) {
		double preval = (x*features[i]) - y[i];
		if (preval < -p) {
			f += (preval + p)*(preval + p);
			g += 2*features[i]*(preval + p);
		}
		else if (preval > p) {
			f += (preval - p)*(preval - p);
			g += 2*features[i]*(preval - p);
		}
	}
	return;
}

template <class Feature>
Vector L2SmoothSVRLoss<Feature>::evalStochasticGradient(const Vector& x, std::vector<int>& miniBatch) const {
	assert(x.size() == m);
	Vector g = lambda*x;
	for (vector<int>::iterator it = miniBatch.begin(); it != miniBatch.end(); it++) {
		double preval = (x*features[*it]) - y[*it];
		if (preval < -p) {
			g += 2*features[*it]*(preval + p);;
		}
		else if (preval > p) {
			g += 2*features[*it]*(preval - p);
		}
	}
	return g;
}

template <class Feature>
void L2SmoothSVRLoss<Feature>::evalStochastic(const Vector& x, double& f, Vector& g, std::vector<int>& miniBatch) const {
	assert(x.size() == m);
	g = lambda*x;
	f = 0.5*lambda*(x*x);
	for (vector<int>::iterator it = miniBatch.begin(); it != miniBatch.end(); it++) {
		double preval = (x*features[*it]) - y[*it];
		if (preval < -p) {
			f += (preval + p)*(preval + p);
			g += 2*features[*it]*(preval + p);
		}
		else if (preval > p) {
			f += (preval - p)*(preval - p);
			g += 2*features[*it]*(preval - p);
		}
	}
	return;
}

template class L2SmoothSVRLoss<SparseFeature>;
template class L2SmoothSVRLoss<DenseFeature>;
}
