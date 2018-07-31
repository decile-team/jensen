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

#include "L2HuberSVMLoss.h"
#include "../../representation/VectorOperations.h"
#include <assert.h>
#define EPSILON 1e-6
#define MAX 1e2
namespace jensen {
template <class Feature>
L2HuberSVMLoss<Feature>::L2HuberSVMLoss(int m, std::vector<Feature>& features, Vector& y, double thresh, double lambda) :
	ContinuousFunctions(true, m, features.size()), features(features), y(y), thresh(thresh), lambda(lambda)
{
	if (n > 0) {
		assert(features[0].numFeatures == m);
	}
	assert(features.size() == y.size());
}

template <class Feature>
L2HuberSVMLoss<Feature>::L2HuberSVMLoss(const L2HuberSVMLoss& l) :
	ContinuousFunctions(true, l.m, l.n), features(l.features), y(l.y), thresh(l.thresh), lambda(l.lambda) {
}

template <class Feature>
L2HuberSVMLoss<Feature>::~L2HuberSVMLoss(){
}

template <class Feature>
double L2HuberSVMLoss<Feature>::eval(const Vector& x) const {
	assert(x.size() == m);
	double sum = 0.5*lambda*(x*x);
	for (int i = 0; i < n; i++) {
		double val = y[i]*(x*features[i]);
		if (val <= thresh) {
			sum += (1 - thresh)*(1 - thresh) + 2*(1 - thresh)*(thresh - val);
		}
		else if ((val > thresh) && (val < 1)) {
			sum += (1 - val)*(1 - val);
		}
	}
	return sum;
}

template <class Feature>
Vector L2HuberSVMLoss<Feature>::evalGradient(const Vector& x) const {
	assert(x.size() == m);
	Vector g = lambda*x;
	for (int i = 0; i < n; i++) {
		double val = y[i]*(x*features[i]);
		if (val <= thresh) {
			g -= features[i]*2*(1 - thresh)*y[i];
		}
		else if ((val > thresh) && (val < 1)) {
			g -= features[i]*2*(1 - val)*y[i];
		}
	}
	return g;
}

template <class Feature>
void L2HuberSVMLoss<Feature>::eval(const Vector& x, double& f, Vector& g) const {
	assert(x.size() == m);
	g = lambda*x;
	f = 0.5*lambda*(x*x);
	for (int i = 0; i < n; i++) {
		double val = y[i]*(x*features[i]);
		if (val <= thresh) {
			f += (1 - thresh)*(1 - thresh) + 2*(1 - thresh)*(thresh - val);
			g -= features[i]*2*(1 - thresh)*y[i];
		}
		else if ((val > thresh) && (val <= 1)) {
			f += (1 - val)*(1 - val);
			g -= features[i]*2*(1 - val)*y[i];
		}
	}
	return;
}

template <class Feature>
Vector L2HuberSVMLoss<Feature>::evalStochasticGradient(const Vector& x, std::vector<int>& miniBatch) const {
	assert(x.size() == m);
	Vector g = lambda*x;
	for (vector<int>::iterator it = miniBatch.begin(); it != miniBatch.end(); it++) {
		double val = y[*it]*(x*features[*it]);
		if (val <= thresh) {
			g -= features[*it]*2*(1 - thresh)*y[*it];
		}
		else if ((val > thresh) && (val < 1)) {
			g -= features[*it]*2*(1 - val)*y[*it];
		}
	}
	return g;
}

template <class Feature>
void L2HuberSVMLoss<Feature>::evalStochastic(const Vector& x, double& f, Vector& g, std::vector<int>& miniBatch) const {
	assert(x.size() == m);
	g = lambda*x;
	f = 0.5*lambda*(x*x);
	double val;
	for (vector<int>::iterator it = miniBatch.begin(); it != miniBatch.end(); it++) {
		double val = y[*it]*(x*features[*it]);
		if (val <= thresh) {
			f += (1 - thresh)*(1 - thresh) + 2*(1 - thresh)*(thresh - val);
			g -= features[*it]*2*(1 - thresh)*y[*it];
		}
		else if ((val > thresh) && (val < 1)) {
			f += (1 - val)*(1 - val);
			g -= features[*it]*2*(1 - val)*y[*it];
		}
	}
	return;
}
template class L2HuberSVMLoss<SparseFeature>;
template class L2HuberSVMLoss<DenseFeature>;


}
