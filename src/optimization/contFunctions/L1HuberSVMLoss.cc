// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*


        Jensen: A Convex Optimization And Machine Learning ToolKit
 *	Huber SVM Loss with L1 regularization
        Author: Rishabh Iyer
 *
 */

#include <iostream>
#include <math.h>
using namespace std;

#include "L1HuberSVMLoss.h"
#include "../../representation/VectorOperations.h"
#include <assert.h>
#define EPSILON 1e-6
#define MAX 1e2
namespace jensen {
template <class Feature>
L1HuberSVMLoss<Feature>::L1HuberSVMLoss(int m, std::vector<Feature>& features, Vector& y, double thresh, double lambda) :
	ContinuousFunctions(true, m, features.size()), features(features), y(y), thresh(thresh), lambda(lambda)
{
	if (n > 0) {
		assert(features[0].numFeatures == m);
	}
	assert(features.size() == y.size());
}

template <class Feature>
L1HuberSVMLoss<Feature>::L1HuberSVMLoss(const L1HuberSVMLoss& l) :
	ContinuousFunctions(true, l.m, l.n), features(l.features), y(l.y), thresh(l.thresh), lambda(l.lambda) {
}

template <class Feature>
L1HuberSVMLoss<Feature>::~L1HuberSVMLoss(){
}

template <class Feature>
double L1HuberSVMLoss<Feature>::eval(const Vector& x) const {
	assert(x.size() == m);
	double sum = lambda*norm(x, 1);
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
Vector L1HuberSVMLoss<Feature>::evalGradient(const Vector& x) const {
	assert(x.size() == m);
	Vector g = Vector(m, 0);
	for (int i = 0; i < n; i++) {
		double val = y[i]*(x*features[i]);
		if (val <= thresh) {
			g -= features[i]*2*(1 - thresh)*y[i];
		}
		else if ((val > thresh) && (val < 1)) {
			g -= features[i]*2*(1 - val)*y[i];
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
	return g;
}

template <class Feature>
void L1HuberSVMLoss<Feature>::eval(const Vector& x, double& f, Vector& g) const {
	assert(x.size() == m);
	g = Vector(m,0);
	f = lambda*norm(x, 1);
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
Vector L1HuberSVMLoss<Feature>::evalStochasticGradient(const Vector& x, std::vector<int>& miniBatch) const {
	assert(x.size() == m);
	Vector g = Vector(m, 0);
	for (vector<int>::iterator it = miniBatch.begin(); it != miniBatch.end(); it++) {
		double val = y[*it]*(x*features[*it]);
		if (val <= thresh) {
			g -= features[*it]*2*(1 - thresh)*y[*it];
		}
		else if ((val > thresh) && (val < 1)) {
			g -= features[*it]*2*(1 - val)*y[*it];
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
	return g;
}

template <class Feature>
void L1HuberSVMLoss<Feature>::evalStochastic(const Vector& x, double& f, Vector& g, std::vector<int>& miniBatch) const {
	assert(x.size() == m);
	g = Vector(m, 0);
	f = lambda*norm(x, 1);
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
template class L1HuberSVMLoss<SparseFeature>;
template class L1HuberSVMLoss<DenseFeature>;

}
