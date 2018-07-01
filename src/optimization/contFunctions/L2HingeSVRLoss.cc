// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*
	Jensen: A Convex Optimization And Machine Learning ToolKit
  	Smooth SVR Loss
  Author: Rishabh Iyer
  *
  */

#include<iostream>
#include <math.h>
using namespace std;

#include "L2HingeSVRLoss.h"
#include "../../representation/VectorOperations.h"
#include "../../utils/utils.h"
#include <assert.h>
#define EPSILON 1e-6
#define MAX 1e2
namespace jensen {
  template <class Feature>
  L2HingeSVRLoss<Feature>::L2HingeSVRLoss(int m, std::vector<Feature>& features, Vector& y, double lambda, double p): 
    ContinuousFunctions(true, m, features.size()), features(features), y(y), p(p), lambda(lambda)
  {
    if (n > 0)
      assert(features[0].numFeatures == m);
    assert(features.size() == y.size());
  }

  template <class Feature>	
  L2HingeSVRLoss<Feature>::L2HingeSVRLoss(const L2HingeSVRLoss& l) : 
    ContinuousFunctions(true, l.m, l.n), features(l.features), y(l.y), p(l.p), lambda(l.lambda) {}

  template <class Feature>
  L2HingeSVRLoss<Feature>::~L2HingeSVRLoss(){}

  template <class Feature>	
  double L2HingeSVRLoss<Feature>::eval(const Vector& x) const{
    assert(x.size() == m);
    double sum = 0.5*lambda*(x*x);
    for (int i = 0; i < n; i++){
      double preval = (x*features[i]) - y[i];
      if (fabs(preval) - p > 0){
		  	sum += fabs(preval) - p;
      }
    }
    return sum;
  }

  template <class Feature>	
  Vector L2HingeSVRLoss<Feature>::evalGradient(const Vector& x) const{
    assert(x.size() == m);
    Vector g = lambda*x;
    for (int i = 0; i < n; i++){
        double preval = (x*features[i]) - y[i];
        if (fabs(preval) - p > 0){
			g += features[i]*sign(preval);
		}
	}
    return g;
  }

  template <class Feature>
  void L2HingeSVRLoss<Feature>::eval(const Vector& x, double& f, Vector& g) const{
    assert(x.size() == m);
    g = lambda*x;
    f = 0.5*lambda*(x*x);
    for (int i = 0; i < n; i++){
        double preval = (x*features[i]) - y[i];
        if (fabs(preval) - p > 0){
			f += fabs(preval) - p;
			g += features[i]*sign(preval);
		}
    }
    return;
  }

  template <class Feature>	
  Vector L2HingeSVRLoss<Feature>::evalStochasticGradient(const Vector& x, std::vector<int>& miniBatch) const{
    assert(x.size() == m);
    Vector g = lambda*x;
    for (vector<int>::iterator it = miniBatch.begin(); it != miniBatch.end(); it++){
      double preval = (x*features[*it]) - y[*it];
      if (fabs(preval) - p > 0){
		g += features[*it]*sign(preval);
	}
    }
    return g;
  }

  template <class Feature>
  void L2HingeSVRLoss<Feature>::evalStochastic(const Vector& x, double& f, Vector& g, std::vector<int>& miniBatch) const{
    assert(x.size() == m);
    g = lambda*x;
    f = 0.5*lambda*(x*x);
    for (vector<int>::iterator it = miniBatch.begin(); it != miniBatch.end(); it++){
        double preval = (x*features[*it]) - y[*it];
        if (fabs(preval) - p > 0){
			f += fabs(preval) - p;
			g += features[*it]*sign(preval);
		}
    }
    return;
  }

  template class L2HingeSVRLoss<SparseFeature>;
  template class L2HingeSVRLoss<DenseFeature>;
}
