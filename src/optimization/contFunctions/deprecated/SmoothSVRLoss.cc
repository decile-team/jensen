// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*
  Jensen: A Convex Optimization And Machine Learning ToolKit
  *	Smooth SVR Loss
  Author: Rishabh Iyer
  *
  */

#include<iostream>
#include <math.h>
using namespace std;

#include "SmoothSVRLoss.h"
#include "../datarep/VectorOperations.h"
#include <assert.h>
#define EPSILON 1e-6
#define MAX 1e2
namespace jensen {	
  template <class Feature>
  SmoothSVRLoss<Feature>::SmoothSVRLoss(int m, std::vector<Feature>& features, Vector& y, double p): 
    ContinuousFunctions(true, m, features.size()), features(features), y(y), p(p)
  {
    if (n > 0)
      assert(features[0].numFeatures == m);
    assert(features.size() == y.size());
  }

  template <class Feature>	
  SmoothSVRLoss<Feature>::SmoothSVRLoss(const SmoothSVRLoss& l) : 
    ContinuousFunctions(true, l.m, l.n), features(l.features), y(l.y), p(l.p) {}

  template <class Feature>
  SmoothSVRLoss<Feature>::~SmoothSVRLoss(){}

  template <class Feature>	
  double SmoothSVRLoss<Feature>::eval(const Vector& x) const{
    assert(x.size() == m);
    double sum = 0;
    for (int i = 0; i < n; i++){
      double preval = (x*features[i]) - y[i];
      if (preval < -p){
		  	sum += (preval + p)*(preval + p);
      }
	  else if (preval > p){
		  sum += (preval - p)*(preval - p);
	  }
    }
    return sum;
  }

  template <class Feature>	
  Vector SmoothSVRLoss<Feature>::evalGradient(const Vector& x) const{
    assert(x.size() == m);
    Vector g = Vector(m, 0);
    for (int i = 0; i < n; i++){
        double preval = (x*features[i]) - y[i];
        if (preval < -p){
  		  	g += 2*features[i]*(preval + p);
        }
  	  else if (preval > p){}
  		  g += 2*features[i]*(preval - p);
      }
    return g;
  }

  template <class Feature>
  void SmoothSVRLoss<Feature>::eval(const Vector& x, double& f, Vector& g) const{
    assert(x.size() == m);
    g = Vector(m, 0);
    f = 0;
    for (int i = 0; i < n; i++){
        double preval = (x*features[i]) - y[i];
        if (preval < -p){
  		  	f += (preval + p)*(preval + p);
  		  	g += 2*features[i]*(preval + p);
        }
  	  else if (preval > p){
  		  f += (preval - p)*(preval - p);
  		  g += 2*features[i]*(preval - p);
  	  }
    }
    return;
  }

  template <class Feature>	
  Vector SmoothSVRLoss<Feature>::evalStochasticGradient(const Vector& x, std::vector<int>& miniBatch) const{
    assert(x.size() == m);
    Vector g = Vector(m, 0);
    for (vector<int>::iterator it = miniBatch.begin(); it != miniBatch.end(); it++){
      double preval = (x*features[*it]) - y[*it];
      if (preval < -p){
		  g += 2*features[*it]*(preval + p);;
      }
  	  else if (preval > p){
  		  g += 2*features[*it]*(preval - p);
	  }
    }
    return g;
  }

  template <class Feature>
  void SmoothSVRLoss<Feature>::evalStochastic(const Vector& x, double& f, Vector& g, std::vector<int>& miniBatch) const{
    assert(x.size() == m);
    g = Vector(m, 0);
    f = 0;
    for (vector<int>::iterator it = miniBatch.begin(); it != miniBatch.end(); it++){
        double preval = (x*features[*it]) - y[*it];
        if (preval < -p){
  		  	f += (preval + p)*(preval + p);
  		  	g += 2*features[*it]*(preval + p);
        }
  	  else if (preval > p){
  		  f += (preval - p)*(preval - p);
  		  g += 2*features[*it]*(preval - p);
  	  }
    }
    return;
  }

  template class SmoothSVRLoss<SparseFeature>;
  template class SmoothSVRLoss<DenseFeature>;
}
