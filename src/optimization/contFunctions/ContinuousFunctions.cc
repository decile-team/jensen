// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*

  Jensen: A Convex Optimization And Machine Learning ToolKit

  *	Abstract base class for Continuous Functions
  Author: Rishabh Iyer
  *
  */

#include<iostream>
using namespace std;

#include "ContinuousFunctions.h"
#define EPSILON 1e-6
namespace jensen {
  ContinuousFunctions::ContinuousFunctions(bool isSmooth): isSmooth(isSmooth){m = 0; n = 0;}
  ContinuousFunctions::ContinuousFunctions(bool isSmooth, int m, int n): isSmooth(isSmooth), m(m), n(n){}
  ContinuousFunctions::ContinuousFunctions(const ContinuousFunctions& c) : isSmooth(c.isSmooth), m(c.m), n(c.n) {}

  ContinuousFunctions::~ContinuousFunctions(){}
	
  double ContinuousFunctions::eval(const Vector& x) const{
    return 0;
  }
	
  Vector ContinuousFunctions::evalGradient(const Vector& x) const{ // in case the function is non-differentiable, this is the subgradient
    Vector gradient(m, 0);
    Vector xdiff(x);
    for (int i = 0; i < m; i++){
      xdiff[i]+=EPSILON;
      gradient[i] = (eval(xdiff) - eval(x))/EPSILON;
      xdiff[i]-=EPSILON;
    }
    return gradient;
  }

  void ContinuousFunctions::eval(const Vector& x, double& f, Vector& gradient) const{
    gradient = evalGradient(x);
    f = eval(x);
    return;
  }
	
  Vector ContinuousFunctions::evalStochasticGradient(const Vector& x, std::vector<int>& batch) const{
    return evalGradient(x);
  }
	
  void ContinuousFunctions::evalStochastic(const Vector& x, double& f, Vector& g, std::vector<int>& miniBatch) const{
    return eval(x, f, g);
  }
  Matrix ContinuousFunctions::evalHessian(const Vector& x) const{
    Matrix hessian;
    Vector xdiff(x);
    for (int i = 0; i < m; i++){
      xdiff[i]+=EPSILON;
      hessian.push_back(evalGradient(xdiff) - evalGradient(x));
      xdiff[i]-=EPSILON;
    }
    return hessian;
  }
	
  void ContinuousFunctions::evalHessianVectorProduct(const Vector& x, const Vector& v, Vector& Hxv) const{
    Matrix hessian = evalHessian(x);
    Hxv = hessian*v;
  }
	
  double ContinuousFunctions::operator()(const Vector& x) const
  {
    return eval(x);
  }
	
  int ContinuousFunctions::size() const{ // number of features or dimension size
    return m;
  }
	
  int ContinuousFunctions::length() const{ // number of convex functions adding up
    return n;
  }
}
