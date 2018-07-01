// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*
  Jensen: A Convex Optimization And Machine Learning ToolKit
  *	Least Squares Loss with L1 regularization
  Author: Rishabh Iyer
  *
  */

#ifndef L1_H
#define L1_H

#include "../datarep/Vector.h"
#include "../datarep/Matrix.h"
#include "../datarep/VectorOperations.h"
#include "ContinuousFunctions.h"
namespace jensen {

  class L1: public ContinuousFunctions{
  public:
    L1(int m);
    L1(const L1& l2); // copy constructor
		
    ~L1();
		
    double eval(const Vector& x) const; // functionEval
    Vector evalGradient(const Vector& x) const; // gradientEval
    void eval(const Vector& x, double& f, Vector& gradient) const; // combined function and gradient eval
  };
	
}
#endif
