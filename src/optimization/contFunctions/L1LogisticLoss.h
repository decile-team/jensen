// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*
	Jensen: A Convex Optimization And Machine Learning ToolKit
 *	Logistic Loss with L2 regularization
	Author: Rishabh Iyer
 *
 */

#ifndef L1_LOGISTIC_LOSS_H
#define L1_LOGISTIC_LOSS_H

#include "../../representation/Vector.h"
#include "../../representation/Matrix.h"
#include "../../representation/VectorOperations.h"
#include "ContinuousFunctions.h"
#include "../../utils/utils.h"

namespace jensen {
  template <class Feature>
	class L1LogisticLoss: public ContinuousFunctions{
	protected:
		std::vector<Feature>& features; // size of features is number of trainins examples (n)
		Vector& y; // size of y is number of training examples (n)
		double lambda;
	public:
		L1LogisticLoss(int numFeatures, std::vector<Feature>& features, Vector& y, double lambda);
        L1LogisticLoss(const L1LogisticLoss& c); // copy constructor
		
		~L1LogisticLoss();
		
		double eval(const Vector& x) const; // functionEval
		Vector evalGradient(const Vector& x) const; // gradientEval
		void eval(const Vector& x, double& f, Vector& gradient) const; // combined function and gradient eval
		Vector evalStochasticGradient(const Vector& x, std::vector<int>& miniBatch) const; // stochastic gradient
		void evalStochastic(const Vector& x, double& f, Vector& g, std::vector<int>& miniBatch) const; // stochastic evaluation
	};
	
}
#endif
