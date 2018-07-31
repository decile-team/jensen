// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*
        Jensen: A Convex Optimization And Machine Learning ToolKit
 *	Abstract base class for Classifiers
        Author: Rishabh Iyer
 *
 */

#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include "../representation/Vector.h"
#include "../representation/Matrix.h"
#include "../representation/VectorOperations.h"
#include "../representation/MatrixOperations.h"
#include "../representation/DenseFeature.h"
#include "../representation/SparseFeature.h"

#include <vector>
using namespace std;

namespace jensen {

template <class Feature>
class Classifiers {
protected:
int m;
int n;
public:
Classifiers();
Classifiers(int m, int n);
Classifiers(const Classifiers& c);         // copy constructor
virtual ~Classifiers();

virtual void train() = 0;                 // train

virtual int saveModel(char* model) = 0;                 // save the model
virtual int loadModel(char* model) = 0;                 // save the model

virtual double predict(const Feature& testFeature) = 0;
virtual double predict(const Feature& testFeature, double& val) = 0;
virtual void predictProbability(const Feature& testFeature, Vector& prob) = 0;

double operator()(const Feature& testFeature);

int size();                 // number of features or dimension size (m)
int length();                 // number of training examples
};

}
#endif
