// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*
        Jensen: A Convex Optimization And Machine Learning ToolKit
 *	L1 Regularized Logistic Regression (Useful if you want to encourage sparsity in the classifier)
        Author: Rishabh Iyer

    algtype: type of algorithm:
                0 (Dual Coordinate Descent from LIBLINEAR )
                1 (LBFGS)
                2 (Gradient Descent),
                3 (Gradient Descent with Line Search),
                4 (Gradient Descent with Barzelie Borwein step size),
                5 (Conjugate Gradient),
                6 (Nesterov's optimal method),
                7 (Stochastic Gradient Descent with fixed step length)
                8 (Stochastic Gradient Descent with decaying step size)
                9 (Adaptive Gradient Algorithm (AdaGrad))
 *
 */

#ifndef L2_SMOOTH_SVM_H
#define L2_SMOOTH_SVM_H

#include "../Classifiers.h"
#include "../../representation/Vector.h"
#include "../../representation/Matrix.h"
#include "../../representation/VectorOperations.h"
#include "../../representation/MatrixOperations.h"
#include <vector>
using namespace std;

namespace jensen {

template <class Feature>
class L2SmoothSVM : public Classifiers<Feature>{
protected:
vector<Feature>& trainFeatures;                 // training features
Vector& y;             // size of y is number of training examples (n)
int algtype;                 // the algorithm type used for training, default is the trust region newton.
vector<Vector> wMany;                 // the weights in the multiclass scenario -- nClasses number of weight vectors.
Vector w;                 // the weights in the binary scenario.
int nClasses;                 // the number of classes
double lambda;                 // regularization
int maxIter;                 // maximum number of iterations for the algorithms
double eps;                 // stopping criterion for the algorithms
int miniBatch;                 // This is the miniBatch size for Stochastic Gradient Descent
int lbfgsMemory;                 // memory of the LBFGS algorithm.
using Classifiers<Feature>::m;
using Classifiers<Feature>::n;
public:
L2SmoothSVM(vector<Feature>& trainFeatures, Vector& y, int m, int n, int nClasses = 2, double lambda = 1,
            int algtype = 0, int maxIter = 250, double eps = 1e-2, int miniBatch = 100, int lbfgsMemory = 100);
L2SmoothSVM(const L2SmoothSVM& c);         // copy constructor
~L2SmoothSVM();

void trainOne(Vector& yOne, Vector& wcurr);                 // a member function for binary classification.
void train();                 // train

int saveModel(char* model);                 // save the model
int loadModel(char* model);                 // save the model

double predict(const Feature& testFeature);
double predict(const Feature& testFeature, double& val);
void predictProbability(const Feature& testFeature, Vector& prob);

};

}
#endif
