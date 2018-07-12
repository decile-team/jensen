// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*
	Author: Rishabh Iyer
 *
*/

#include <iostream>
#include <cstdlib>
#include "../src/jensen.h"
using namespace jensen;
using namespace std;

int main(int argc, char** argv){
	// char* featureFile = "../data/20newsgroup.feat";
	// char* labelFile = "../data/20newsgroup.label";
	char* featureFile = "../data/20newsgroup.feat";
	char* labelFile = "../data/20newsgroup.label";
	int n; // number of data items
	int m; // numFeatures
	int numEpochs = 50;
	bool checkOld = false;
	vector<struct SparseFeature> features = readFeatureVectorSparse(featureFile, n, m);
	Vector y = readVector(labelFile, n);
	// cout<<y.size()<<"\n";
	L1LogisticLoss <SparseFeature>l(m, features, y, 0);
	// L1 r(m);
	// SumContinuousFunctions ll(l, r, 1);
	L1LogisticLoss<SparseFeature> ll(m, features, y, 1);
	double f;
	Vector g;

	cout<<"*******************************************************************\n";
	cout<<"Testing Gradient Descent with L1-Logistic Loss, press enter to continue...\n";
	#ifndef DEBUG
	cin.get();
	#endif
	gd(ll, Vector(m, 0), 1e-5, numEpochs);
	// cout<<"*******************************************************************\n"
	// cout<<"Testing Gradient Descent with L1-Logistic Loss\n";
	// gradientDescent(ss, Vector(m, 0), 1e-8, numEpochs);

	cout<<"*******************************************************************\n";
	cout<<"Testing Gradient Descent with Line Search for L1-Logistic Loss, press enter to continue...\n";
	#ifndef DEBUG
	cin.get();
	#endif
	gdLineSearch(ll, Vector(m, 0), 1, 1e-4, numEpochs);
	// gradientDescentLS(ss, Vector(m, 0), 1, 1e-4, numEpochs);

	cout<<"*******************************************************************\n";
	cout<<"Testing Gradient Descent with Barzilia-Borwein Step Length for L1-Logistic Loss, press enter to continue...\n";
	#ifndef DEBUG
	cin.get();
	#endif
	gdBarzilaiBorwein(ll, Vector(m, 0), 1, 1e-4, numEpochs);
	// gradientDescentBB(ss, Vector(m, 0), 1, 1e-4, numEpochs);

	cout<<"*******************************************************************\n";
	cout<<"Testing Nesterov's Method for L1-Logistic Loss, press enter to continue...\n";
	#ifndef DEBUG
	cin.get();
	#endif
	gdNesterov(ll, Vector(m, 0), 1, 1e-4, numEpochs);

	cout<<"*******************************************************************\n";
	cout<<"Testing Conjugate Gradient for L1-Logistic Loss, press enter to continue...\n";
	#ifndef DEBUG
	cin.get();
	#endif
	cg(ll, Vector(m, 0), 1, 1e-4, numEpochs);

	cout<<"*******************************************************************\n";
	cout<<"L-BFGS for L1-Logistic Loss, press enter to continue...\n";
	#ifndef DEBUG
	cin.get();
	#endif
	lbfgsMin(ll, Vector(m, 0), 1, 1e-4, numEpochs);

	cout<<"*******************************************************************\n";
	cout<<"Testing LibLinear Coordinate Descent Algorithm, press enter to continue\n";
	#ifndef DEBUG
	cin.get();
	#endif
	Vector x;
	L1LRPrimal(features, y, x, 1, 1e-2);
	// lbfgsMin(ss, Vector(m, 0), 1, 1e-4, numEpochs);

	cout<<"*******************************************************************\n";
	cout<<"L-BFGS-Owl for L1-Logistic Loss, press enter to continue...\n";
	#ifndef DEBUG
	cin.get();
	#endif
	lbfgsMinOwl(ll, Vector(m, 0), 1, 1e-4, numEpochs);
	// lbfgsMin(ss, Vector(m, 0), 1, 1e-4, numEpochs);

	cout<<"*******************************************************************\n";
	cout<<"Stochastic Gradient Descent for L1-Logistic Loss, press enter to continue...\n";
	#ifndef DEBUG
	cin.get();
	#endif
	sgd(ll, Vector(m, 0), n, 1e-4, 100, 1e-4, numEpochs);

	cout<<"*******************************************************************\n";
	cout<<"Stochastic Gradient Descent with Decaying Learning Rate for L1-Logistic Loss, press enter to continue...\n";
	#ifndef DEBUG
	cin.get();
	#endif
	sgdDecayingLearningRate(ll, Vector(m, 0), n, 1e-4, 200, 1e-4, numEpochs, 0.6);

	cout<<"*******************************************************************\n";
	cout<<"Stochastic Gradient Descent with AdaGrad for L1-Logistic Loss, press enter to continue...\n";
	#ifndef DEBUG
	cin.get();
	#endif
	sgdAdagrad(ll, Vector(m, 0), n, 1e-4, 200, 1e-4, numEpochs);

	cout<<"*******************************************************************\n";
	cout<<"LBFGS with orthant-wise projection for L1-regularized Logistic Loss, press enter to continue...\n";
	#ifndef DEBUG
	cin.get();
	#endif
	lbfgsMinOwl(ll, Vector(m, 0), 1, 1e-0, numEpochs, 100, 1e-4);

	cout<<"*******************************************************************\n";

	cout<<"SGD with L1-regularized Dual-Averaging for Logistic Loss, press enter to continue...\n";
	#ifndef DEBUG
	cin.get();
	#endif
	sgdRegularizedDualAveraging(l, ll, Vector(m, 0), n, 1e-1, 1e-3, 200, 1e-4, numEpochs, 0.5);
}
