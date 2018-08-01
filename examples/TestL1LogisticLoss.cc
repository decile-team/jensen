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
	// const char* featureFile = "../jensenData/sidoF.txt";
	// const char* labelFile = "../jensenData/sidoL.txt";
	const char* featureFile = "../jensenData/20newsgroup.feat";
	const char* labelFile = "../jensenData/20newsgroup.label";
	int n; // number of data items
	int m; // numFeatures
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
	gd(ll, Vector(m, 0), 1e-5, 250);
	// cout<<"*******************************************************************\n"
	// cout<<"Testing Gradient Descent with L1-Logistic Loss\n";
	// gradientDescent(ss, Vector(m, 0), 1e-8, 250);

	cout<<"*******************************************************************\n";
	cout<<"Testing Gradient Descent with Line Search for L1-Logistic Loss, press enter to continue...\n";
	#ifndef DEBUG
	cin.get();
	#endif
	gdLineSearch(ll, Vector(m, 0), 1, 1e-4, 250);
	// gradientDescentLS(ss, Vector(m, 0), 1, 1e-4, 250);

	cout<<"*******************************************************************\n";
	cout<<"Testing Gradient Descent with Barzilia-Borwein Step Length for L1-Logistic Loss, press enter to continue...\n";
	#ifndef DEBUG
	cin.get();
	#endif
	gdBarzilaiBorwein(ll, Vector(m, 0), 1, 1e-4, 250);
	// gradientDescentBB(ss, Vector(m, 0), 1, 1e-4, 250);

	cout<<"*******************************************************************\n";
	cout<<"Testing Nesterov's Method for L1-Logistic Loss, press enter to continue...\n";
	#ifndef DEBUG
	cin.get();
	#endif
	gdNesterov(ll, Vector(m, 0), 1, 1e-4, 250);

	cout<<"*******************************************************************\n";
	cout<<"Testing Conjugate Gradient for L1-Logistic Loss, press enter to continue...\n";
	#ifndef DEBUG
	cin.get();
	#endif
	cg(ll, Vector(m, 0), 1, 1e-4, 250);

	cout<<"*******************************************************************\n";
	cout<<"L-BFGS for L1-Logistic Loss, press enter to continue...\n";
	#ifndef DEBUG
	cin.get();
	#endif
	lbfgsMin(ll, Vector(m, 0), 1, 1e-4, 250);

	cout<<"*******************************************************************\n";
	cout<<"Testing LibLinear Coordinate Descent Algorithm, press enter to continue\n";
	#ifndef DEBUG
	cin.get();
	#endif
	Vector x;
	L1LRPrimal(features, y, x, 1, 1e-2);
	// lbfgsMin(ss, Vector(m, 0), 1, 1e-4, 250);

	cout<<"*******************************************************************\n";
	cout<<"L-BFGS-Owl for L1-Logistic Loss, press enter to continue...\n";
	#ifndef DEBUG
	cin.get();
	#endif
	lbfgsMinOwl(ll, Vector(m, 0), 1, 1e-4, 250);
	// lbfgsMin(ss, Vector(m, 0), 1, 1e-4, 250);

	cout<<"*******************************************************************\n";
	cout<<"Stochastic Gradient Descent for L1-Logistic Loss, press enter to continue...\n";
	#ifndef DEBUG
	cin.get();
	#endif
	sgd(ll, Vector(m, 0), n, 1e-4, 100, 1e-4, 250);

	cout<<"*******************************************************************\n";
	cout<<"Stochastic Gradient Descent with Decaying Learning Rate for L1-Logistic Loss, press enter to continue...\n";
	#ifndef DEBUG
	cin.get();
	#endif
	sgdDecayingLearningRate(ll, Vector(m, 0), n, 1e-4, 200, 1e-4, 250, 0.6);

	cout<<"*******************************************************************\n";
	cout<<"Stochastic Gradient Descent with AdaGrad for L1-Logistic Loss, press enter to continue...\n";
	#ifndef DEBUG
	cin.get();
	#endif
	sgdAdagrad(ll, Vector(m, 0), n, 1e-4, 200, 1e-4, 250);

	cout<<"*******************************************************************\n";
	cout<<"LBFGS with orthant-wise projection for L1-regularized Logistic Loss, press enter to continue...\n";
	#ifndef DEBUG
	cin.get();
	#endif
	lbfgsMinOwl(ll, Vector(m, 0), 1, 1e-0, 250, 100, 1e-4);

	cout<<"*******************************************************************\n";
	cout<<"SGD with L1-regularized Dual-Averaging and AdaGrad for Logistic Loss, press enter to continue...\n";
	#ifndef DEBUG
	cin.get();
	#endif
	sgdRegularizedDualAveragingAdagrad(ll, l, Vector(m, 0), n, 1e-1, 1e-3, 200, 1e-4, 250);

	cout<<"*******************************************************************\n";
	cout<<"SGD with L1-regularized Dual-Averaging for Logistic Loss, press enter to continue...\n";
	#ifndef DEBUG
	cin.get();
	#endif
	sgdRegularizedDualAveraging(ll, l, Vector(m, 0), n, 1e-1, 1e-3, 200, 1e-4, 250, 0.5);


	// cout<<"*******************************************************************\n";
	// cout<<"Stochastic Gradient Descent with Line Search for L1-Logistic Loss, press enter to continue...\n";
	// #ifndef DEBUG
	// cin.get();
	// #endif
	// sgdLineSearch(ll, Vector(m, 0), n, 1e-2, 200, 1e-4, 250);

}
