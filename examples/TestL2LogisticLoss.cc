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
	//char* featureFile = "../jensenData/sidoF.txt";
	//char* labelFile = "../jensenData/sidoL.txt";
	char* featureFile = "../jensenData/20newsgroup.feat";
	char* labelFile = "../jensenData/20newsgroup.label";
	int n; // number of data items
	int m; // numFeatures
	bool checkOld = false;
	vector<struct SparseFeature> features = readFeatureVectorSparse(featureFile, n, m);
	Vector y = readVector(labelFile, n);
	cout<<y.size()<<"\n";
	// LogisticLoss l(m, features, y);
	// L2 r(m);
	// SumContinuousFunctions ll(l, r, 0.5);
	L2LogisticLoss<SparseFeature> ll(m, features, y, 1);
	Vector x(m, 1);
	double f;
	Vector g;

	cout<<"*******************************************************************\n";
	cout<<"Testing Gradient Descent with Logistic Loss, press enter to continue...\n";
	#ifndef DEBUG
	cin.get();
	#endif
	gd(ll, Vector(m, 0), 1e-5, 250);
	// cout<<"*******************************************************************\n"
	// cout<<"Testing Gradient Descent with Logistic Loss\n";
	// gradientDescent(ss, Vector(m, 0), 1e-8, 250);

	cout<<"*******************************************************************\n";
	cout<<"Testing Gradient Descent with Line Search for Logistic Loss, press enter to continue...\n";
	#ifndef DEBUG
	cin.get();
	#endif
	gdLineSearch(ll, Vector(m, 0), 1, 1e-4, 250);
	// gradientDescentLS(ss, Vector(m, 0), 1, 1e-4, 250);

	cout<<"*******************************************************************\n";
	cout<<"Testing Gradient Descent with Barzilia-Borwein Step Length for Logistic Loss, press enter to continue...\n";
	#ifndef DEBUG
	cin.get();
	#endif
	gdBarzilaiBorwein(ll, Vector(m, 0), 1, 1e-4, 250);
	// gradientDescentBB(ss, Vector(m, 0), 1, 1e-4, 250);

	cout<<"*******************************************************************\n";
	cout<<"Testing Nesterov's Method for Logistic Loss, press enter to continue...\n";
	#ifndef DEBUG
	cin.get();
	#endif
	gdNesterov(ll, Vector(m, 0), 1, 1e-4, 250);

	cout<<"*******************************************************************\n";
	cout<<"Testing Conjugate Gradient for Logistic Loss, press enter to continue...\n";
	#ifndef DEBUG
	cin.get();
	#endif
	cg(ll, Vector(m, 0), 1, 1e-4, 250);

	cout<<"*******************************************************************\n";
	cout<<"L-BFGS for Logistic Loss, press enter to continue...\n";
	#ifndef DEBUG
	cin.get();
	#endif
	Vector w = lbfgsMin(ll, Vector(m, 0), 1, 1e-4, 250);
	// lbfgsMin(ss, Vector(m, 0), 1, 1e-4, 250);

	cout<<"*******************************************************************\n";
	cout<<"Trust Region Newton Method for Logistic Loss, press enter to continue...\n";
	#ifndef DEBUG
	cin.get();
	#endif
	tron(ll, Vector(m, 0), 250);
	// lbfgsMin(ss, Vector(m, 0), 1, 1e-4, 250);

	cout<<"*******************************************************************\n";
	cout<<"Stochastic Gradient Descent for Logistic Loss, press enter to continue...\n";
	#ifndef DEBUG
	cin.get();
	#endif
	sgd(ll, Vector(m, 0), n, 1e-4, 100, 1e-4, 250);

	cout<<"*******************************************************************\n";
	cout<<"Stochastic Gradient Descent with Decaying Learning Rate for Logistic Loss, press enter to continue...\n";
	#ifndef DEBUG
	cin.get();
	#endif
	sgdDecayingLearningRate(ll, Vector(m, 0), n, 0.5*1e-1, 200, 1e-4, 250, 0.6);

	cout<<"*******************************************************************\n";
	cout<<"Stochastic Gradient Descent with AdaGrad for Logistic Loss, press enter to continue...\n";
	#ifndef DEBUG
	cin.get();
	#endif
	sgdAdagrad(ll, Vector(m, 0), n, 1e-2, 200, 1e-4, 250);

	cout<<"*******************************************************************\n";
	cout<<"SGD with Stochastic Average Gradient for Logistic Loss, press enter to continue...\n";
	#ifndef DEBUG
	cin.get();
	#endif
	sgdStochasticAverageGradient(ll, Vector(m, 0), n, 2, 1, 200, 1e-4, 250, 1.0);
	// cout<<"*******************************************************************\n";
	// cout<<"Stochastic Gradient Descent with Line Search for Logistic Loss, press enter to continue...\n";
	// #ifndef DEBUG
	// cin.get();
	// #endif
	// sgdLineSearch(ll, Vector(m, 0), n, 1e-2, 200, 1e-4, 250);

}
