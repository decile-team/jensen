// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*
        Author: Rishabh Iyer
 *
 */

#include <iostream>
#include "../src/jensen.h"
using namespace jensen;
using namespace std;

int main(int argc, char** argv){
	const char* featureFile = "../jensenData/sidoF.txt";
	const char* labelFile = "../jensenData/sidoL.txt";
	int n; // number of data items
	int m; // numFeatures
	vector<struct SparseFeature> features = readFeatureVectorSparse(featureFile, n, m);
	Vector y = readVector(labelFile, n);
	cout<<y.size()<<"\n";
	// HuberSVMLoss l(m, features, y, 0.5);
	// L1 r(m);
	// SumContinuousFunctions ll(l, r, 1);
	L1HuberSVMLoss<SparseFeature> ll(m, features, y, 0.5, 1);

	double f;
	Vector g;
	cout<<"*******************************************************************\n";
	cout<<"Testing Gradient Descent with Huber SVM Loss, press enter to continue...\n";
	#ifndef DEBUG
	cin.get();
	#endif
	gd(ll, Vector(m, 0), 1e-5, 250);

	cout<<"*******************************************************************\n";
	cout<<"Testing Gradient Descent with Line Search for Huber SVM Loss, press enter to continue...\n";
	#ifndef DEBUG
	cin.get();
	#endif
	gdLineSearch(ll, Vector(m, 0), 1, 1e-4, 250);
	// gradientDescentLS(ss, Vector(m, 0), 1, 1e-4, 250);

	cout<<"*******************************************************************\n";
	cout<<"Testing Gradient Descent with Barzilia-Borwein Step Length for Huber SVM Loss, press enter to continue...\n";
	#ifndef DEBUG
	cin.get();
	#endif
	gdBarzilaiBorwein(ll, Vector(m, 0), 1, 1e-4, 250);
	// gradientDescentBB(ss, Vector(m, 0), 1, 1e-4, 250);

	cout<<"*******************************************************************\n";
	cout<<"Testing Nesterov's Method for Huber SVM Loss, press enter to continue...\n";
	#ifndef DEBUG
	cin.get();
	#endif
	gdNesterov(ll, Vector(m, 0), 1, 1e-4, 250);

	cout<<"*******************************************************************\n";
	cout<<"Testing Conjugate Gradient for Huber SVM Loss, press enter to continue...\n";
	#ifndef DEBUG
	cin.get();
	#endif
	cg(ll, Vector(m, 0), 1, 1e-4, 250);

	cout<<"*******************************************************************\n";
	cout<<"L-BFGS for Huber SVM Loss, press enter to continue...\n";
	#ifndef DEBUG
	cin.get();
	#endif
	lbfgsMin(ll, Vector(m, 0), 1, 1e-4, 250);
	// lbfgsMin(ss, Vector(m, 0), 1, 1e-4, 250);

	cout<<"*******************************************************************\n";
	cout<<"L-BFGS-OWL for Huber SVM Loss, press enter to continue...\n";
	#ifndef DEBUG
	cin.get();
	#endif
	lbfgsMinOwl(ll, Vector(m, 0), 1, 1e-4, 250);

	cout<<"*******************************************************************\n";
	cout<<"Stochastic Gradient Descent for Huber SVM Loss, press enter to continue...\n";
	#ifndef DEBUG
	cin.get();
	#endif
	sgd(ll, Vector(m, 0), n, 1e-4, 100, 1e-4, 250);

	cout<<"*******************************************************************\n";
	cout<<"Stochastic Gradient Descent with Decaying Learning Rate for Huber SVM Loss, press enter to continue...\n";
	#ifndef DEBUG
	cin.get();
	#endif
	sgdDecayingLearningRate(ll, Vector(m, 0), n, 0.5*1e-1, 200, 1e-4, 250, 0.6);

	cout<<"*******************************************************************\n";
	cout<<"Stochastic Gradient Descent with AdaGrad for Huber SVM Loss, press enter to continue...\n";
	#ifndef DEBUG
	cin.get();
	#endif
	sgdAdagrad(ll, Vector(m, 0), n, 1e-2, 200, 1e-4, 250);
}
