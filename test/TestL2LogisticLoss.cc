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
	char* featureFile = "../data/20newsgroup.feat";
	char* labelFile = "../data/20newsgroup.label";
	int n; // number of data items
	int m; // numFeatures
	bool checkOld = false;
	vector<struct SparseFeature> features = readFeatureVectorSparse(featureFile, n, m);
	Vector y = readVector(labelFile, n) - 1;
	int numEpochs = 50;
	L2LogisticLoss<SparseFeature> ll(m, features, y, 1);

	double f;
	Vector x0(m, 0), g, x;
	cout<<"*******************************************************************\n";
	cout<<"Testing Gradient Descent with Logistic Loss, press enter to continue...\n";
	#ifndef DEBUG
	cin.get();
	#endif
	x = gd(ll, x0, 1e-5, numEpochs);

	cout<<"*******************************************************************\n";
	cout<<"Testing Gradient Descent with Line Search for Logistic Loss, press enter to continue...\n";
	#ifndef DEBUG
	cin.get();
	#endif
	x = gdLineSearch(ll, x0, 1, 1e-4, numEpochs);

	cout<<"*******************************************************************\n";
	cout<<"Testing Gradient Descent with Barzilia-Borwein Step Length for Logistic Loss, press enter to continue...\n";
	#ifndef DEBUG
	cin.get();
	#endif
	x = gdBarzilaiBorwein(ll, x0, 1, 1e-4, numEpochs);

	cout<<"*******************************************************************\n";
	cout<<"Testing Nesterov's Method for Logistic Loss, press enter to continue...\n";
	#ifndef DEBUG
	cin.get();
	#endif
	x = gdNesterov(ll, x0, 1, 1e-4, numEpochs);

	cout<<"*******************************************************************\n";
	cout<<"Testing Conjugate Gradient for Logistic Loss, press enter to continue...\n";
	#ifndef DEBUG
	cin.get();
	#endif
	x = cg(ll, x0, 1, 1e-4, numEpochs);

	cout<<"*******************************************************************\n";
	cout<<"L-BFGS for Logistic Loss, press enter to continue...\n";
	#ifndef DEBUG
	cin.get();
	#endif
	x = lbfgsMin(ll, x0, 1, 1e-4, numEpochs);

	cout<<"*******************************************************************\n";
	cout<<"Trust Region Newton Method for Logistic Loss, press enter to continue...\n";
	cout<<"Note: This method does not check norm(gradient) < tol to determine convergence\n";
	#ifndef DEBUG
	cin.get();
	#endif
	x = tron(ll, x0, numEpochs);

	cout<<"*******************************************************************\n";
	cout<<"Stochastic Gradient Descent for Logistic Loss, press enter to continue...\n";
	#ifndef DEBUG
	cin.get();
	#endif
	x = sgd(ll, x0, n, 1e-4, 100, 1e-4, numEpochs);

	cout<<"*******************************************************************\n";
	cout<<"Stochastic Gradient Descent with Decaying Learning Rate for Logistic Loss, press enter to continue...\n";
	#ifndef DEBUG
	cin.get();
	#endif
	x = sgdDecayingLearningRate(ll, x0, n, 0.5*1e-1, 200, 1e-4, numEpochs, 0.6);

	cout<<"*******************************************************************\n";
	cout<<"Stochastic Gradient Descent with AdaGrad for Logistic Loss, press enter to continue...\n";
	#ifndef DEBUG
	cin.get();
	#endif
	x = sgdAdagrad(ll, x0, n, 1e-2, 200, 1e-4, numEpochs);

	cout<<"*******************************************************************\n";
	cout<<"SGD with Stochastic Average Gradient for Logistic Loss, press enter to continue...\n";
	#ifndef DEBUG
	cin.get();
	#endif
	x = sgdStochasticAverageGradient(ll, x0, n, 2, 1, 200, 1e-4, numEpochs, 1.0);
}
