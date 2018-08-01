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
char* featureFile = NULL;
char* labelFile = NULL;
int algtype = 0;
double lambda = 1;
char* outFile = NULL;
int maxIter = 250;
double tau = 1e-4;
double eps = 1e-2;
int verb = 0;
char* help = NULL;

Arg Arg::Args[]={
	Arg("featureFile", Arg::Req, featureFile, "the input data file",Arg::SINGLE),
	Arg("labelFile", Arg::Req, labelFile, "the input label file",Arg::SINGLE),
	Arg("Regularization", Arg::Opt, lambda, "Regularization parameter (default 1)", Arg::SINGLE),
	Arg("MaxIter", Arg::Opt, maxIter, "Maximum number of iterations (default 250)", Arg::SINGLE),
	Arg("Epsilon", Arg::Opt, eps, "Epsilon for convergence (default: 1e-2)", Arg::SINGLE),
	Arg("algtype", Arg::Opt, algtype, "type of algorithm: 0(GD), 1(GD-LS), 2(GD-BB), 3(Nesterov), 4(CG), 5(LBFGS), 6(TRON)",Arg::SINGLE),
	Arg("model", Arg::Opt, outFile, "the output model",Arg::SINGLE),
	Arg("verb", Arg::Opt, verb, "verbosity",Arg::SINGLE),
	Arg("help", Arg::Help, help, "Print this message"),
	Arg()
};

int main(int argc, char** argv){
	bool parse_was_ok = Arg::parse(argc,(char**)argv);
	if(!parse_was_ok) {
		Arg::usage(); exit(-1);
	}

	int n; // number of data items
	int m; // numFeatures
	vector<struct SparseFeature> features = readFeatureVectorSparse(featureFile, n, m);
	Vector y = readVector(labelFile, n);
	cout<<y.size()<<"\n";
	// LogisticLoss l(m, features, y);
	// L2 r(m);
	// SumContinuousFunctions ll(l, r, 0.5);
	L2LogisticLoss<SparseFeature> ll(m, features, y, lambda);
	Vector x(m, 1);
	double f;
	Vector g;
	if (algtype == 0) {
		cout<<"*******************************************************************\n";
		cout<<"Testing Gradient Descent with Logistic Loss, press enter to continue...\n";
		x = gd(ll, Vector(m, 0), 1e-5, maxIter, eps);
	}
	// cout<<"*******************************************************************\n"
	// cout<<"Testing Gradient Descent with Logistic Loss\n";
	// gradientDescent(ss, Vector(m, 0), 1e-8, 250);
	else if (algtype == 1) {
		cout<<"*******************************************************************\n";
		cout<<"Testing Gradient Descent with Line Search for Logistic Loss, press enter to continue...\n";
		gdLineSearch(ll, Vector(m, 0), 1, tau, maxIter, eps);
	}
	else if (algtype == 2) {
		cout<<"*******************************************************************\n";
		cout<<"Testing Gradient Descent with Barzilia-Borwein Step Length for Logistic Loss, press enter to continue...\n";
		gdBarzilaiBorwein(ll, Vector(m, 0), 1, tau, maxIter, eps);
	}
	// gradientDescentBB(ss, Vector(m, 0), 1, 1e-4, 250);
	else if (algtype == 3) {
		cout<<"*******************************************************************\n";
		cout<<"Testing Nesterov's Method for Logistic Loss, press enter to continue...\n";
		gdNesterov(ll, Vector(m, 0), 1, tau, maxIter, eps);
	}
	else if (algtype == 4) {
		cout<<"*******************************************************************\n";
		cout<<"Testing Conjugate Gradient for Logistic Loss, press enter to continue...\n";
		cg(ll, Vector(m, 0), 1, tau, maxIter, eps);
	}
	else if (algtype == 5) {
		cout<<"*******************************************************************\n";
		cout<<"L-BFGS for Logistic Loss, press enter to continue...\n";
		Vector w = lbfgsMin(ll, Vector(m, 0), 1, 1e-4, maxIter, 100, eps);
		// lbfgsMin(ss, Vector(m, 0), 1, 1e-4, 250);
	}
	else if (algtype == 6) {
		cout<<"*******************************************************************\n";
		cout<<"Trust Region Newton Method for Logistic Loss, press enter to continue...\n";
		tron(ll, Vector(m, 0), maxIter, eps);
	}
	// lbfgsMin(ss, Vector(m, 0), 1, 1e-4, 250);

	/*cout<<"*******************************************************************\n";
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
	   // sgdLineSearch(ll, Vector(m, 0), n, 1e-2, 200, 1e-4, 250);*/

}
