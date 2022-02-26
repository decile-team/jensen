// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*
        Author: Rishabh Iyer
 *
 */

#include <iostream>
#include <cstdlib>
#include <string>
#include "../src/jensen.h"
using namespace jensen;
using namespace std;

char* trainFeatureFile = NULL;
char* trainLabelFile = NULL;
char* testFeatureFile = NULL;
char* testLabelFile = NULL;
int method = 5;
int nClasses = 2;
double lambda = 1;
char* outFile = NULL;
int maxIter = 1000;
int algtype = 0;
int reg_type = 0;
double tau = 1e-4;
double eps = 1e-2;
int verb = 0;
char* help = NULL;
bool startwith1 = false;

#define L1LR 1
#define L2LR 2
#define L1SSVM 3
#define L2SSVM 4
#define L2HSVM 5

Arg Arg::Args[]={
	Arg("trainFeatureFile", Arg::Req, trainFeatureFile, "the input training feature file",Arg::SINGLE),
	Arg("trainLabelFile", Arg::Req, trainLabelFile, "the input training label file",Arg::SINGLE),
	Arg("testFeatureFile", Arg::Req, testFeatureFile, "the input test feature file",Arg::SINGLE),
	Arg("testLabelFile", Arg::Req, testLabelFile, "the input test label file",Arg::SINGLE),
	Arg("nClasses", Arg::Opt, nClasses, "The number of classes", Arg::SINGLE),
	Arg("method", Arg::Opt, method, "Training method: 1(L1LR), 2(L2LR), 3(L1SSVM), 4(L2SSVM), 5(L2HSVM)", Arg::SINGLE),
	Arg("reg", Arg::Opt, lambda, "Regularization parameter (default 1)", Arg::SINGLE),
	Arg("maxIter", Arg::Opt, maxIter, "Maximum number of iterations (default 250)", Arg::SINGLE),
	Arg("epsilon", Arg::Opt, eps, "epsilon for convergence (default: 1e-2)", Arg::SINGLE),
	Arg("algtype", Arg::Opt, algtype, "type of algorithm for training the corresponding method",Arg::SINGLE),
	Arg("model", Arg::Opt, outFile, "saving the training model",Arg::SINGLE),
	Arg("verb", Arg::Opt, verb, "verbosity",Arg::SINGLE),
	Arg("help", Arg::Help, help, "Print this message"),
	Arg("startwith1", Arg::Opt, startwith1, "Whether the Label file starts with one or zero"),
	Arg()
};

string algs[] = {"L1 Logistic Regression", "L2 Logistic Regression", "L1 Smooth SVM",
	         "L2 Smooth SVM", "L2 Hinge SVM"};

template <class Feature>
double predictAccuracy(Classifiers<Feature>* c, vector<Feature>& testFeatures, Vector& ytest){
	assert(testFeatures.size() == ytest.size());
	double accuracy = 0;
	for (int i = 0; i < testFeatures.size(); i++) {
		if (c->predict(testFeatures[i]) == ytest[i])
			accuracy++;
	}
	return accuracy;
}

int main(int argc, char** argv){
	bool parse_was_ok = Arg::parse(argc,(char**)argv);
	if(!parse_was_ok) {
		Arg::usage(); exit(-1);
	}

	int ntrain; // number of data items in the training set
	int mtrain; // numFeatures of the training data
	int ntest; // number of data items in the test set
	int mtest; // numFeatures of the test data
	vector<struct SparseFeature> trainFeatures = readFeatureVectorSparse(trainFeatureFile, ntrain, mtrain);
	Vector ytrain = readVector(trainLabelFile, ntrain);
	vector<struct SparseFeature> testFeatures = readFeatureVectorSparse(testFeatureFile, ntest, mtest);
	Vector ytest = readVector(testLabelFile, ntest);
	if (startwith1)
	{
		ytrain = ytrain - 1;
		ytest = ytest - 1;
	}
	cout << "Done reading the file, the size of the training set is " << ytrain.size() << " and the size of the test set is " <<ytest.size() << endl;
	if ((method < 0) || (method > 5)) {
		cout << "Invalid method.\n";
		return -1;
	}
	cout << "Now training a " << algs[method-1] << " classifier.\n";
	double accuracy = 0;
	if (method == L1LR)
	{
		reg_type = 0;
		Classifiers<SparseFeature>* c = new LogisticRegression<SparseFeature>(trainFeatures, ytrain, mtrain, ntrain, nClasses,
		                                                                        lambda, algtype, reg_type, maxIter, eps);
		c->train();
		cout << "Done with Training ... now testing\n";
		accuracy = predictAccuracy(c, testFeatures, ytest);
		delete c;
	}
	else if (method == L2LR)
	{
		reg_type = 1;
		Classifiers<SparseFeature>* c = new LogisticRegression<SparseFeature>(trainFeatures, ytrain, mtrain, ntrain, nClasses,
		                                                                        lambda, algtype, reg_type, maxIter, eps);
		c->train();
		cout << "Done with Training ... now testing\n";
		accuracy = predictAccuracy(c, testFeatures, ytest);
		delete c;

	}
	else if (method == L1SSVM)
	{
		Classifiers<SparseFeature>* c = new L1SmoothSVM<SparseFeature>(trainFeatures, ytrain, mtrain, ntrain, nClasses,
		                                                               lambda, algtype, maxIter, eps);
		c->train();
		cout << "Done with Training ... now testing\n";
		accuracy = predictAccuracy(c, testFeatures, ytest);
		delete c;
	}
	else if (method == L2SSVM)
	{
		Classifiers<SparseFeature>* c = new L2SmoothSVM<SparseFeature>(trainFeatures, ytrain, mtrain, ntrain, nClasses,
		                                                               lambda, algtype, maxIter, eps);
		c->train();
		cout << "Done with Training ... now testing\n";
		accuracy = predictAccuracy(c, testFeatures, ytest);
		delete c;
	}
	else if (method == L2HSVM)
	{
		Classifiers<SparseFeature>* c = new L2HingeSVM<SparseFeature>(trainFeatures, ytrain, mtrain, ntrain, nClasses,
		                                                              lambda, algtype, maxIter, eps);
		c->train();
		cout << "Done with Training ... now testing\n";
		accuracy = predictAccuracy(c, testFeatures, ytest);
		delete c;
	}
	else
	{
		cout << "Invalid mode\n";
		return -1;
	}
	double accuracy_percentage = accuracy/ytest.size();
	cout << "The acuracy of the classifier is "<< accuracy_percentage << "("<< accuracy << "/"<< ytest.size()
	     << ")" << "\n";
}
