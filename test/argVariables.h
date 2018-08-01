// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
#ifndef ARGVARIABLES_H
#define ARGVARIABLES_H

extern char* trainFeatureFile = NULL;
extern char* trainLabelFile = NULL;
extern char* testFeatureFile = NULL;
extern char* testLabelFile = NULL;
extern int method = 5;
extern int nClasses = 2;
extern double lambda = 1;
extern char* outFile = NULL;
extern int maxIter = 1000;
extern int algtype = 0;
extern double tau = 1e-4;
extern double eps = 1e-2;
extern int verb = 0;
extern char* help = NULL;
extern bool startwith1 = false;
extern float percentTrain = 0.5;
extern int kfold = 1;

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
	Arg("percentTrain", Arg::Opt, percentTrain, "percent of data to use for training (default: 0.5)", Arg::SINGLE),
	Arg("kfold", Arg::Opt, kfold, "average cross-validation over k runs (k-fold cross validation)", Arg::SINGLE),
	Arg("verb", Arg::Opt, verb, "verbosity",Arg::SINGLE),
	Arg("help", Arg::Help, help, "Print this message"),
	Arg("startwith1", Arg::Opt, startwith1, "Whether the Label file starts with one or zero"),
	Arg()
};

string algs[] = {"L1 Logistic Regression", "L2 Logistic Regression",
	         "L1 Smooth SVM", "L2 Smooth SVM", "L2 Hinge SVM"};

#endif
