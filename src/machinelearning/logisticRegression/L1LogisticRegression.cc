// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*
 *	L2 Regularized Logistic Regression (Useful if you want to encourage sparsity in the classifier)
	Author: Rishabh Iyer

    algtype: type of algorithm: 
		0 (LBFGS-OWL)
	 	1 (LBFGS)
		2 (Gradient Descent)
		3 (Stochastic Gradient Descent, Dual Averaging)
		4 (Adaptive Gradient Descent, Dual Averaging)
		5 (Gradient Descent with Line Search), 
		6 (Gradient Descent with Barzelie Borwein step size), 
		7 (Nesterov's optimal method), 
		8 (Conjugate Gradient),  
		9 (Stochastic Gradient Descent with fixed step length)
		10 (Stochastic Gradient Descent with decaying step size)
		11 (Adaptive Gradient Algorithm (AdaGrad))

 *
*/

#include<iostream>
#include <assert.h>
#include<string.h>
#include<cmath>
using namespace std;

#include "L1LogisticRegression.h"
#include "../../optimization/contAlgorithms/contAlgorithms.h"
#include "../../optimization/contFunctions/L1LogisticLoss.h"
#include "../../representation/Set.h"

#define EPSILON 1e-6
namespace jensen {
	template <class Feature>
	L1LogisticRegression<Feature>::L1LogisticRegression(vector<Feature>& trainFeatures, Vector& y, int m, int n, int nClasses, 
		double lambda, int algtype, int maxIter, double eps, int miniBatch, int lbfgsMemory): Classifiers<Feature>(m, n), trainFeatures(trainFeatures), 
		y(y), nClasses(nClasses), lambda(lambda), algtype(algtype), maxIter(maxIter), eps(eps), miniBatch(miniBatch), lbfgsMemory(lbfgsMemory) {}

	template <class Feature>
	L1LogisticRegression<Feature>::L1LogisticRegression(const L1LogisticRegression<Feature>& c) : Classifiers<Feature>(c.m, c.n), nClasses(c.nClasses),
	trainFeatures(c.trainFeatures), y(c.y), lambda(c.lambda), algtype(c.algtype),
	maxIter(c.maxIter), eps(c.eps), miniBatch(c.miniBatch), lbfgsMemory(c.lbfgsMemory){}

	template <class Feature>
	L1LogisticRegression<Feature>::~L1LogisticRegression(){}
  		
	template <class Feature>
	void L1LogisticRegression<Feature>::train(){ // train L1 regularized logistic regression
		if (nClasses == 2){		
			trainOne(y, w);
		}
		else{
			vector<Set> yMapping = vector<Set>(nClasses); // a reverse mapping for indices of a particular label
			for (int i = 0; i < n; i++)
				yMapping[y[i]].insert(i);
			wMany = vector<Vector>(nClasses, Vector(m));
			for (int i = 0; i < nClasses; i++)
			{
				Vector yOne(n, -1);
				for (Set::iterator it = yMapping[i].begin(); it != yMapping[i].end(); it++)
					yOne[*it] = 1;
				trainOne(yOne, wMany[i]);
			}
		}
	}
	
	template <class Feature> 
	void L1LogisticRegression<Feature>::trainOne(Vector& yOne, Vector& wcurr){
		cout << trainFeatures.size() << " " << yOne.size() << "\n";
	L1LogisticLoss<Feature> ll(m, trainFeatures, yOne, lambda);
	L1LogisticLoss<Feature>l(m, trainFeatures, yOne, 0);
	
	if (algtype == 0){
		cout<<"*******************************************************************\n";
		cout<<"Training with LBFGS-OWL...\n";
		wcurr = lbfgsMinOwl(ll, Vector(m, 0), 1, 1e-4, maxIter, lbfgsMemory, eps);
	}		
	else if (algtype == 1){
		cout<<"*******************************************************************\n";
		cout<<"Training with LBFGS...\n";
		wcurr = lbfgsMin(ll, Vector(m, 0), 1, 1e-4, maxIter, lbfgsMemory, eps);
	}
	else if (algtype == 2){
		cout<<"*******************************************************************\n";
		cout<<"Training with Gradient Descent with fixed step size...\n";
		wcurr = gd(ll, Vector(m, 0), 1e-5, maxIter, eps);
	}
	else if (algtype == 3){
		cout<<"*******************************************************************\n";
		cout<<"Training with Stochastic Gradient Descent and Dual Averaging...\n";
		wcurr = sgdRegularizedDualAveraging(ll, l, Vector(m, 0), n, 1e-1, lambda, miniBatch, eps, maxIter, 0.5);
	}
	else if (algtype == 4){
		cout<<"*******************************************************************\n";
		cout<<"Training with Adaptive Gradient Descent and Dual Averaging...\n";
		wcurr = sgdRegularizedDualAveraging(ll, l, Vector(m, 0), n, 1e-1, lambda, miniBatch, eps, maxIter);
	}
	else if (algtype == 5){
		cout<<"*******************************************************************\n";
		cout<<"Training with Gradient Descent with Line Search...\n";
		wcurr = gdLineSearch(ll, Vector(m, 0), 1, 1e-5, maxIter, eps);
	}
	else if (algtype == 6){
		cout<<"*******************************************************************\n";
		cout<<"Training with Gradient Descent with Barzilia-Borwein Step Length\n";
		wcurr = gdBarzilaiBorwein(ll, Vector(m, 0), 1, 1e-5, maxIter, eps);
	}
	// gradientDescentBB(ss, Vector(m, 0), 1, 1e-4, 250);
	else if (algtype == 7){
		cout<<"*******************************************************************\n";
		cout<<"Training with Conjugate Gradient...\n";
		wcurr = cg(ll, Vector(m, 0), 1, 1e-5, maxIter, eps);
	}	
	else if (algtype == 8){
		cout<<"*******************************************************************\n";
		cout<<"Training with Nesterov's Method\n";
		wcurr = gdNesterov(ll, Vector(m, 0), 1, 1e-5, maxIter, eps);
	}
	else if (algtype == 9){
		cout<<"*******************************************************************\n";
		cout<<"Training with Stochastic Gradient Descent\n";
		wcurr = sgd(ll, Vector(m, 0), n, 1e-4, miniBatch, eps, maxIter);
	}
	else if (algtype == 10){
		cout<<"*******************************************************************\n";
		cout<<"Training with Stochastic Gradient Descent with decaying learning rate\n";
		wcurr = sgdDecayingLearningRate(ll, Vector(m, 0), n, 0.5*1e-1, miniBatch, eps, maxIter);
	}
	else if (algtype == 11){
		cout<<"*******************************************************************\n";
		cout<<"Training with Adaptive Gradient Algorithm\n";
		wcurr = sgdAdagrad(ll, Vector(m, 0), n, 1e-2, miniBatch, eps, maxIter);
	}
}

// save the model
template <class Feature>
int L1LogisticRegression<Feature>::saveModel(char* model){
	FILE *fp = fopen(model,"w");
	if(fp==NULL) return -1;

	fprintf(fp, "algtype %s\n", algtype);
	fprintf(fp, "nClasses %d\n", nClasses);
	fprintf(fp, "nFeatures %d\n", m);
	fprintf(fp, "n %d\n", n);
	fprintf(fp, "w\n");
	if (nClasses == 2){
		for(int i=0; i<w.size(); i++)
		{
			fprintf(fp, "%.16g ", w[i]);
		}
		fprintf(fp, "\n");
	}
	else{
		for (int i = 0; i < nClasses; i++){
			for (int j = 0; j < m; j++){
				fprintf(fp, "%.16g ", wMany[i][j]);
			}
		}
		fprintf(fp, "\n");	
	}

	if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
	else return 0;
} 

//brief: Load an already saved model of the classifier
template <class Feature>
int L1LogisticRegression<Feature>::loadModel(char* model){
	FILE *fp = fopen(model,"r");
	if(fp==NULL) return -1;

	char cmd[81];
	while(1)
	{
		fscanf(fp,"%80s",cmd);
		if(strcmp(cmd,"algtype")==0)
			fscanf(fp,"%d",&algtype);
		else if(strcmp(cmd,"nClasses")==0)
			fscanf(fp,"%d",&nClasses);
		else if(strcmp(cmd,"nFeatures")==0)
			fscanf(fp,"%d",&m);
		else if(strcmp(cmd,"n")==0)
			fscanf(fp,"%d",&n);
		else if(strcmp(cmd,"w")==0)
		{
			break;
		}
		else
		{
			fprintf(stderr,"unknown text in model file: [%s]\n",cmd);
			return -1;
		}
	}

	if(nClasses==2){
		w = Vector(m, 0);
		for (int i = 0; i < m; i++)
			fscanf(fp, "%f ", w[i]);
	}
	else{
		wMany = vector<Vector>(nClasses, Vector());
		for (int i = 0; i < nClasses; i++){
			for (int j = 0; j < m; j++){
				fscanf(fp, "%f ", wMany[i][j]);
			}
		}
	}
	if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
	return 1;
}


template <class Feature>
double L1LogisticRegression<Feature>::predict(const Feature& testFeature, double& val){
	// the assumption here is that train and test datasets have the same number of features
	if (nClasses == 2){
		val = featureProductCheck(w, testFeature);
		double argval = 0;
		if (val > 0)
			argval = 1;
		else
			argval = -1;
		return argval;
	}
	else{
		val = -1e30;
		double argval = 0;
		for (int j = 0; j < nClasses; j++){
			if (featureProductCheck(wMany[j], testFeature) > val){
				val = featureProductCheck(wMany[j], testFeature);
				argval = j;
			}
		}
		return argval;
	}	
}

template <class Feature>
double L1LogisticRegression<Feature>::predict(const Feature& testFeature){
	// the assumption here is that train and test datasets have the same number of features
	if (nClasses == 2){
		double val = featureProductCheck(w, testFeature);
		double argval = 0;
		if (val > 0)
			argval = 1;
		else
			argval = -1;
		return argval;
	}
	else{
		double val = -1e30;
		double argval = 0;
		for (int j = 0; j < nClasses; j++){
			if (featureProductCheck(wMany[j], testFeature) > val){
				val = featureProductCheck(wMany[j], testFeature);
				argval = j;
			}
		}
		return argval;
	}
}	

// prob is a vector. The assumption is that prob[0] corresponds to -1 and prob[1] corresponds to +1 in binary 
// classification.
template <class Feature>
void L1LogisticRegression<Feature>::predictProbability(const Feature& testFeature, Vector& prob){
	// the assumption here is that train and test datasets have the same number of features
	prob = Vector(nClasses, 0);
	double val;
	if (nClasses == 2){
		val = featureProductCheck(w, testFeature);
		prob[1] = 1/(1+exp(-val));
		prob[0] = 1 - prob[1];
	}
	else{
		double sum = 0;
		for (int j = 0; j < nClasses; j++){
			val = featureProductCheck(wMany[j], testFeature);
			prob[j] = 1/(1 + exp(-val));
			sum += prob[j];
		}
		for (int j = 0; j < nClasses; j++)
			prob[j] = prob[j]/sum;
	}
	return;
}		 	
	
  template class L1LogisticRegression<SparseFeature>;
  template class L1LogisticRegression<DenseFeature>;
	 	
		 	
}
