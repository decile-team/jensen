# Jensen
Jensen: A toolkit with API support for Convex Optimization and Machine Learning
For further documentation, please see https://arxiv.org/abs/1807.06574

## License
Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
Licensed under the Open Software License version 3.0
See COPYING or http://opensource.org/licenses/OSL-3.0

## Contributors:
- Rishabh Iyer
- John Halloran
- Kai Wei

## Features Supported

1) Convex Function API
 - Base class for convex optimization
 - `L1LogistocLoss` and `L2LogistocLoss`, 
 - `L1SmoothSVMLoss` and `L2SmoothSVMLoss`, 
 - `L1HingeSVMLoss` and `L2HingeSVMLoss`, 
 - `L1ProbitLossLoss` and `L2ProbitLoss`, 
 - `L1HuberSVMLoss` and `L2HuberSVMLoss`, 
 - `L1SmoothSVRLoss` and `L2SmoothSVRLoss`, 
 - `L1HingeSVMLoss` and `L2HingeSVMLoss`

2) Convex Optimization Algorithms API
 - `Trust Region Newton` (TRON)
 - `LBFGS Algorithm`
 - `LBFGS OWL` (L1 regularization)
 - `Conjugate Gradient Descent`
 - `Dual Coordinate Descent for SVMs` (SVCDual)
 - `Gradient Descent`
 - `Gradient Descent with Line Search`
 - `Gradient Descent with Nesterov's algorithm`
 - `Gradient Descent with Barzilai-Borwein step size`
 - `Stochastic Gradient Descent`
 - `Stochastic Gradient Descent with AdaGrad`
 - `Stochastic Gradient Descent with Dual Averaging`
 - `Stochastic Gradient Descent with Decaying Learning Rate`
  
3) ML Classification API 
 - `L1 Logistic Regression`, 
 - `L2 Logistic Regression`
 - `L1 Smooth SVM`
 - `L2 Smooth SVM`
 - `L2 Smooth SVM`
 
4) ML Regression API 
 - `L1 Linear Regression`
 - `L2 Linear Regression`
 - `L1 Smooth SVRs`
 - `L2 Smooth SVRs`
 - `L2 Hinge SVRs`
 
## Install and Build
1) Install CMake
2) Go to the main directory of jensen
3) mkdir build
4) cd build/
5) cmake ..
6) make

Once you run make, it should automatically build the entire library. Once the library is built, please try out the example codes in the build directory.

## Testing the Convex Optimization Algorithms
To test the optimization algorithms please run the test executables:
./TestL1LogisticLoss
./TestL2LogisticLoss
./TestL1SmoothSVMLoss
./TestL2LeastSquaresLoss etc.

You can also play around with the examples for testing classification and regression models. You can try them out as:
./ClassificationExample -trainFeatureFile ../data/heart_scale.feat -trainLabelFile ../data/heart_scale.label -testFeatureFile ../data/heart_scale.feat -testLabelFile ../data/heart_scale.label 
Optionally you can also play around with the method (L1LR, L2LR etc.), the algtype (LBFGS, TRON etc.), the regularization and so on.
