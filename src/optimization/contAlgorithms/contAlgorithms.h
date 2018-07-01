// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*
Anthor: Rishabh Iyer, John Halloran and Kai Wei
*/

#ifndef CONT_ALGORITHMS
#define CONT_ALGORITHMS

// Batch gradient descent
#include "gd.h"
#include "gdLineSearch.h"
#include "gdBarzilaiBorwein.h"
#include "gdNesterov.h"
#include "cg.h"
#include "lbfgsMin.h"
#include "lbfgsMinOwl.h"
#include "tron.h"
// Stochastic gradient descent
#include "sgd.h"
#include "sgdAdagrad.h"
#include "sgdDecayingLearningRate.h"
#include "sgdLineSearch.h"
#include "sgdRegularizedDualAveragingAdagrad.h"
#include "sgdRegularizedDualAveraging.h"
#include "sgdStochasticAverageGradient.h"
#include "specialized/L1LRPrimal.h"
#include "specialized/SVCDual.h"
#include "specialized/SVRDual.h"

#endif
