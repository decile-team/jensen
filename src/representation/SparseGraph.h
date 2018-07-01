// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
#ifndef SMTK_SPARSE_GRAPH
#define SMTK_SPARSE_GRAPH
#include <vector>
namespace jensen {

struct SparseGraphItem{ //Stores the K nearest neighbor index and similarity for each item in the ground    set
    long int index; // index of the item
    int num_NN_items; // number of nearest neighbors for this item
    std::vector<int> NNIndex; // Indices for the nearest neighbors (generally sparse)
    std::vector<float> NNSim; // similarity for each neighbor.
};	
}

#endif
