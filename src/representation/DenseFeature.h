// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
#ifndef DENSE_FEATURE
#define DENSE_FEATURE

#include <vector>
namespace jensen {

struct DenseFeature { //Stores the feature vector for each item in the groundset
	long int index; // index of the item
	std::vector<double> featureVec; // score of the dense feature vector.
	int numFeatures;
};
// This is a dense feature representation. Each row has numFeatures number of items (i.e the size of featureVec should be equal to numFeatures)

}

#endif
