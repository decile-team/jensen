// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*
        Jensen: A Convex Optimization And Machine Learning ToolKit
 *	A typedef defining vectors (since the vector class in STL is quite versatile, we just use it in this toolkit)
        Author: Rishabh Iyer
 *
 */

#ifndef VECTOR_H
#define VECTOR_H

#include <stdio.h>
#include <stdlib.h>
#include <vector>

namespace jensen {

typedef std::vector<double> Vector;
}
#endif
