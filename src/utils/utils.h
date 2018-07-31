// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*
    A simple utils library. Implementations of min, max ...
 */

#ifndef Jensen_UTILS_H
#define Jensen_UTILS_H

namespace jensen {
inline double min(double a, double b)
{
	if (a < b) {
		return a;
	}
	else{
		return b;
	}
}

inline double max(double a, double b)
{
	if (a > b) {
		return a;
	}
	else{
		return b;
	}
}

inline int sign(double x){
	return (0 < x) - (x < 0);
}
}

#endif
