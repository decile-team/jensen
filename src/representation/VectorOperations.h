// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*
Common Vector Operations like addition, subtraction and scalar multiplication.
Implemented on the Vector.
Author: Rishabh Iyer
*/


#ifndef VECTOR_OPERATIONS_H
#define VECTOR_OPERATIONS_H

#include "Vector.h"
#include "Matrix.h"
#include "SparseFeature.h"
#include "DenseFeature.h"
#include <iostream>
namespace jensen {
	double sum(const Vector& x);
	void vectorAddition(const Vector& x, const Vector& y, Vector& z);	
	void vectorFeatureAddition(const Vector& x, const SparseFeature& f, Vector& z);
	void vectorFeatureAddition(const Vector& x, const DenseFeature& f, Vector& z);
	void vectorScalarAddition(const Vector& x, const double a, Vector& z);	
	void vectorSubtraction(const Vector& x, const Vector& y, Vector& z);	
	void vectorFeatureSubtraction(const Vector& x, const SparseFeature& f, Vector& z);
	void vectorFeatureSubtraction(const Vector& x, const DenseFeature& f, Vector& z);
	void vectorScalarSubtraction(const Vector& x, const double a, Vector& z);
	void elementMultiplication(const Vector& x, const Vector& y, Vector& z);
	Vector elementMultiplication(const Vector& x, const Vector& y);
	Vector elementPower(const Vector& x, const double a);
	void elementPower(const Vector& x, const double a, Vector& z);
	void scalarMultiplication(const Vector& x, const double a, Vector& z);
	void scalarMultiplication(const SparseFeature& f, const double a, SparseFeature& g);
	void scalarMultiplication(const DenseFeature& f, const double a, DenseFeature& g);
	double innerProduct(const Vector& x, const Vector& y);	
	double featureProduct(const Vector& x, const SparseFeature& f);
	double featureProduct(const Vector& x, const DenseFeature& f);
	double featureProductCheck(const Vector& x, const SparseFeature& f);
	double featureProductCheck(const Vector& x, const DenseFeature& f);
	void outerProduct(const Vector& x, const Vector& y, Matrix& m);	
	double norm(const Vector& x, const int type = 2);	// default is l_2 norm
	void print(const Vector& x);
	Vector abs(const Vector& x);
	Vector sign(const Vector& x);
	void abs(const Vector& x, Vector& absx);
	void sign(const Vector& x, Vector& sx);
	void multiplyAccumulate(Vector& z, const Vector& x, const double alpha, const Vector& g);
	void multiplyAccumulate(Vector& x, const double alpha, const Vector& g);
	
	template <size_t N> Vector assign(double (&array)[N]);
	
	const Vector operator+(const Vector& x, const Vector &y);
	const Vector operator+(const Vector& x, const SparseFeature& f);
	const Vector operator+(const Vector& x, const DenseFeature& f);
	const Vector operator+(const Vector& x, const double a);
	const Vector operator-(const Vector& x, const Vector &y);
	const Vector operator-(const Vector& x, const SparseFeature& f);
	const Vector operator-(const Vector& x, const DenseFeature& f);
	const Vector operator-(const Vector& x, const double a);	
	const double operator*(const Vector& x, const Vector &y);
	const double operator*(const Vector& x, const SparseFeature &f);
	const double operator*(const Vector& x, const DenseFeature &f);
	const Vector operator*(const Vector& x, const double a);
	const Vector operator*(const double a, const Vector& x);
	
	const SparseFeature operator*(const SparseFeature& f, const double a);
	const SparseFeature operator*(const double a, const SparseFeature& f);
	const DenseFeature operator*(const DenseFeature& f, const double a);
	const DenseFeature operator*(const double a, const DenseFeature& f);
	
	Vector& operator+=(Vector& x, const Vector &y);
	Vector& operator+=(Vector& x, const SparseFeature &f);
	Vector& operator+=(Vector& x, const DenseFeature &f);
	Vector& operator+=(Vector& x, const double a);
	Vector& operator-=(Vector& x, const Vector &y);
	Vector& operator-=(Vector& x, const SparseFeature &f);
	Vector& operator-=(Vector& x, const DenseFeature &f);
	Vector& operator-=(Vector& x, const double a);
	Vector& operator*=(Vector& x, const double a);
    bool operator== (const Vector& x, const Vector& y);
	bool operator!= (const Vector& x, const Vector& y);
	bool operator< (const Vector& x, const Vector& y);
	bool operator<= (const Vector& x, const Vector& y);
	bool operator> (const Vector& x, const Vector& y);
	bool operator>= (const Vector& x, const Vector& y);
	
	std::ostream& operator<<(std::ostream& os, const Vector& x);
}

#endif
