// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*
        Jensen: A Convex Optimization And Machine Learning ToolKit
 *	Matrix class
        Author: Rishabh Iyer
 *
 */

#include "Vector.h"
#include "Matrix.h"
#include <assert.h>
#include <iostream>
namespace jensen {

Matrix::Matrix() : m(0), n(0){
}

Matrix::Matrix(int m, int n) : m(m), n(n){
	matrix.reserve(n);
	for (int i = 0; i < m; i++) {
		Vector v(n, 0);
		matrix.push_back(v);
	}
}

Matrix::Matrix(int m, int n, int val) : m(m), n(n){
	matrix.reserve(n);
	for (int i = 0; i < m; i++) {
		Vector v(n, val);
		matrix.push_back(v);
	}
}

Matrix::Matrix(int m, int n, bool) : m(m), n(n){        // Identity Matrix constructor
	matrix.reserve(n);
	assert(m == n);         // works only for square matrices
	for (int i = 0; i < m; i++) {
		Vector v(n, 0);
		v[i] = 1;
		matrix.push_back(v);
	}
}

Matrix::Matrix(const Matrix& M) : m(M.m), n(M.n){
	matrix.reserve(n);
	Vector v(n, 0);
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			v[j] = M(i, j);
		}
		matrix.push_back(v);
	}
}

double& Matrix::operator()(const int i, const int j){         // Access to element
	return matrix[i][j];
}

const double& Matrix::operator()(const int i, const int j) const {        // Const Access to element
	return matrix[i][j];
}

Vector& Matrix::operator[](const int i){         // Row access
	return matrix[i];
}

const Vector& Matrix::operator[](const int i) const {        // Row access
	return matrix[i];
}

Vector Matrix::operator()(const int i) const {        // Column Access (this is one is value only and const)
	Vector v(n, 0);
	for (int j = 0; j < n; j++)
	{
		v[j] = matrix[j][i];
	}
	return v;
}

void Matrix::push_back(const Vector& v){         // Add a row
	if (m == 0) {
		matrix.push_back(v);
		m++;
		n = v.size();
	}
	else{
		assert(v.size() == n);
		matrix.push_back(v);
		m++;
	}
}

void Matrix::remove(int i){
	assert((i >= 0) && (i < m));
	matrix.erase(matrix.begin()+i);
	m--;
}

int Matrix::numRows() const {
	return m;
}

int Matrix::numColumns() const {
	return n;
}

int Matrix::size() const {
	return m*n;
}

}
