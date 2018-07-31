// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*
   Common Vector Operations like addition, subtraction and scalar multiplication.
   Implemented on the Vector.
   Author: Rishabh Iyer
 */
#include "Vector.h"
#include "MatrixOperations.h"
#include "VectorOperations.h"
#include <assert.h>

namespace jensen {
// z = x + y
Matrix matrixAddition(const Matrix& A, const Matrix& B)
{
	assert((A.numRows() == B.numRows()) && (A.numColumns() == B.numColumns()));
	Matrix C(A.numRows(), A.numColumns());
	for (int i = 0; i < A.numRows(); i++)
	{
		for (int j = 0; j < A.numColumns(); j++)
		{
			C(i, j) = A(i, j) + B(i, j);
		}
	}
	return C;
}

Matrix matrixSubtraction(const Matrix& A, const Matrix& B)
{
	assert((A.numRows() == B.numRows()) && (A.numColumns() == B.numColumns()));
	Matrix C(A.numRows(), A.numColumns());
	for (int i = 0; i < A.numRows(); i++)
	{
		for (int j = 0; j < A.numColumns(); j++)
		{
			C(i, j) = A(i, j) - B(i, j);
		}
	}
	return C;
}

// z = A*x
Vector leftMatrixVectorProduct(const Matrix& A, const Vector& x)
{
	assert(A.numColumns() == x.size());
	Vector z(x.size(), 0);
	for (int i = 0; i < x.size(); i++)
	{
		z[i] = x*A[i];
	}
	return z;
}

// z = x*A
Vector rightMatrixVectorProduct(const Matrix& A, const Vector& x)
{
	assert(A.numRows() == x.size());
	Vector z(A.numColumns(), 0);
	for (int i = 0; i < A.numColumns(); i++)
	{
		for (int j = 0; j < A.numRows(); j++) {
			z[i]+=A(j, i)*x[j];
		}
	}
	return z;
}

// C = A*B
Matrix matrixMatrixProduct(const Matrix& A, const Matrix& B)
{
	assert(A.numColumns() == B.numRows());
	int dsize = A.numColumns();
	Matrix C(A.numRows(), B.numColumns());
	for (int i = 0; i < A.numRows(); i++)
	{
		for (int j = 0; j < B.numColumns(); j++) {
			for (int k = 0; k < dsize; k++) {
				C(i, j) += A(i, k)*B(k, j);
			}
		}
	}
	return C;
}

const Matrix operator+(const Matrix& A, const Matrix &B){
	Matrix C = matrixAddition(A, B);
	return C;
}

const Matrix operator-(const Matrix& A, const Matrix &B){
	Matrix C = matrixSubtraction(A, B);
	return C;
}

const Vector operator*(const Matrix& A, const Vector &x){
	Vector z = leftMatrixVectorProduct(A, x);
	return z;
}

const Vector operator*(const Vector &x, const Matrix& A){
	Vector z = rightMatrixVectorProduct(A, x);
	return z;
}
const Matrix operator*(const Matrix& A, const Matrix& B){
	Matrix C = matrixMatrixProduct(A, B);
	return C;
}
const Matrix operator*(const Matrix& A, const double a){
	Matrix C(A.numRows(), A.numColumns());
	for (int i = 0; i < A.numRows(); i++)
	{
		for (int j = 0; j < A.numColumns(); j++)
		{
			C(i, j) = A(i, j) + a;
		}
	}
	return C;
}

// x == y
bool operator== (const Matrix& A, const Matrix& B){
	if ( (A.numRows() != B.numRows()) && (A.numColumns() == B.numColumns()) )
		return false;
	for (int i = 0; i < A.numRows(); i++) {
		if (A[i] != B[i])
			return false;
	}
	return true;
}

// x != y
bool operator!= (const Matrix& A, const Matrix& B){
	return !(A == B);
}

bool operator< (const Matrix& A, const Matrix& B){
	assert((A.numRows() == B.numRows()) && (A.numColumns() == B.numColumns()));
	for (int i = 0; i < A.numRows(); i++)
	{
		if (A[i] < B[i])
			continue;
		else
			return false;
	}
	return true;
}

bool operator<= (const Matrix& A, const Matrix& B){
	assert((A.numRows() == B.numRows()) && (A.numColumns() == B.numColumns()));
	for (int i = 0; i < A.numRows(); i++)
	{
		if (A[i] <= B[i])
			continue;
		else
			return false;
	}
	return true;
}

bool operator> (const Matrix& A, const Matrix& B){
	assert((A.numRows() == B.numRows()) && (A.numColumns() == B.numColumns()));
	for (int i = 0; i < A.numRows(); i++)
	{
		if (A[i] > B[i])
			continue;
		else
			return false;
	}
	return true;
}

bool operator>= (const Matrix& A, const Matrix& B){
	assert((A.numRows() == B.numRows()) && (A.numColumns() == B.numColumns()));
	for (int i = 0; i < A.numRows(); i++)
	{
		if (A[i] >= B[i])
			continue;
		else
			return false;
	}
	return true;
}
}
