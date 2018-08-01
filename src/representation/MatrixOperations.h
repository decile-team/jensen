// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*
   Common Matrix Operations like addition, subtraction and matrix and vector multiplication etc.
   Implemented on the Vector and Matrix.
   Author: Rishabh Iyer
 */


#ifndef MATRIX_OPERATIONS_H
#define MATRIX_OPERATIONS_H


#include "Matrix.h"

namespace jensen {
Matrix matrixAddition(const Matrix& A, const Matrix& B);
Matrix matrixSubtraction(const Matrix& A, const Matrix& B);
Vector leftMatrixVectorProduct(const Matrix& A, const Vector& x);
Vector rightMatrixVectorProduct(const Matrix& A, const Vector& x);
Matrix matrixMatrixProduct(const Matrix& A, const Matrix& B);
Matrix matrixScalarProduct(const Matrix& A, const double a);

double norm(const Matrix& A, const int type = 2);

const Matrix operator+(const Matrix& A, const Matrix &B);
const Matrix operator-(const Matrix& A, const Matrix &B);
const Vector operator*(const Matrix& A, const Vector &x);
const Vector operator*(const Vector &x, const Matrix& A);
const Matrix operator*(const Matrix& A, const Matrix& B);
const Matrix operator*(const Matrix& A, const double a);

bool operator== (const Matrix& A, const Matrix& B);
bool operator!= (const Matrix& A, const Matrix& B);
bool operator< (const Matrix& A, const Matrix& B);
bool operator<= (const Matrix& A, const Matrix& B);
bool operator> (const Matrix& A, const Matrix& B);
bool operator>= (const Matrix& A, const Matrix& B);

}

#endif
