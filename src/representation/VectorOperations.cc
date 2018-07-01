// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*
Common Vector Operations like addition, subtraction and scalar multiplication.
Implemented on the Vector.
Author: Rishabh Iyer
*/

#include "VectorOperations.h"
#include <cmath>
#include <iostream>
#include <assert.h>
#include "../utils/utils.h"

namespace jensen {
	// z = x + y. 
	void vectorAddition(const Vector& x, const Vector& y, Vector& z)
	{
		// assert(n == y.size());
		z = Vector(x.size(), 0);
		for (int i = 0; i < x.size(); i++)
		{
			z[i] = x[i] + y[i];
		}
		return;
	}

	void vectorFeatureAddition(const Vector& x, const SparseFeature& f, Vector& z)
	{
		// assert(x.size() == f.numFeatures);
		z = Vector(x);
		for (int i = 0; i < f.featureIndex.size(); i++)
		{
			int j = f.featureIndex[i];
			z[j] += f.featureVec[i];
		}
		return;
	}
	
	void vectorFeatureAddition(const Vector& x, const DenseFeature& f, Vector& z)
	{
		// assert(x.size() == f.featureVec.size());
		z = Vector(x.size(), 0);
		for (int i = 0; i < x.size(); i++)
		{
			z[i] = x[i] + f.featureVec[i];
		}
		return;
	}
	
	void vectorScalarAddition(const Vector& x, const double a, Vector& z)
	{
		z = Vector(x.size(), 0);
		for (int i = 0; i < x.size(); i++)
		{
			z[i] = x[i] + a;
		}
		return;
	}
	
	// z = x - y
	void vectorSubtraction(const Vector& x, const Vector& y, Vector& z)
	{
		// assert(x.size() == y.size());
		z = Vector(x.size(), 0);
		for (int i = 0; i < x.size(); i++)
		{
			z[i] = x[i] - y[i];
		}
		return;
	}	
	
	void vectorFeatureSubtraction(const Vector& x, const SparseFeature& f, Vector& z)
	{
		// assert(x.size() == f.numFeatures);
		z = Vector(x);
		for (int i = 0; i < f.featureIndex.size(); i++)
		{
			int j = f.featureIndex[i];
			z[j] -= f.featureVec[i];
		}
		return;
	}
	
	void vectorFeatureSubtraction(const Vector& x, const DenseFeature& f, Vector& z)
	{
		// assert(x.size() == f.featureVec.size());
		z = Vector(x.size(), 0);
		for (int i = 0; i < x.size(); i++)
		{
			z[i] = x[i] - f.featureVec[i];
		}
		return;
	}
	
	void vectorScalarSubtraction(const Vector& x, const double a, Vector& z)
	{
		z = Vector(x.size(), 0);
		for (int i = 0; i < x.size(); i++)
		{
			z[i] = x[i] - a;
		}
		return;
	}	

	// z = x.*y, x and y vectors
	void elementMultiplication(const Vector& x, const Vector& y, Vector& z)
	{
		// assert(x.size() == y.size());
		z = Vector(x.size(), 0);
		for (int i = 0; i < x.size(); i++)
		{
			z[i] = x[i]*y[i];
		}
		return;
	}
	
	Vector elementMultiplication(const Vector& x, const Vector& y)
	{
		Vector z;
		elementMultiplication(x, y, z);
		return z;
	}
	// z = x.^a, x and y vectors
	void elementPower(const Vector& x, const double a, Vector& z)
	{
		z = Vector(x.size(), 0);
		for (int i = 0; i < x.size(); i++)
		{
		  z[i] = pow(x[i], a);
		}
		return;
	}

	Vector elementPower(const Vector& x, const double a)
	{
		Vector z;
		elementPower(x, a, z);
		return z;
	}
	
	// z = a*x (a scalar)
	void scalarMultiplication(const Vector& x, const double a, Vector& z)
	{
		z = Vector(x.size(), 0);
		for (int i = 0; i < x.size(); i++)
		{
			z[i] = a*x[i];
		}
		return;
	}

	void scalarMultiplication(const SparseFeature& f, const double a, SparseFeature& g)
	{
		g = f;
		for (int i = 0; i < f.featureIndex.size(); i++)
		{
			g.featureVec[i] = a*f.featureVec[i];
		}
		return;
	}
	
	void scalarMultiplication(const DenseFeature& f, const double a, DenseFeature& g)
	{
		g = f;
		for (int i = 0; i < f.featureVec.size(); i++)
		{
			g.featureVec[i] = a*f.featureVec[i];
		}
		return;
	}
	
	double innerProduct(const Vector& x, const Vector& y)
	{
		// assert(x.size() == y.size());
		double d = 0;
		for (int i = 0; i < x.size(); i++)
		{
			d += x[i]*y[i];
		}
		return d;
	}

	double featureProduct(const Vector& x, const SparseFeature& f)
	{
		// assert(x.size() == f.numFeatures);
		double d = 0;
		for (int i = 0; i < f.featureIndex.size(); i++)
		{
			int j = f.featureIndex[i];
			d += x[j]*f.featureVec[i];
		}
		return d;
	}

	double featureProduct(const Vector& x, const DenseFeature& f)
	{
		// assert(x.size() == f.featureVec.size());
		double d = 0;
		for (int i = 0; i < f.featureVec.size(); i++)
		{
			d += x[i]*f.featureVec[i];
		}
		return d;
	}
	// An implementation of a feature-vector product, in the case when the feature dimension exceeds that of x.
	double featureProductCheck(const Vector& x, const SparseFeature& f)
	{
		// assert(x.size() == f.numFeatures);
		double d = 0;
		for (int i = 0; i < f.featureIndex.size(); i++)
		{
			int j = f.featureIndex[i];
			if (j < x.size())
				d += x[j]*f.featureVec[i];
		}
		return d;
	}

	// An implementation of a feature-vector product, in the case when the feature dimension exceeds that of x.
	double featureProductCheck(const Vector& x, const DenseFeature& f)
	{
		// assert(x.size() == f.featureVec.size());
		double d = 0;
		for (int i = 0; i < x.size(); i++)
		{
			d += x[i]*f.featureVec[i];
		}
		return d;
	}
	
	void outerProduct(const Vector& x, const Vector& y, Matrix& m)
	{
		for (int i = 0; i < x.size(); i++)
		{
			Vector v(y.size(), 0);
			for (int j = 0; j < y.size(); j++){
				v[j] = x[i]*y[j];
			}
			m.push_back(v);
		}
		return;
	}
	// z = x - alpha*g
	void multiplyAccumulate(Vector& z, const Vector& x, const double alpha, const Vector& g)
	{
		// assert(x.size() == g.size());
		z = Vector(x.size(), 0);
		for (int i = 0; i < x.size(); i++){
			z[i] = x[i] - alpha*g[i];
		}
	}
	
	// x = x - alpha*g
	void multiplyAccumulate(Vector& x, const double alpha, const Vector& g){
		// assert(x.size() == g.size());
		for (int i = 0; i < x.size(); i++){
			x[i] -= alpha*g[i];
		}
	}
	
	double norm(const Vector& x, const int type)
	{
		double val = 0;
		for (int i = 0; i < x.size(); i++)
		{
			if (type == 0) // l_0 norm
				val+= (x[i]==0);
			else if (type == 1) // l_1 norm
				val+= std::abs(x[i]);
			else if (type == 2) // l_2 norm
				val+= pow(x[i], 2);
			else if (type == 3) // l_{\infty} norm
			{
				if (val < x[i])
					val = x[i];
			}
		}
		if (type == 2)
			return sqrt(val);
		else
			return val;
	}
	
	Vector abs(const Vector& x){
		Vector absx(x.size(), 0);
		for (int i = 0; i < x.size(); i++){
			absx[i] = std::abs(x[i]);
		}
		return absx;
	}
	
	Vector sign(const Vector& x){
		Vector sx(x.size(), 0);
		for (int i = 0; i < x.size(); i++){
			sx[i] = sign(x[i]);
		}
		return sx;
	}
	
	void abs(const Vector& x, Vector& absx){
		absx = Vector(x.size(), 0);
		for (int i = 0; i < x.size(); i++){
			absx[i] = std::abs(x[i]);
		}
		return;
	}
	
	void sign(const Vector& x, Vector& sx){
		sx = Vector(x.size(), 0);
		for (int i = 0; i < x.size(); i++){
			sx[i] = sign(x[i]);
		}
		return;
	}
	
	void print(Vector& x){
		for (int i = 0; i < x.size(); i++){
			std::cout<<x[i]<<" ";
		}
		std::cout<<"\n";
	}
	
	template <size_t N> 
	Vector assign(double (&array)[N]){
		Vector v(array, array+N);
		return v;
	}
	
	const Vector operator+(const Vector& x, const Vector &y){ 
		Vector z; 
		vectorAddition(x, y, z);
		return z;
	}

	const Vector operator+(const Vector& x, const SparseFeature &f){ 
		Vector z; 
		vectorFeatureAddition(x, f, z);
		return z;
	}
	
	const Vector operator+(const Vector& x, const DenseFeature &f){ 
		Vector z; 
		vectorFeatureAddition(x, f, z);
		return z;
	}
	
	const Vector operator+(const Vector& x, const double a){ 
		Vector z;
		vectorScalarAddition(x, a, z);
		return z;
	}
	
	const Vector operator-(const Vector& x, const Vector &y){ 
		Vector z; 
		vectorSubtraction(x, y, z);
		return z;
	}

	const Vector operator-(const Vector& x, const SparseFeature &f){ 
		Vector z; 
		vectorFeatureSubtraction(x, f, z);
		return z;
	}
	
	const Vector operator-(const Vector& x, const DenseFeature &f){ 
		Vector z; 
		vectorFeatureSubtraction(x, f, z);
		return z;
	}
	
	const Vector operator-(const Vector& x, const double a){ 
		Vector z;
		vectorScalarSubtraction(x, a, z);
		return z;
	}
	
	const double operator*(const Vector& x, const Vector &y){ 
	 	double d = innerProduct(x, y);
		return d;
	}
	
	const double operator*(const Vector& x, const SparseFeature &f){ 
	 	double d = featureProduct(x, f);
		return d;
	}
	
	const double operator*(const Vector& x, const DenseFeature &f){ 
	 	double d = featureProduct(x, f);
		return d;
	}
	
	const Vector operator*(const Vector& x, const double a){ 
		Vector z; 
		scalarMultiplication(x, a, z);
		return z;
	}
	
	const Vector operator*(const double a, const Vector& x){ 
		Vector z;
		scalarMultiplication(x, a, z);
		return z;
	}
	
	const SparseFeature operator*(const SparseFeature& f, const double a){ 
		SparseFeature g;
		scalarMultiplication(f, a, g);
		return g;
	}
	
	const SparseFeature operator*(const double a, const SparseFeature& f){ 
		SparseFeature g;
		scalarMultiplication(f, a, g);
		return g;
	}
	
	const DenseFeature operator*(const DenseFeature& f, const double a){ 
		DenseFeature g;
		scalarMultiplication(f, a, g);
		return g;
	}
	
	const DenseFeature operator*(const double a, const DenseFeature& f){ 
		DenseFeature g;
		scalarMultiplication(f, a, g);
		return g;
	}
	
	Vector& operator+=(Vector& x, const Vector &y){
		// assert(x.size() == y.size());
		for (int i = 0; i < x.size(); i++)
		{
			x[i] += y[i];
		}
		return x;
	}

	Vector& operator+=(Vector& x, const SparseFeature &f){
		// assert(x.size() == f.numFeatures);
		for (int i = 0; i < f.featureIndex.size(); i++)
		{
			int j = f.featureIndex[i];
			x[j] += f.featureVec[i];
		}
		return x;
	}
	
	Vector& operator+=(Vector& x, const DenseFeature &f){
		// assert(x.size() == f.featureVec.size());
		for (int i = 0; i < x.size(); i++)
		{
			x[i] += f.featureVec[i];
		}
		return x;
	}
	
	Vector& operator+=(Vector& x, const double a){
		for (int i = 0; i < x.size(); i++)
		{
			x[i] += a;
		}
		return x;
	}
	
	Vector& operator-=(Vector& x, const Vector &y){
		// assert(x.size() == y.size());
		for (int i = 0; i < x.size(); i++)
		{
			x[i] -= y[i];
		}
		return x;
	}

	Vector& operator-=(Vector& x, const SparseFeature &f){
		// assert(x.size() == f.numFeatures);
		for (int i = 0; i < f.featureIndex.size(); i++)
		{
			int j = f.featureIndex[i];
			x[j] -= f.featureVec[i];
		}
		return x;
	}
	
	Vector& operator-=(Vector& x, const DenseFeature &f){
		// assert(x.size() == f.featureVec.size());
		for (int i = 0; i < x.size(); i++)
		{
			x[i] -= f.featureVec[i];
		}
		return x;
	}
		
	Vector& operator-=(Vector& x, const double a){
		for (int i = 0; i < x.size(); i++)
		{
			x[i] -= a;
		}
		return x;
	}
	
	Vector& operator*=(Vector& x, const double a){
		for (int i = 0; i < x.size(); i++)
		{
			x[i] *= a;
		}
		return x;
	}
	
    // x == y
    bool operator== (const Vector& x, const Vector& y){
		if (x.size() != y.size())
			return false;
		for (int i = 0; i < x.size(); i++){
			if (x[i] != y[i])
				return false;
		}
		return true;
	}
	
    // x != y
    bool operator!= (const Vector& x, const Vector& y) { return !(x == y); }
	
	bool operator< (const Vector& x, const Vector& y) { 
		// assert(x.size() == y.size());
		for (int i = 0; i < x.size(); i++)
		{
			if (x[i] >= y[i])
				return false;
		}
		return true;
	}
	
	bool operator<= (const Vector& x, const Vector& y) { 
		// assert(x.size() == y.size());
		for (int i = 0; i < x.size(); i++)
		{
			if (x[i] > y[i])
				return false;
		}
		return true;
	}
	
	bool operator> (const Vector& x, const Vector& y) { 
		// assert(x.size() == y.size());
		for (int i = 0; i < x.size(); i++)
		{
			if (x[i] <= y[i])
				return false;
		}
		return true;
	}
	
	bool operator>= (const Vector& x, const Vector& y) { 
		// assert(x.size() == y.size());
		for (int i = 0; i < x.size(); i++)
		{
			if (x[i] < y[i])
				return false;
		}
		return true;
	}
	
	std::ostream& operator<<(std::ostream& os, const Vector& x)
	{
		for (int i = 0; i < x.size(); i++){
			os << x[i] << " ";
		}
	    return os;
	}
	
}


/*
A fast implementation of vector addition. Unfortunately, doesn't seem to be much of use!.
	Vector vectorAddition(const Vector& x, const Vector& y)
	{
		int n = x.size();
		assert(n == y.size());
		Vector z(n, 0);
		int m = n % 4;
		if (m != 0)
		{
			for (int i = 0; i < m; i++){
				z[i] = x[i] + y[i];
			}
			if (n < 4){
				return z;
			}
		}
		for (int i = m; i < x.size(); i+=4)
		{
			z[i] = x[i] + y[i];
			z[i+1] = x[i+1] + y[i+1];
			z[i+2] = x[i+2] + y[i+2];
			z[i+3] = x[i+3] + y[i+3];
			
		}
		return z;
	}
*/
