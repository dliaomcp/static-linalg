/**
 * Author: Dominic Liao-McPherson 
 * Contact: dliaomcp@umich.edu
 * 
 * This file is part of the static-linalg library.
 * Copyright (C) 2018-2019 University of Michigan.
 * 
 * This software is distributed under the BSD-3-Clause license. 
 * You should have received a LICENSE file along with this program. 
 * If not see: <https://opensource.org/licenses/BSD-3-Clause>
 */

#pragma once
#include <iostream>
#include <Eigen/Dense>
#include <cstddef>

// if STATICMATRIX_EXCEPTIONS is defined then
// matrix operations in the static matrix class will throw
// exceptions
#define STATICMATRIX_EXCEPTIONS
#define CHOL_PD_TOL 1e-13

class StaticMatrix{
public:
	// properties *************************************
	// A pointer to the underlying memory
	double *data;
	// sizes
	int nrows;
	int ncols;
	int nels;
	int cap;
	int stride;

	// methods *************************************

	// Constructors
	StaticMatrix(double* mem, int nrows, int ncols = 1);
	StaticMatrix();

	// Copy constructor
	StaticMatrix(const StaticMatrix &A);

	// create special matrices (in place)
	void fill(double a);
	void eye();
	void rand();

	// zero out upper triangle
	void tril();
	// project all elements into the interval [a,b] 
	void clip(double a,double b);

	// Size checks *************************************
	bool IsVector() const;
	bool IsRow() const;
	bool IsCol() const;
	bool IsSquare() const;

	//  Geters *************************************
	int rows() const;
	int cols() const;
	int size() const;

	// Seters *************************************
	void SetStride(int stride);
	void SetCap(int cap);

	// Operator overloads *************************************
	// 2D (matrix) indexing
	double& operator()(int i,int j) const;
	// 1D (vector) indexing
	double& operator()(int i) const;
	// assignment -> shallow copy
	StaticMatrix &operator=(const StaticMatrix& A);
	// y <- a*y
	StaticMatrix& operator*=(double a);

	// Slicing and mapping *************************************
	// used to map a StaticMatrix on top of existing memory
	void map(double* mem, int nrows, int ncols);
	// reshape
	void reshape(int nrows, int ncols);
	// return a reshaped alias
	StaticMatrix getreshape(int nrows, int ncols);
	// return a StaticMatrix which aliases the ith column
	StaticMatrix col(int i);
	// return a StaticMatrix which aliases the ith row
	StaticMatrix row(int i);

	// BLAS operations *************************************
	// deep copy
	// will resize the target matrix if it has sufficient memory
	void copy(const StaticMatrix& A);
	// y <- a*x + y
	void axpy(const StaticMatrix &x,double a);
	// y <- a*A*x + b*y
	void gemv(const StaticMatrix &A, const StaticMatrix &x, double a = 1.0,double b = 0.0,bool transA = false);
	// C <- a*A*B + b*C
	void gemm(const StaticMatrix &A, const StaticMatrix &B, double a,double b, bool transA = false, bool transB = false);

	// Diagonal Matrix products *************************************
	// compute C <- A'*A + C
	void gram(const StaticMatrix &A);
	// compute C <-A'*diag(d)*A + C where d is a vector
	void gram(const StaticMatrix &A,const StaticMatrix& d);
	// compute A <- diag(d)*A where d is a vector
	void RowScale(const StaticMatrix &d);
	// compute A <- A*diag(d) where d is a vector
	void ColScale(const StaticMatrix &d);


	// norms *************************************
	double norm(); // 2 norm of a vector or F norm of a matrix
	double asum(); // 1 norm

	// Factorizations *************************************
	// In place factorization
	// returns 0 if successful
	// return -1 if dynamic regularization was applied
	int llt(); // LL'

	// Apply a cholesky factorization to a matrix or vector
	// computes x <- inv(A) x, A = LL'
	void CholSolve(const StaticMatrix &L);
	// computes A <- A*inv(L) or A <- A*inv(L)'
	void RightCholApply(const StaticMatrix &L, bool transL = false);
	// computes A <- inv(L)*A or A <- inv(L)'*A
	void LeftCholApply(const StaticMatrix &L, bool transL = false);


	// Print
	friend std::ostream &operator<<(std::ostream& output, const StaticMatrix &A);

	// static methods *************************************
	// max and min operations
	template <class T>
	static T max(T a, T b){
		return (a>b) ? a : b;
	}

	template <class T>
	static T min(T a, T b){
		return (a>b) ? b : a;
	}
	// y'*x
	static double dot(const StaticMatrix& y, const StaticMatrix& x);
	
	// checks if two StaticMatrix objects are the same size
	static bool SameSize(const StaticMatrix &A, const StaticMatrix &B);

};




