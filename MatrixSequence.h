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
#include "StaticMatrix.h"

// TODO: template this on matrix type?
class MatrixSequence{

public:

	// properties *************************************
	int nrows; // rows in each matrix
	int ncols; // columns in each matrix
	int nseq; // number of matrices in the sequence
	double **data; // array of pointers to the raw
	StaticMatrix


	// methods *************************************

	// constructor from array of matrix types
	MatrixSequence(StaticMatrix *seq);
	// constructor from raw memory
	MatrixSequence(double *mem, int nrows, int ncols,int nseq);



	int rows() const;
	int cols() const;
	int len() const;

	// element access

	// matrix access

	// multiplcation?

};