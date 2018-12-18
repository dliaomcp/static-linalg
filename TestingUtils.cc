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


#include "TestingUtils.h"
#include <Eigen/Dense>
#include "StaticMatrix.h"

namespace testutils{
// used to compute a norm of the difference between an eigen matrix and a StaticMatrix
double DiffNorm(const StaticMatrix &A, const Eigen::Ref<const Eigen::MatrixXd> &B){
	int m = A.rows();
	int n = A.cols();

	double a = 0;
	for(int i = 0;i<m;i++){
		for(int j = 0;j<n;j++){
			a += abs(A(i,j) - B(i,j));
		}
	}
	return a;
}

void CopyEig(const StaticMatrix &A, Eigen::MatrixXd &B){
	int m = A.rows();
	int n = A.cols();

	for(int i = 0;i<m;i++){
		for(int j = 0;j<n;j++){
			B(i,j) = A(i,j);
		}
	}
}

} // namespace testutils