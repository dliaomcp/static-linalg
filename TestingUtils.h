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
#include <Eigen/Dense>
#include "StaticMatrix.h"

namespace testutils {
	// difference metric between an Static and Eigen Matrix
	double DiffNorm(const StaticMatrix &A, const Eigen::Ref<const Eigen::MatrixXd> &B);
	// Copy a static matrix into an eigen matrix
	void CopyEig(const StaticMatrix &A, Eigen::MatrixXd &B);
} // namespace testutils