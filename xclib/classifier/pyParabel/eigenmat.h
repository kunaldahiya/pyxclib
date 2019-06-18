// INC=-I../Tools/c++
// CXXFLAGS=-std=c++11 -O3 -g
// LIBFLAGS=-pthread -I/home/cse/msr/siy177545/Eigen -I/home/cse/msr/siy177545/scratch/anaconda3/include/python3.7m -I/home/cse/msr/siy177545/.local/include/python3.7m
// $(CXX) -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` parabel.cpp $(LIBFLAGS) -o parabel`python3-config --extension-suffix`

#include <iostream>
#include <vector>
#include <map>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <ctime>

#include "config.h"
#include "utils.h"
#include "mat.h"

#include <pybind11/pybind11.h>
#include <Python.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Cholesky>

using namespace std;
using namespace Eigen;

using MatrixXdR = SparseMatrix<float, RowMajor>;

MatrixXdR *SMAT2SPARSE(SMatF *Mat)
{
	MatrixXdR *x;
	x = new MatrixXdR(Mat->nc, Mat->nr);
	for (int i = 0; i < Mat->nc; i++)
		for (int j = 0; j < Mat->size[i]; j++)
			x->insert(i, Mat->data[i][j].first) = Mat->data[i][j].second;
	return x;
}

SMatF *SPARSE2SMAT(MatrixXdR *Mat)
{
	SMatF *x;
	x = new SMatF(Mat->cols(), Mat->rows());
	vector<_int> inds;
	vector<float> vals;
	for (int k = 0; k < Mat->outerSize(); ++k)
	{
		inds.clear();
		vals.clear();
		for (MatrixXdR::InnerIterator it(*Mat, k); it; ++it)
		{
			inds.push_back(it.col());
			vals.push_back(it.value());
		}
		assert(inds.size() == vals.size());
		assert(inds.size() == 0 || inds[inds.size() - 1] < x->nr);

		x->size[k] = inds.size();
		x->data[k] = new pair<_int, float>[inds.size()];

		for (_int j = 0; j < x->size[k]; j++)
		{
			x->data[k][j].first = inds[j];
			x->data[k][j].second = vals[j];
		}
	}
	return x;
}
