#pragma once

#include <iostream>
#include <immintrin.h>

#include "utils.h"
#include "mat.h"
#include "timer.h"

using namespace std;

class sparse_operator
{
public:
	static _float nrm2_sq( _int siz, pairIF* x )
	{
		_float ret = 0;
		for( _int i=0; i<siz; i++ )
		{
			ret += SQ( x[i].second );
		}
		return (ret);
	}

	static _float dot( const _float *s, _int siz, pairIF* x )
	{
		_float ret = 0;
		for( _int i=0; i<siz; i++ )
			ret += s[ x[i].first ] * x[i].second;
		return (ret);
	}

	static void axpy(const _float a, _int siz, pairIF* x, _float *y)
	{
		for( _int i=0; i<siz; i++ )
		{
			y[x[i].first ] += a * x[i].second;
		}
	}
};

// To support weights for instances, use GETI(i) (i)

void solve_l2r_lr_dual( SMatF* X_Xf, _int* y, _float *w, _float eps, _float Cp, _float Cn, _int svm_iter );
void solve_l2r_l1l2_svc( SMatF* X_Xf, _int* y, _float *w, _float eps, _float Cp, _float Cn, _int svm_iter );


