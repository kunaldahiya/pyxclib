#pragma once

#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <vector>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>

#include "config.h"

using namespace std;


/* ------------------------ defines, typedefs, inlines, templates --------------------------- */

typedef pair<_int,_float> pairIF;
typedef pair<_int,_double> pairID;
typedef pair<_int,_int> pairII;
typedef pair<_int,_bool> pairIB;

#define LINE (cout<<__LINE__<<endl)
#define NAME_LEN 1000
#define SQ(x) ((x)*(x))
#define INF FLT_MAX
#define NEG_INF FLT_MIN
#define EPS 1e-10

enum LOGLVL {QUIET, PROGRESS, DEBUG};

template <typename T1,typename T2>
_bool comp_pair_by_second_desc(pair<T1,T2> a, pair<T1,T2> b)
{
	if(a.second>b.second)
		return true;
	return false;
}

template <typename T1,typename T2>
_bool comp_pair_by_second(pair<T1,T2> a, pair<T1,T2> b)
{
	if(a.second<b.second)
		return true;
	return false;
}

template <typename T1,typename T2>
_bool comp_pair_by_first(pair<T1,T2> a, pair<T1,T2> b)
{
	if(a.first<b.first)
		return true;
	return false;
}

template <typename T>
void Realloc(_int old_size, _int new_size, T*& vec)
{
	T* new_vec = new T[new_size];
	_int size = min(old_size,new_size);
	copy_n(vec,size,new_vec);
	delete [] vec;
	vec = new_vec;
}

template <typename T>
void copy_S_to_D(_int size, pair<_int,T>* sarr, T* darr)
{
	for(_int i=0; i<size; i++)
	{
		darr[sarr[i].first] = sarr[i].second;
	}
}

template <typename T>
void reset_D(_int size, pair<_int,T>* sarr, T* darr)
{
	for(_int i=0; i<size; i++)
	{
		darr[sarr[i].first] = 0;
	}
}

inline void check_valid_filename(string fname, _bool read=true)
{
	_bool valid;
	ifstream fin;
	ofstream fout;
	if(read)
	{
		fin.open(fname);
		valid = fin.good();
	}
	else
	{
		fout.open(fname);
		valid = fout.good();
	}

	if(!valid)
	{
		cerr<<"error: invalid file name: "<<fname<<endl<<"exiting..."<<endl;
		exit(1);
	}
	if(read)
	{
		fin.close();
	}
	else
	{
		fout.close();
	}
}

inline void check_valid_foldername(string fname)
{
	string tmp_file = fname+"/tmp.txt";
	ofstream fout(tmp_file);

	if(!fout.good())
	{
		cerr<<"error: invalid folder name: "<<fname<<endl<<"exiting..."<<endl;
		exit(1);
	}
	remove(tmp_file.c_str());
}

template <typename T>
void print_vector( ostream& fout, vector<T>& vec )
{
	for( _int i=0; i<vec.size(); i++ )
		if(i==0)
			fout << vec[i];
		else
			fout << " " << vec[i];
	fout << endl;
}

template <typename T1, typename T2>
void print_vector( ostream& fout, vector< pair<T1,T2> >& vec )
{
	for( _int i=0; i<vec.size(); i++ )
		if(i==0)
			fout << vec[i].first << ":" << vec[i].second;
		else
			fout << " " << vec[i].first << ":" << vec[i].second;
	fout << endl;
}

