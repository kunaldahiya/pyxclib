#pragma once

#include <iostream>
#include <iomanip>
#include <string>
#include <algorithm>
#include <vector>
#include <fstream>
#include <map>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <cassert>

#include "config.h"
#include "utils.h"
#include "timer.h"

using namespace std;

/* ------------------- Sparse and dense matrix and vector resources ---------------------- */

typedef vector<_int> VecI;
typedef vector<_float> VecF;
typedef vector<_double> VecD;
typedef vector<pairII> VecII;
typedef vector<pairIF> VecIF;
typedef vector<_bool> VecB;

template <typename T>
class SMat // a column-major sparse matrix of type T
{
public:
	_bool contiguous;
	_int nc;
	_int nr;
	_int* size;
	pair<_int,T>** data;
	pair<_int,T>* cdata;

	SMat( _bool contiguous = false )
	{
		this->contiguous = contiguous;
		nc = 0;
		nr = 0;
		size = NULL;
		data = NULL;
		cdata = NULL;
	}

	SMat( _int nr, _int nc, _bool contiguous = false )
	{
		this->contiguous = contiguous;
		this->nr = nr;
		this->nc = nc;
		size = new _int[nc]();
		data = new pair<_int,T>*[nc];
		for(_int i=0; i<nc; i++)
			data[i] = NULL;
		cdata = NULL;
	}

	SMat( _int nr, _int nc, _int nnz, _bool contiguous = false )
	{
		this->contiguous = contiguous;
		this->nr = nr;
		this->nc = nc;
		size = new _int[nc]();
		data = new pair<_int,T>*[nc];
		for(_int i=0; i<nc; i++)
			data[i] = NULL;

		if( contiguous )
			cdata = new pair<_int,T>[ nnz ];
		else
			cdata = NULL;
	}
	
	SMat(SMat<T>* mat)
	{
		contiguous = false;
		nc = mat->nc;
		nr = mat->nr;
		size = new _int[nc];

		for(_int i=0; i<nc; i++)
			size[i] = mat->size[i];

		data = new pair<_int,T>*[nc];
		for(_int i=0; i<nc; i++)
		{
			data[i] = new pair<_int,T>[size[i]];
			for(_int j=0; j<size[i]; j++)
			{
				data[i][j] = mat->data[i][j];
			}	
		}
	}

	friend istream& operator>>( istream& fin, SMat<T>& mat )
	{
		vector<_int> inds;
		vector<T> vals;

		fin >> mat.nc >> mat.nr;

		mat.size = new _int[ mat.nc ];
		mat.data = new pair<_int,T>*[ mat.nc ];

		fin.ignore();
		for(_int i=0; i<mat.nc; i++)
		{
			inds.clear();
			vals.clear();
			string line;
			getline(fin,line);
			line += "\n";
			_int pos = 0;
			_int next_pos;

			while(next_pos=line.find_first_of(": \n",pos))
			{
				if((size_t)next_pos==string::npos)
					break;
				inds.push_back(stoi(line.substr(pos,next_pos-pos)));
				pos = next_pos+1;

				next_pos = line.find_first_of(": \n",pos);
				if((size_t)next_pos==string::npos)
					break;

				vals.push_back(stof(line.substr(pos,next_pos-pos)));
				pos = next_pos+1;

			}

			assert(inds.size()==vals.size());
			assert(inds.size()==0 || inds[inds.size()-1]<mat.nr);

			mat.size[i] = inds.size();
			mat.data[i] = new pair<_int,T>[inds.size()];

			for(_int j=0; j<mat.size[i]; j++)
			{
				mat.data[i][j].first = inds[j];
				mat.data[i][j].second = (T)vals[j];
			}
		}

		return fin;
	}

	SMat(string fname)
	{
		contiguous = false;
		check_valid_filename(fname,true);

		ifstream fin;
		fin.open(fname);		

		vector<_int> inds;
		vector<T> vals;

		fin>>nc>>nr;
		size = new _int[nc];
		data = new pair<_int,T>*[nc];

		fin.ignore();
		for(_int i=0; i<nc; i++)
		{
			inds.clear();
			vals.clear();
			string line;
			getline(fin,line);
			line += "\n";
			_int pos = 0;
			_int next_pos;

			while(next_pos=line.find_first_of(": \n",pos))
			{
				if((size_t)next_pos==string::npos)
					break;
				inds.push_back(stoi(line.substr(pos,next_pos-pos)));
				pos = next_pos+1;

				next_pos = line.find_first_of(": \n",pos);
				if((size_t)next_pos==string::npos)
					break;

				vals.push_back(stof(line.substr(pos,next_pos-pos)));
				pos = next_pos+1;

			}

			assert(inds.size()==vals.size());
			assert(inds.size()==0 || inds[inds.size()-1]<nr);

			size[i] = inds.size();
			data[i] = new pair<_int,T>[inds.size()];

			for(_int j=0; j<size[i]; j++)
			{
				data[i][j].first = inds[j];
				data[i][j].second = (T)vals[j];
			}
		}

		fin.close();
	}

	_float get_ram()
	{
		_float ram = 0;
		ram += 4*2 + 8*2 + 4*nc + 8*nc;

		for( _int i=0; i<nc; i++ )
			ram += 8*size[i];

		return ram;
	}

	SMat<T>* transpose()
	{
		SMat<T>* tmat = new SMat<T>;
		tmat->nr = nc;
		tmat->nc = nr;
		tmat->size = new _int[tmat->nc]();
		tmat->data = new pair<_int,T>*[tmat->nc];

		for(_int i=0; i<nc; i++)
		{
			for(_int j=0; j<size[i]; j++)
			{
				tmat->size[data[i][j].first]++;
			}
		}

		for(_int i=0; i<tmat->nc; i++)
		{
			tmat->data[i] = new pair<_int,T>[tmat->size[i]];
		}

		_int* count = new _int[tmat->nc]();
		for(_int i=0; i<nc; i++)
		{
			for(_int j=0; j<size[i]; j++)
			{
				_int ind = data[i][j].first;
				T val = data[i][j].second;

				tmat->data[ind][count[ind]].first = i;
				tmat->data[ind][count[ind]].second = val;
				count[ind]++;
			}
		}

		delete [] count;
		return tmat;
	}

	void threshold( _float th )
	{
		for( _int i=0; i<nc; i++ )
		{
			_int count = 0;
			for( _int j=0; j<size[i]; j++ )
				count += fabs( data[i][j].second )>th;

			pair<_int,T>* newvec = new pair<_int,T>[count];
			count = 0;
			for( _int j=0; j<size[i]; j++ )
			{
				_int id = data[i][j].first;
				T val = data[i][j].second;
				if( fabs(val)>th )
					newvec[ count++ ] = make_pair( id, val );
			}
			size[i] = count;
			delete [] data[i];
			data[i] = newvec;
		}
	}

	void unit_normalize_columns()
	{
		for(_int i=0; i<nc; i++)
		{
			T normsq = 0;
			for(_int j=0; j<size[i]; j++)
				normsq += SQ(data[i][j].second);
			normsq = sqrt(normsq);

			if(normsq==0)
				normsq = 1;

			for(_int j=0; j<size[i]; j++)
				data[i][j].second /= normsq;
		}
	}

	vector<T> column_norms()
	{
		vector<T> norms(nc,0);

		for(_int i=0; i<nc; i++)
		{
			T normsq = 0;
			for(_int j=0; j<size[i]; j++)
				normsq += SQ(data[i][j].second);
			norms[i] = sqrt(normsq);
		}

		return norms;
	}

	~SMat()
	{
		delete [] size;

		if( contiguous )
		{
			delete [] cdata;
			delete [] data; 
		}
		else
		{
			for( _int i=0; i<nc; i++ )
				delete [] data[i];
			delete [] data;
		}
	}

	friend ostream& operator<<( ostream& fout, const SMat<T>& mat )
	{
		_int nc = mat.nc;
		_int nr = mat.nr;
		_int* size = mat.size;
		pairIF** data = mat.data;

		fout<<nc<<" "<<nr<<endl;

		fout << fixed << setprecision( 6 );
		for(_int i=0; i<nc; i++)
		{
			for(_int j=0; j<size[i]; j++)
			{
				if(j==0)
					fout<<data[i][j].first<<":"<<data[i][j].second;
				else
					fout<<" "<<data[i][j].first<<":"<<data[i][j].second;
			}
			fout<<endl;
		}

		return fout;
	}

	void write(string fname)
	{
		check_valid_filename(fname,false);

		ofstream fout;
		fout.open(fname);

		fout << (*this);

		fout.close();
	}

	void add(SMat<T>* smat)
	{
		if(nc != smat->nc || nr != smat->nr)
		{
			cerr<<"SMat::add : Matrix dimensions do not match"<<endl;
			cerr<<"Matrix 1: "<<nc<<" x "<<nr<<endl;
			cerr<<"Matrix 2: "<<smat->nc<<" x "<<smat->nr<<endl;
			exit(1);
		}

		bool* ind_mask = new bool[nr]();
		T* sum = new T[nr]();

		for(_int i=0; i<nc; i++)
		{
			vector<_int> inds;
			for(_int j=0; j<size[i]; j++)
			{
				_int ind = data[i][j].first;
				T val = data[i][j].second;

				sum[ind] += val;
				if(!ind_mask[ind])
				{
					ind_mask[ind] = true;
					inds.push_back(ind);
				}
			}

			for(_int j=0; j<smat->size[i]; j++)
			{
				_int ind = smat->data[i][j].first;
				T val = smat->data[i][j].second;

				sum[ind] += val;
				if(!ind_mask[ind])
				{
					ind_mask[ind] = true;
					inds.push_back(ind);
				}
			}

			sort(inds.begin(), inds.end());
			Realloc(size[i], inds.size(), data[i]);
			for(_int j=0; j<inds.size(); j++)
			{
				_int ind = inds[j];
				data[i][j] = make_pair(ind,sum[ind]);
				ind_mask[ind] = false;
				sum[ind] = 0;
			}
			size[i] = inds.size();
		}

		delete [] ind_mask;
		delete [] sum;
	}

	SMat<T>* prod(SMat<T>* mat2)
	{
		_int dim1 = nr;
		_int dim2 = mat2->nc;

		assert(nc==mat2->nr);

		SMat<T>* prodmat = new SMat<T>(dim1,dim2);
		vector<T> sum(dim1,0);

		for(_int i=0; i<dim2; i++)
		{
			vector<_int> indices;
			for(_int j=0; j<mat2->size[i]; j++)
			{
				_int ind = mat2->data[i][j].first;
				T prodval = mat2->data[i][j].second;

				for(_int k=0; k<size[ind]; k++)
				{
					_int id = data[ind][k].first;
					T val = data[ind][k].second;

					if(sum[id]==0)
						indices.push_back(id);

					sum[id] += val*prodval;
				}
			}

			sort(indices.begin(), indices.end());

			_int siz = indices.size();
			prodmat->size[i] = siz;
			prodmat->data[i] = new pair<_int,T>[siz];

			for(_int j=0; j<indices.size(); j++)
			{
				_int id = indices[j];
				T val = sum[id];
				prodmat->data[i][j] = make_pair(id,val);
				sum[id] = 0;
			}
		}

		return prodmat;
	}

	void append_bias_feat( _float bias_feat ) // assumes non-contiguous matrix
	{
		for( _int i=0; i<nc; i++ )
		{
			_int siz = size[i];
			Realloc( siz, siz+1, data[i] );
			data[i][siz] = make_pair( nr, bias_feat );
			size[i]++;	
		}
		nr++;
	}

	void active_dims( VecI& cols, VecI& dims, VecI& counts, VecI& countmap )
	{
		dims.clear();
		counts.clear();

		for( _int i=0; i<cols.size(); i++ )
		{
			_int inst = cols[i];
			for( _int j=0; j<size[inst]; j++ )
			{
				_int dim = data[inst][j].first;
				if( countmap[ dim ]==0 )
					dims.push_back(dim);
				countmap[ dim ]++;
			}
		}

		sort(dims.begin(),dims.end());

		for( _int i=0; i<dims.size(); i++ )
		{
			counts.push_back( countmap[ dims[i] ] );
			countmap[ dims[i] ] = 0;
		}
	}
    
    void shrink_mat( VecI& cols, SMat<T>*& s_mat, VecI& rows, VecI& countmap, _bool transpose )
    {
        _int s_nc = cols.size();
        VecI counts;
        active_dims( cols, rows, counts, countmap );

        _int nnz = 0;
        for( _int i=0; i<counts.size(); i++ )
            nnz += counts[i];

        _int* maps = new _int[ nr ];
        for( _int i=0; i<rows.size(); i++ )
            maps[ rows[i] ] = i;

        _int s_nr = rows.size();
        
        if( transpose )
        {
            s_mat = new SMat<T>( s_nc, s_nr, nnz, true );
        
            _int sumsize = 0;
            for( _int i=0; i<s_nr; i++ )
            {
                s_mat->size[i] = counts[i];
                s_mat->data[i] = s_mat->cdata + sumsize;
                sumsize += counts[i];
            }
            
            for( _int i=0; i<s_nr; i++ )
                counts[i] = 0;
        }
        else
        {
            s_mat = new SMat<T>( s_nr, s_nc, nnz, true );

            _int sumsize = 0;
            for( _int i=0; i<s_nc; i++)
            {
                _int col = cols[i];
                s_mat->size[i] = size[ col ];
                s_mat->data[i] = s_mat->cdata + sumsize;
                sumsize += size[ col ];
            }
        }
            
        for( _int i=0; i<s_nc; i++ )
        {	
            _int col = cols[ i ];
            for( _int j=0; j<size[ col ]; j++ )
            {
                _int row = maps[ data[ col ][ j ].first ];
                _float val = data[ col ][ j ].second;
                
                if( transpose )
                {
                    s_mat->data[row][counts[row]] = make_pair( i, val );
                    counts[row]++;
                }
                else
                    s_mat->data[i][j] = make_pair( row, val );
            }
        }

        delete [] maps;
    }
};
                  
template <typename T>
class DMat // a column-major dense matrix of type T
{
public:
	_int nc;
	_int nr;
	T** data;

	DMat()
	{
		nc = 0;
		nr = 0;
		data = NULL;
	}

	DMat(_int nc, _int nr)
	{
		this->nc = nc;
		this->nr = nr;
		data = new T*[nc];
		for(_int i=0; i<nc; i++)
			data[i] = new T[nr]();
	}

	DMat(SMat<T>* mat)
	{
		nc = mat->nc;
		nr = mat->nr;
		data = new T*[nc];
		for(_int i=0; i<nc; i++)
			data[i] = new T[nr]();

		for(_int i=0; i<mat->nc; i++)
		{
			pair<_int,T>* vec = mat->data[i];
			for(_int j=0; j<mat->size[i]; j++)
			{
				data[i][vec[j].first] = vec[j].second;
			}
		}
	}

	~DMat()
	{
		for(_int i=0; i<nc; i++)
			delete [] data[i];
		delete [] data;
	}
};

typedef SMat<_float> SMatF;
typedef DMat<_float> DMatF;

void reindex_VecIF( VecIF& vec, VecI& index );
