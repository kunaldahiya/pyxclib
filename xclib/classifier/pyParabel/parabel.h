#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <iomanip>
#include <random>
#include <thread>
#include <mutex>
#include <functional>

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <ctime>

#include "config.h"
#include "utils.h"
#include "mat.h"
#include "timer.h"
#include "svm.h"
#include "eigenmat.h"

using namespace std;

enum _Classifier_Kind { L2R_L2LOSS_SVC=0, L2R_LR };

class Node
{
public:
	_bool is_leaf;
	_int pos_child;
	_int neg_child;
	_int depth;
	VecI Y;
	SMatF* w;
	VecIF X;

	Node()
	{
		is_leaf = false;
		pos_child = neg_child = -1;
		depth = 0;
		w = NULL;
	}

	Node( VecI Y, _int depth, _int max_depth )
	{
		this->Y = Y;
		this->depth = depth;
		this->pos_child = -1;
		this->neg_child = -1;
		this->is_leaf = (depth >= max_depth-1);
		this->w = NULL;
	}

	~Node()
	{
		delete w;
	}

	_float get_ram()
	{
		_float ram = 0;
		ram += sizeof( Node );
		if( this->is_leaf )  // Label values in internal nodes are not essential for model and hence not included in model size measurements
			ram += sizeof( _int ) * Y.size();
		ram += w->get_ram();
		return ram;
	}

	friend ostream& operator<<( ostream& fout, const Node& node )
	{
		fout << node.is_leaf << "\n";
		fout << node.pos_child << " " << node.neg_child << "\n";
		fout << node.depth << "\n";

		fout << node.Y.size();
		for( _int i=0; i<node.Y.size(); i++ )
			fout << " " << node.Y[i];
		fout << "\n";

		fout << (*node.w);

		return fout;
	}

 	friend istream& operator>>( istream& fin, Node& node )
	{
		fin >> node.is_leaf;
		fin >> node.pos_child >> node.neg_child;
		fin >> node.depth;

		_int Y_size;
		fin >> Y_size;
		node.Y.resize( Y_size );

		for( _int i=0; i<Y_size; i++ )
			fin >> node.Y[i];

		node.w = new SMatF;
		fin >> (*node.w);

		return fin;
	} 
};

class Tree
{
public:
	_int num_Xf;
	_int num_Y;
	vector<Node*> nodes;

	Tree()
	{
		
	}

	Tree( string model_dir, _int tree_no )
	{
		ifstream fin;
		fin.open( model_dir + "/" + to_string( tree_no ) + ".tree" );

		fin >> num_Xf;
		fin >> num_Y;
		_int num_node;
		fin >> num_node;

		for( _int i=0; i<num_node; i++ )
		{
			Node* node = new Node;
			nodes.push_back( node );
		}

		for( _int i=0; i<num_node; i++ )
			fin >> (*nodes[i]);

		fin.close();
	}

	~Tree()
	{
		for(_int i=0; i<nodes.size(); i++)
			delete nodes[i];
	}

	_float get_ram()
	{
		_float ram = 0;
		ram += sizeof( Tree );
		for(_int i=0; i<nodes.size(); i++)
			ram += nodes[i]->get_ram();
		return ram;
	}

	void write( string model_dir, _int tree_no )
	{
		ofstream fout;
		fout.open( model_dir + "/" + to_string( tree_no ) + ".tree" );

		fout << num_Xf << "\n";
		fout << num_Y << "\n";
		_int num_node = nodes.size();
		fout << num_node << "\n";

		for( _int i=0; i<num_node; i++ )
			fout << (*nodes[i]);

		fout.close();
	}
};

class Param
{
public:
	_int num_trn;
	_int num_Xf;
	_int num_Y;
	_int num_thread;
	_int start_tree;
	_int num_tree;
	_float classifier_cost;
	_int max_leaf;
	_float bias_feat;
	_float classifier_threshold;
	_float centroid_threshold;
	_float clustering_eps;
	_int classifier_maxitr;
	_Classifier_Kind classifier_kind;
	_bool quiet;
	_int beam_width;

	Param()
	{
		num_Xf = 0;
		num_Y = 0;
		num_thread = 1;
		start_tree = 0;
		num_tree = 3;
		classifier_cost = 1.0;
		max_leaf = 100;
		bias_feat = 1.0;
		classifier_threshold = 0.1;
		centroid_threshold = 0;
		clustering_eps = 1e-4;
		classifier_maxitr = 20;
		classifier_kind = L2R_L2LOSS_SVC;
		quiet = false;
		beam_width = 10;
	}

	Param(string fname)
	{
		check_valid_filename(fname,true);
		ifstream fin;
		fin.open(fname);

		fin>>num_Xf;
		fin>>num_Y;
		fin>>num_thread;
		fin>>start_tree;
		fin>>num_tree;
		fin>>classifier_cost;
		fin>>max_leaf;
		fin>>bias_feat;
		fin>>classifier_threshold;
		fin>>centroid_threshold;
		fin>>clustering_eps;
		fin>>classifier_maxitr;
		_int ck;
		fin>>ck;
		classifier_kind = (_Classifier_Kind)ck;
		fin>>quiet;
		fin>>beam_width;
		fin.close();
	}

	void write(string fname)
	{
		check_valid_filename(fname,false);
		ofstream fout;
		fout.open(fname);

		fout<<num_Xf<<"\n";
		fout<<num_Y<<"\n";
		fout<<num_thread<<"\n";
		fout<<start_tree<<"\n";
		fout<<num_tree<<"\n";
		fout<<classifier_cost<<"\n";
		fout<<max_leaf<<"\n";
		fout<<bias_feat<<"\n";
		fout<<classifier_threshold<<"\n";
		fout<<centroid_threshold<<"\n";
		fout<<clustering_eps<<"\n";
		fout<<classifier_maxitr<<"\n";
		fout<<classifier_kind<<"\n";
		fout<<quiet<<"\n";
		fout<<beam_width<<"\n";

		fout.close();
	}

	friend ostream& operator<<( ostream& fout, const Param& param )
	{
		fout<<param.num_Xf<<"\n";
		fout<<param.num_Y<<"\n";
		fout<<param.num_thread<<"\n";
		fout<<param.start_tree<<"\n";
		fout<<param.num_tree<<"\n";
		fout<<param.classifier_cost<<"\n";
		fout<<param.max_leaf<<"\n";
		fout<<param.bias_feat<<"\n";
		fout<<param.classifier_threshold<<"\n";
		fout<<param.centroid_threshold<<"\n";
		fout<<param.clustering_eps<<"\n";
		fout<<param.classifier_maxitr<<"\n";
		fout<<param.classifier_kind<<"\n";
		fout<<param.quiet<<"\n";
		fout<<param.beam_width<<"\n";
		return fout;
	}
};

Tree* train_tree( SMatF* trn_X_Xf, SMatF* trn_X_Y, Param& param, _int tree_no );
void train_trees( SMatF* trn_X_Xf, SMatF* trn_X_Y, Param& param, string model_dir, _float& train_time );
void train(MatrixXdR ft_file, MatrixXdR lbl_file, string model_dir, _int num_thread, _int start_tree, _int num_tree,
		   _float bias_feat, _float classifier_cost, _int max_leaf, _float classifier_threshold,
		   _float centroid_threshold, _float clustering_eps, _int classifier_maxitr,
		   _int classifier_kind, _bool quite);

SMatF* predict_tree( SMatF* tst_X_Xf, Tree* tree, Param& param );
SMatF* predict_trees( SMatF* tst_X_Xf, Param& param, string model_dir, _float& prediction_time, _float& model_size );

MatrixXdR *predict(MatrixXdR ft_file, string model_dir, _int num_thread,
				   _int start_tree, _int num_tree, _bool quite);
