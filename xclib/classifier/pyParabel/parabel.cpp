#include "parabel.h"

namespace py = pybind11;
using namespace std;

mutex mtx;
thread_local mt19937 reng; // random number generator used during training
thread_local VecI countmap;


void setup_thread_locals(_int num_X, _int num_Xf, _int num_Y)
{
	countmap.resize(max(max(num_Xf, num_Y), num_X), 0);
}

_int get_rand_num(_int siz)
{
	_llint r = reng();
	_int ans = r % siz;
	return ans;
}

Node *init_root(_int num_Y, _int max_depth)
{
	VecI lbls;
	for (_int i = 0; i < num_Y; i++)
		lbls.push_back(i);
	Node *root = new Node(lbls, 0, max_depth);
	return root;
}

pairII get_pos_neg_count(VecI &pos_or_neg)
{
	pairII counts = make_pair(0, 0);
	for (_int i = 0; i < pos_or_neg.size(); i++)
	{
		if (pos_or_neg[i] == +1)
			counts.first++;
		else
			counts.second++;
	}
	return counts;
}

void reset_d_with_s(pairIF *svec, _int siz, _float *dvec)
{
	for (_int i = 0; i < siz; i++)
		dvec[svec[i].first] = 0;
}

void set_d_with_s(pairIF *svec, _int siz, _float *dvec)
{
	for (_int i = 0; i < siz; i++)
		dvec[svec[i].first] = svec[i].second;
}

void init_2d_float(_int dim1, _int dim2, _float **&mat)
{
	mat = new _float *[dim1];
	for (_int i = 0; i < dim1; i++)
		mat[i] = new _float[dim2];
}

void delete_2d_float(_int dim1, _int dim2, _float **&mat)
{
	for (_int i = 0; i < dim1; i++)
		delete[] mat[i];
	delete[] mat;
	mat = NULL;
}

void reset_2d_float(_int dim1, _int dim2, _float **&mat)
{
	for (_int i = 0; i < dim1; i++)
		for (_int j = 0; j < dim2; j++)
			mat[i][j] = 0;
}

_float mult_d_s_vec(_float *dvec, pairIF *svec, _int siz)
{
	_float prod = 0;
	for (_int i = 0; i < siz; i++)
	{
		_int id = svec[i].first;
		_float val = svec[i].second;
		prod += dvec[id] * val;
	}
	return prod;
}

void add_s_to_d_vec(pairIF *svec, _int siz, _float *dvec)
{
	for (_int i = 0; i < siz; i++)
	{
		_int id = svec[i].first;
		_float val = svec[i].second;
		dvec[id] += val;
	}
}

_float get_norm_d_vec(_float *dvec, _int siz)
{
	_float norm = 0;
	for (_int i = 0; i < siz; i++)
		norm += SQ(dvec[i]);
	norm = sqrt(norm);
	return norm;
}

void div_d_vec_by_scalar(_float *dvec, _int siz, _float s)
{
	for (_int i = 0; i < siz; i++)
		dvec[i] /= s;
}

void normalize_d_vec(_float *dvec, _int siz)
{
	_float norm = get_norm_d_vec(dvec, siz);
	if (norm > 0)
		div_d_vec_by_scalar(dvec, siz, norm);
}

void balanced_kmeans(SMatF *mat, _float acc, VecI &partition)
{
	_int nc = mat->nc;
	_int nr = mat->nr;

	_int c[2] = {-1, -1};
	c[0] = get_rand_num(nc);
	c[1] = c[0];
	while (c[1] == c[0])
		c[1] = get_rand_num(nc);

	_float **centers;
	init_2d_float(2, nr, centers);
	reset_2d_float(2, nr, centers);
	for (_int i = 0; i < 2; i++)
		set_d_with_s(mat->data[c[i]], mat->size[c[i]], centers[i]);

	_float **cosines;
	init_2d_float(2, nc, cosines);

	pairIF *dcosines = new pairIF[nc];

	partition.resize(nc);

	_float old_cos = -10000;
	_float new_cos = -1;

	while (new_cos - old_cos >= acc)
	{

		for (_int i = 0; i < 2; i++)
		{
			for (_int j = 0; j < nc; j++)
				cosines[i][j] = mult_d_s_vec(centers[i], mat->data[j], mat->size[j]);
		}

		for (_int i = 0; i < nc; i++)
		{
			dcosines[i].first = i;
			dcosines[i].second = cosines[0][i] - cosines[1][i];
		}

		sort(dcosines, dcosines + nc, comp_pair_by_second_desc<_int, _float>);

		old_cos = new_cos;
		new_cos = 0;
		for (_int i = 0; i < nc; i++)
		{
			_int id = dcosines[i].first;
			_int part = (_int)(i < nc / 2);
			partition[id] = 1 - part;
			new_cos += cosines[partition[id]][id];
		}
		new_cos /= nc;

		reset_2d_float(2, nr, centers);

		for (_int i = 0; i < nc; i++)
		{
			_int p = partition[i];
			add_s_to_d_vec(mat->data[i], mat->size[i], centers[p]);
		}

		for (_int i = 0; i < 2; i++)
			normalize_d_vec(centers[i], nr);
	}

	delete_2d_float(2, nr, centers);
	delete_2d_float(2, nc, cosines);
	delete[] dcosines;
}

#define GETI(i) (y[i] + 1)
typedef signed char schar;

void solve_l2r_lr_dual(SMatF *X_Xf, _int *y, _float *w, _float eps, _float Cp, _float Cn, _int classifier_maxitr)
{
	_int l = X_Xf->nc;
	_int w_size = X_Xf->nr;
	_int i, s, iter = 0;

	_double *xTx = new _double[l];
	_int max_iter = classifier_maxitr;
	_int *index = new _int[l];
	_double *alpha = new _double[2 * l]; // store alpha and C - alpha
	_int max_inner_iter = 100;			 // for inner Newton
	_double innereps = 1e-2;
	_double innereps_min = min(1e-8, (_double)eps);
	_double upper_bound[3] = {Cn, 0, Cp};

	_int *size = X_Xf->size;
	pairIF **data = X_Xf->data;

	// Initial alpha can be set here. Note that
	for (i = 0; i < l; i++)
	{
		alpha[2 * i] = min(0.001 * upper_bound[GETI(i)], 1e-8);
		alpha[2 * i + 1] = upper_bound[GETI(i)] - alpha[2 * i];
	}

	for (i = 0; i < w_size; i++)
		w[i] = 0;

	for (i = 0; i < l; i++)
	{
		xTx[i] = sparse_operator::nrm2_sq(size[i], data[i]);
		sparse_operator::axpy(y[i] * alpha[2 * i], size[i], data[i], w);
		index[i] = i;
	}

	while (iter < max_iter)
	{
		for (i = 0; i < l; i++)
		{
			_int j = i + get_rand_num(l - i);
			swap(index[i], index[j]);
		}

		_int newton_iter = 0;
		_double Gmax = 0;
		for (s = 0; s < l; s++)
		{
			i = index[s];
			const _int yi = y[i];
			_double C = upper_bound[GETI(i)];
			_double ywTx = 0, xisq = xTx[i];
			ywTx = yi * sparse_operator::dot(w, size[i], data[i]);
			_double a = xisq, b = ywTx;

			// Decide to minimize g_1(z) or g_2(z)
			_int ind1 = 2 * i, ind2 = 2 * i + 1, sign = 1;
			if (0.5 * a * (alpha[ind2] - alpha[ind1]) + b < 0)
			{
				ind1 = 2 * i + 1;
				ind2 = 2 * i;
				sign = -1;
			}

			_double alpha_old = alpha[ind1];
			_double z = alpha_old;
			if (C - z < 0.5 * C)
				z = 0.1 * z;
			_double gp = a * (z - alpha_old) + sign * b + log(z / (C - z));
			Gmax = max(Gmax, fabs(gp));

			// Newton method on the sub-problem
			const _double eta = 0.1; // xi in the paper
			_int inner_iter = 0;
			while (inner_iter <= max_inner_iter)
			{
				if (fabs(gp) < innereps)
					break;
				_double gpp = a + C / (C - z) / z;
				_double tmpz = z - gp / gpp;
				if (tmpz <= 0)
					z *= eta;
				else // tmpz in (0, C)
					z = tmpz;
				gp = a * (z - alpha_old) + sign * b + log(z / (C - z));
				newton_iter++;
				inner_iter++;
			}

			if (inner_iter > 0) // update w
			{
				alpha[ind1] = z;
				alpha[ind2] = C - z;
				sparse_operator::axpy(sign * (z - alpha_old) * yi, size[i], data[i], w);
			}
		}

		iter++;

		if (Gmax < eps)
			break;

		if (newton_iter <= l / 10)
			innereps = max(innereps_min, 0.1 * innereps);
	}

	delete[] xTx;
	delete[] alpha;
	delete[] index;
}

void solve_l2r_l1l2_svc(SMatF *X_Xf, _int *y, _float *w, _float eps, _float Cp, _float Cn, _int classifier_maxitr)
{
	_int l = X_Xf->nc;
	_int w_size = X_Xf->nr;

	_int i, s, iter = 0;
	_float C, d, G;
	_float *QD = new _float[l];
	_int max_iter = classifier_maxitr;
	_int *index = new _int[l];
	_float *alpha = new _float[l];
	_int active_size = l;

	_int tot_iter = 0;

	// PG: projected gradient, for shrinking and stopping
	_float PG;
	_float PGmax_old = INF;
	_float PGmin_old = -INF;
	_float PGmax_new, PGmin_new;

	// default solver_type: L2R_L2LOSS_SVC_DUAL
	_float diag[3] = {(_float)0.5 / Cn, (_float)0, (_float)0.5 / Cp};
	_float upper_bound[3] = {INF, 0, INF};

	_int *size = X_Xf->size;
	pairIF **data = X_Xf->data;

	//d = pwd;
	//Initial alpha can be set here. Note that
	// 0 <= alpha[i] <= upper_bound[GETI(i)]

	for (i = 0; i < l; i++)
		alpha[i] = 0;

	for (i = 0; i < w_size; i++)
		w[i] = 0;

	for (i = 0; i < l; i++)
	{
		QD[i] = diag[GETI(i)];
		QD[i] += sparse_operator::nrm2_sq(size[i], data[i]);
		sparse_operator::axpy(y[i] * alpha[i], size[i], data[i], w);
		index[i] = i;
	}

	while (iter < max_iter)
	{
		PGmax_new = -INF;
		PGmin_new = INF;

		for (i = 0; i < active_size; i++)
		{
			_int j = i + get_rand_num(active_size - i);
			swap(index[i], index[j]);
		}

		for (s = 0; s < active_size; s++)
		{
			tot_iter++;

			i = index[s];
			const _int yi = y[i];

			G = yi * sparse_operator::dot(w, size[i], data[i]) - 1;

			C = upper_bound[GETI(i)];
			G += alpha[i] * diag[GETI(i)];

			PG = 0;
			if (alpha[i] == 0)
			{
				if (G > PGmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
				else if (G < 0)
					PG = G;
			}
			else if (alpha[i] == C)
			{
				if (G < PGmin_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
				else if (G > 0)
					PG = G;
			}
			else
				PG = G;

			PGmax_new = max(PGmax_new, PG);
			PGmin_new = min(PGmin_new, PG);

			if (fabs(PG) > 1.0e-12)
			{
				_float alpha_old = alpha[i];
				alpha[i] = min(max(alpha[i] - G / QD[i], (_float)0.0), C);
				d = (alpha[i] - alpha_old) * yi;
				sparse_operator::axpy(d, size[i], data[i], w);
			}
		}

		iter++;

		if (PGmax_new - PGmin_new <= eps)
		{
			if (active_size == l)
				break;
			else
			{
				active_size = l;
				PGmax_old = INF;
				PGmin_old = -INF;
				continue;
			}
		}
		PGmax_old = PGmax_new;
		PGmin_old = PGmin_new;
		if (PGmax_old <= 0)
			PGmax_old = INF;
		if (PGmin_old >= 0)
			PGmin_old = -INF;
	}

	// calculate objective value

	delete[] QD;
	delete[] alpha;
	delete[] index;
}

SMatF *svms(SMatF *trn_X_Xf, SMatF *trn_Y_X, Param &param)
{
	_float eps = 0.1;
	_float f;
	if (param.classifier_kind == L2R_L2LOSS_SVC)
		f = 1.0;
	else if (param.classifier_kind == L2R_LR)
		f = (_float)param.num_trn / (_float)trn_X_Xf->nc;
	_float Cp = param.classifier_cost * f;
	_float Cn = param.classifier_cost * f;
	_float th = param.classifier_threshold;

	_int num_Y = trn_Y_X->nc;
	_int num_X = trn_X_Xf->nc;
	_int num_Xf = trn_X_Xf->nr;

	_int *y = new _int[num_X];
	fill(y, y + num_X, -1);

	SMatF *w_mat = new SMatF(num_Xf, num_Y);
	_float *w = new _float[num_Xf];

	for (_int l = 0; l < num_Y; l++)
	{
		for (_int i = 0; i < trn_Y_X->size[l]; i++)
			y[trn_Y_X->data[l][i].first] = +1;

		if (param.classifier_kind == L2R_L2LOSS_SVC)
			solve_l2r_l1l2_svc(trn_X_Xf, y, w, eps, Cp, Cn, param.classifier_maxitr);
		else if (param.classifier_kind == L2R_LR)
			solve_l2r_lr_dual(trn_X_Xf, y, w, eps, Cp, Cn, param.classifier_maxitr);

		w_mat->data[l] = new pairIF[num_Xf]();
		_int siz = 0;
		for (_int f = 0; f < num_Xf; f++)
		{
			if (fabs(w[f]) > th)
				w_mat->data[l][siz++] = make_pair(f, w[f]);
		}
		Realloc(num_Xf, siz, w_mat->data[l]);
		w_mat->size[l] = siz;

		for (_int i = 0; i < trn_Y_X->size[l]; i++)
			y[trn_Y_X->data[l][i].first] = -1;
	}

	delete[] y;
	delete[] w;

	return w_mat;
}

void reindex_rows(SMatF *mat, _int nr, VecI &rows)
{
	mat->nr = nr;
	for (_int i = 0; i < mat->nc; i++)
	{
		for (_int j = 0; j < mat->size[i]; j++)
			mat->data[i][j].first = rows[mat->data[i][j].first];
	}
}

SMatF *partition_to_assign_mat(SMatF *Y_X, VecI &partition)
{
	_int num_Y = Y_X->nc;
	_int num_X = Y_X->nr;

	VecI pos_Y, neg_Y;
	for (_int i = 0; i < num_Y; i++)
	{
		if (partition[i] == 1)
			pos_Y.push_back(i);
		else
			neg_Y.push_back(i);
	}

	_int pos_nnz = 0, neg_nnz = 0;
	VecI pos_X, neg_X, pos_counts, neg_counts;
	Y_X->active_dims(pos_Y, pos_X, pos_counts, countmap);
	Y_X->active_dims(neg_Y, neg_X, neg_counts, countmap);

	for (_int i = 0; i < pos_counts.size(); i++)
		pos_nnz += pos_counts[i];

	for (_int i = 0; i < neg_counts.size(); i++)
		neg_nnz += neg_counts[i];

	SMatF *assign_mat = new SMatF(num_X, 2);
	_int *size = assign_mat->size;
	pairIF **data = assign_mat->data;

	size[0] = pos_X.size();
	size[1] = neg_X.size();
	data[0] = new pairIF[pos_X.size()];
	data[1] = new pairIF[neg_X.size()];

	for (_int i = 0; i < pos_X.size(); i++)
		data[0][i] = make_pair(pos_X[i], 1);

	for (_int i = 0; i < neg_X.size(); i++)
		data[1][i] = make_pair(neg_X[i], 1);

	return assign_mat;
}

void shrink_data_matrices(SMatF *trn_X_Xf, SMatF *trn_Y_X, SMatF *cent_mat, VecI &n_Y, Param &param, SMatF *&n_trn_X_Xf, SMatF *&n_trn_Y_X, SMatF *&n_cent_mat, VecI &n_X, VecI &n_Xf, VecI &n_cXf)
{
	trn_Y_X->shrink_mat(n_Y, n_trn_Y_X, n_X, countmap, false);
	trn_X_Xf->shrink_mat(n_X, n_trn_X_Xf, n_Xf, countmap, false);
	cent_mat->shrink_mat(n_Y, n_cent_mat, n_cXf, countmap, false);
}

void train_leaf_svms(Node *node, SMatF *X_Xf, SMatF *Y_X, _int nr, VecI &n_Xf, Param &param)
{
	SMatF *w_mat = svms(X_Xf, Y_X, param);
	reindex_rows(w_mat, nr, n_Xf);
	node->w = w_mat;
}

void split_node(Node *node, SMatF *X_Xf, SMatF *Y_X, SMatF *cent_mat, _int nr, VecI &n_Xf, VecI &partition, Param &param)
{
	balanced_kmeans(cent_mat, param.clustering_eps, partition);
	SMatF *assign_mat = partition_to_assign_mat(Y_X, partition);
	SMatF *w_mat = svms(X_Xf, assign_mat, param);
	reindex_rows(w_mat, nr, n_Xf);
	node->w = w_mat;
	delete assign_mat;
}

Tree *train_tree(SMatF *trn_X_Xf, SMatF *trn_Y_X, SMatF *cent_mat, Param &param, _int tree_no)
{
	reng.seed(tree_no);
	_int num_X = trn_X_Xf->nc;
	_int num_Xf = trn_X_Xf->nr;
	_int num_Y = trn_Y_X->nc;

	_int max_depth = ceil(log2(num_Y / param.max_leaf)) + 1;

	Tree *tree = new Tree;
	vector<Node *> &nodes = tree->nodes;

	Node *root = init_root(num_Y, max_depth);
	nodes.push_back(root);

	for (_int i = 0; i < nodes.size(); i++)
	{
		if (i % 100 == 0)
			cout << "node " << i << endl;

		Node *node = nodes[i];
		VecI &n_Y = node->Y;
		SMatF *n_trn_X_Xf;
		SMatF *n_trn_Y_X;
		SMatF *n_cent_mat;
		VecI n_X;
		VecI n_Xf;
		VecI n_cXf;

		shrink_data_matrices(trn_X_Xf, trn_Y_X, cent_mat, n_Y, param, n_trn_X_Xf, n_trn_Y_X, n_cent_mat, n_X, n_Xf, n_cXf);

		if (node->is_leaf)
		{
			train_leaf_svms(node, n_trn_X_Xf, n_trn_Y_X, num_Xf, n_Xf, param);
		}
		else
		{
			VecI partition;
			split_node(node, n_trn_X_Xf, n_trn_Y_X, n_cent_mat, num_Xf, n_Xf, partition, param);

			VecI pos_Y, neg_Y;
			for (_int j = 0; j < n_Y.size(); j++)
				if (partition[j])
					pos_Y.push_back(n_Y[j]);
				else
					neg_Y.push_back(n_Y[j]);

			Node *pos_node = new Node(pos_Y, node->depth + 1, max_depth);
			nodes.push_back(pos_node);
			node->pos_child = nodes.size() - 1;

			Node *neg_node = new Node(neg_Y, node->depth + 1, max_depth);
			nodes.push_back(neg_node);
			node->neg_child = nodes.size() - 1;
		}

		delete n_trn_X_Xf;
		delete n_trn_Y_X;
		delete n_cent_mat;
	}
	tree->num_Xf = num_Xf;
	tree->num_Y = num_Y;

	return tree;
}

void train_trees_thread(SMatF *trn_X_Xf, SMatF *trn_Y_X, SMatF *cent_mat, Param param, _int s, _int t, string model_dir, _float *train_time)
{
	Timer timer;
	timer.tic();
	_int num_X = trn_X_Xf->nc;
	_int num_Xf = trn_X_Xf->nr;
	_int num_Y = trn_Y_X->nc;
	setup_thread_locals(num_X, num_Xf, num_Y);
	{
		lock_guard<mutex> lock(mtx);
		*train_time += timer.toc();
	}

	for (_int i = s; i < s + t; i++)
	{
		timer.tic();
		cout << "tree " << i << " training started" << endl;

		Tree *tree = train_tree(trn_X_Xf, trn_Y_X, cent_mat, param, i);
		{
			lock_guard<mutex> lock(mtx);
			*train_time += timer.toc();
		}

		tree->write(model_dir, i);

		timer.tic();
		delete tree;

		cout << "tree " << i << " training completed" << endl;

		{
			lock_guard<mutex> lock(mtx);
			*train_time += timer.toc();
		}
	}
}

void train_trees(SMatF *trn_X_Xf, SMatF *trn_X_Y, Param &param, string model_dir, _float &train_time)
{
	_float *t_time = new _float;
	*t_time = 0;
	Timer timer;

	timer.tic();
	param.num_trn = trn_X_Xf->nc;
	trn_X_Xf->unit_normalize_columns();
	SMatF *trn_Y_X = trn_X_Y->transpose();
	SMatF *cent_mat = trn_X_Xf->prod(trn_Y_X);
	cent_mat->unit_normalize_columns();
	cent_mat->threshold(param.centroid_threshold);
	trn_X_Xf->append_bias_feat(param.bias_feat);

	_int tree_per_thread = (_int)ceil((_float)param.num_tree / param.num_thread);
	vector<thread> threads;
	_int s = param.start_tree;
	for (_int i = 0; i < param.num_thread; i++)
	{
		if (s < param.start_tree + param.num_tree)
		{
			_int t = min(tree_per_thread, param.start_tree + param.num_tree - s);
			threads.push_back(thread(train_trees_thread, trn_X_Xf, trn_Y_X, cent_mat, param, s, t, model_dir, ref(t_time)));
			s += t;
		}
	}
	*t_time += timer.toc();

	for (_int i = 0; i < threads.size(); i++)
		threads[i].join();

	timer.tic();
	delete trn_Y_X;
	delete cent_mat;

	*t_time += timer.toc();
	train_time = *t_time;
	delete t_time;
}

thread_local float *densew;
void update_svm_scores(Node *node, SMatF *tst_X_Xf, SMatF *score_mat, _Classifier_Kind classifier_kind)
{
	SMatF *w_mat = node->w;
	_int num_svm = w_mat->nc;

	for (_int i = 0; i < num_svm; i++)
	{
		_int target;
		if (node->is_leaf)
			target = node->Y[i];
		else
		{
			if (i == 0)
				target = node->pos_child;
			else
				target = node->neg_child;
		}

		set_d_with_s(w_mat->data[i], w_mat->size[i], densew);

		VecIF &X = node->X;
		for (_int j = 0; j < X.size(); j++)
		{
			_int inst = X[j].first;
			_float oldvalue = X[j].second;
			_float prod = mult_d_s_vec(densew, tst_X_Xf->data[inst], tst_X_Xf->size[inst]);
			_float newvalue;
			if (classifier_kind == L2R_L2LOSS_SVC)
				newvalue = -SQ(max((_float)0.0, 1 - prod));
			else if (classifier_kind == L2R_LR)
				newvalue = -log(1 + exp(-prod));

			newvalue += oldvalue;
			score_mat->data[inst][score_mat->size[inst]++] = make_pair(target, newvalue);
		}

		reset_d_with_s(w_mat->data[i], w_mat->size[i], densew);
	}
}

void update_next_level(_int b, vector<Node *> &nodes, SMatF *score_mat, Param &param)
{
	if (b < nodes.size() - 1 && nodes[b + 1]->depth > nodes[b]->depth)
	{
		_int *size = score_mat->size;
		pairIF **data = score_mat->data;
		_int beam_width = param.beam_width;
		_int num_X = score_mat->nc;

		for (_int i = 0; i < num_X; i++)
		{
			if (size[i] > beam_width)
			{
				sort(data[i], data[i] + size[i], comp_pair_by_second_desc<_int, _float>);
				size[i] = beam_width;
			}
			for (_int j = 0; j < size[i]; j++)
			{
				Node *node = nodes[data[i][j].first];
				node->X.push_back(make_pair(i, data[i][j].second));
			}
			size[i] = 0;
		}
	}
}

void exponentiate_scores(SMatF *mat)
{
	_int nc = mat->nc;
	_int *size = mat->size;
	pairIF **data = mat->data;

	for (_int i = 0; i < nc; i++)
	{
		for (_int j = 0; j < size[i]; j++)
			data[i][j].second = exp(data[i][j].second);

		sort(data[i], data[i] + size[i], comp_pair_by_first<_int, _float>);
	}
}

SMatF *predict_tree(SMatF *tst_X_Xf, Tree *tree, Param &param)
{
	_int num_X = tst_X_Xf->nc;
	_int num_Y = param.num_Y;
	_int beam_width = param.beam_width;
	_int max_leaf = param.max_leaf;
	vector<Node *> &nodes = tree->nodes;
	_int num_node = nodes.size();

	SMatF *score_mat = new SMatF(num_node, num_X);
	for (_int i = 0; i < num_X; i++)
	{
		score_mat->size[i] = 0;
		score_mat->data[i] = new pairIF[beam_width * max_leaf];
	}

	Node *node = nodes[0];
	node->X.clear();

	for (_int i = 0; i < num_X; i++)
		node->X.push_back(make_pair(i, 0));

	for (_int i = 0; i < nodes.size(); i++)
	{
		if (i % 100 == 0)
			cout << "node " << i << endl;
		Node *node = nodes[i];
		update_svm_scores(node, tst_X_Xf, score_mat, param.classifier_kind);
		update_next_level(i, nodes, score_mat, param);
	}

	exponentiate_scores(score_mat);
	score_mat->nr = num_Y;

	return score_mat;
}

void predict_trees_thread(SMatF *tst_X_Xf, SMatF *score_mat, Param param, _int s, _int t, string model_dir, _float *prediction_time, _float *model_size)
{
	Timer timer;

	timer.tic();
	densew = new _float[tst_X_Xf->nr + 1]();
	{
		lock_guard<mutex> lock(mtx);
		*prediction_time += timer.toc();
	}

	for (_int i = s; i < s + t; i++)
	{
		cout << "tree " << i << " predicting started" << endl;
		Tree *tree = new Tree(model_dir, i);

		timer.tic();
		SMatF *tree_score_mat = predict_tree(tst_X_Xf, tree, param);

		{
			lock_guard<mutex> lock(mtx);
			score_mat->add(tree_score_mat);
			*model_size += tree->get_ram();
		}

		delete tree;
		delete tree_score_mat;

		cout << "tree " << i << " predicting completed" << endl;
		{
			lock_guard<mutex> lock(mtx);
			*prediction_time += timer.toc();
		}
	}

	timer.tic();
	delete[] densew;
	{
		lock_guard<mutex> lock(mtx);
		*prediction_time += timer.toc();
	}
}

SMatF *predict_trees(SMatF *tst_X_Xf, Param &param, string model_dir, _float &prediction_time, _float &model_size)
{
	_float *p_time = new _float;
	*p_time = 0;

	_float *m_size = new _float;
	*m_size = 0;

	Timer timer;

	timer.tic();
	tst_X_Xf->unit_normalize_columns();
	tst_X_Xf->append_bias_feat(param.bias_feat);

	_int num_X = tst_X_Xf->nc;
	_int num_Y = param.num_Y;
	_int beam_width = param.beam_width;

	SMatF *score_mat = new SMatF(num_Y, num_X);

	_int tree_per_thread = (_int)ceil((_float)param.num_tree / param.num_thread);
	vector<thread> threads;

	_int s = param.start_tree;
	for (_int i = 0; i < param.num_thread; i++)
	{
		if (s < param.start_tree + param.num_tree)
		{
			_int t = min(tree_per_thread, param.start_tree + param.num_tree - s);
			threads.push_back(thread(predict_trees_thread, tst_X_Xf, ref(score_mat), param, s, t, model_dir, ref(p_time), ref(m_size)));
			s += t;
		}
	}
	*p_time += timer.toc();

	for (_int i = 0; i < threads.size(); i++)
		threads[i].join();

	timer.tic();
	for (_int i = 0; i < score_mat->nc; i++)
		for (_int j = 0; j < score_mat->size[i]; j++)
			score_mat->data[i][j].second /= param.num_tree;

	*p_time += timer.toc();
	prediction_time = *p_time;
	delete p_time;

	model_size = *m_size;
	delete m_size;

	for (_int i = 0; i < score_mat->nc; i++)
	{
		_int siz = score_mat->size[i];
		sort(score_mat->data[i], score_mat->data[i] + siz, comp_pair_by_second_desc<_int, _float>);
		_int newsiz = min(siz, 100);
		Realloc(siz, newsiz, score_mat->data[i]);
		score_mat->size[i] = newsiz;
		sort(score_mat->data[i], score_mat->data[i] + newsiz, comp_pair_by_first<_int, _float>);
	}

	return score_mat;
}

void train(MatrixXdR ft_file, MatrixXdR lbl_file, string model_dir,
		   _int num_thread, _int start_tree, _int num_tree,
		   _float bias_feat, _float classifier_cost, _int max_leaf,
		   _float classifier_threshold, _float centroid_threshold,
		   _float clustering_eps, _int classifier_maxitr,
		   _int classifier_kind, _bool quiet)
{
	std::ios_base::sync_with_stdio(false);

	SMatF *trn_X_Xf = SPARSE2SMAT(&ft_file);
	SMatF *trn_X_Y = SPARSE2SMAT(&lbl_file);

	check_valid_foldername(model_dir);
	Param param = Param();
	param.num_thread = num_thread;
	param.start_tree = start_tree;
	param.num_tree = num_tree;
	param.bias_feat = bias_feat;
	param.classifier_cost = classifier_cost;
	param.max_leaf = max_leaf;
	param.classifier_threshold = classifier_threshold;
	param.centroid_threshold = centroid_threshold;
	param.clustering_eps = clustering_eps;
	param.classifier_maxitr = classifier_maxitr;
	param.classifier_kind = (_Classifier_Kind)classifier_kind;
	param.quiet = quiet;
	param.num_Xf = trn_X_Xf->nr;
	param.num_Y = trn_X_Y->nr;
	param.write(model_dir + "/param");

	_float train_time;
	train_trees(trn_X_Xf, trn_X_Y, param, model_dir, train_time);
	cout << "Training time: " << (train_time / 3600.0) << " hr" << endl;

	delete trn_X_Xf;
	delete trn_X_Y;
}

MatrixXdR *predict(MatrixXdR ft_file, string model_dir, _int num_thread,
				   _int start_tree, _int num_tree, _bool quiet)
{
	std::ios_base::sync_with_stdio(false);
	SMatF *tst_X_Xf = SPARSE2SMAT(&ft_file);
	check_valid_foldername(model_dir);
	Param param = Param(model_dir + "/param");
	param.num_thread = num_thread;
	param.start_tree = start_tree;
	param.num_tree = num_tree;
	param.quiet = quiet;

	_float prediction_time, model_size;
	SMatF *score_mat = predict_trees(tst_X_Xf, param, model_dir, prediction_time, model_size);
	
	cout << "prediction time: " << 1000 * (prediction_time / tst_X_Xf->nc) << " ms/points" << endl;
	cout << "model size: " << model_size / 1e+9 << " GB" << endl;
	return SMAT2SPARSE(score_mat);
}

PYBIND11_MODULE(parabel, m)
{
	m.doc() = "Parabel Python plugin"; // optional module docstring

	m.def("train", &train, "Function to train parabel", py::arg("ft_file"), py::arg("lbl_file"),
		  py::arg("model_dir"), py::arg("num_thread")=3, py::arg("start_tree")=0,
		  py::arg("num_tree")=3, py::arg("bias_feat")=1.0, py::arg("classifier_cost")=1.0,
		  py::arg("max_leaf")=100, py::arg("classifier_threshold")=0.1, py::arg("centroid_threshold")=0,
		  py::arg("clustering_eps")=1e-4, py::arg("classifier_maxitr")=20,
		  py::arg("classifier_kind")=0, py::arg("quiet")=false);

	m.def("predict", &predict, "Function to predict from parabel", py::arg("ft_file"),
		  py::arg("model_dir"), py::arg("num_thread")=3, py::arg("start_tree")=0,
		  py::arg("num_tree")=3, py::arg("quiet")=false);
}