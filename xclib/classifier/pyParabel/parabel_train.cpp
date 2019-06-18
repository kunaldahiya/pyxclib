#include <iostream>
#include <fstream>
#include <string>

#include "timer.h"
#include "parabel.h"

using namespace std;

void help()
{
	cerr<<"Sample Usage :"<<endl;
	cerr<<"./parabel_train [input feature file name] [input label file name] [output model folder name] -T 1 -s 0 -t 3 -b 1.0 -c 1.0 -m 100 -t 0.1 -tc 0 -e 0.0001 -n 20 -k 0 -q 0"<<endl<<endl;

	cerr<<"-T = param.num_thread				: Number of threads									default=1"<<endl;
	cerr<<"-s = param.start_tree				: Starting index of the trees								default=0"<<endl;
	cerr<<"-t = param.num_tree				: Number of trees to be grown								default=3"<<endl;
	cerr<<"-b = param.bias_feat				: Additional feature value to be appended to datapoint's features. Used for bias in linear separators similar to Liblinear.	default=1.0"<<endl;
	cerr<<"-c = param.classifier_cost			: Cost co-efficient for linear classifiers						default=1.0"<<endl;
	cerr<<"-m = param.max_leaf				: Maximum no. of labels in a leaf node. Larger nodes will be split into 2 balanced child nodes.		default=100"<<endl;
	cerr<<"-tcl = param.classifier_threshold			: Threshold value for sparsifying linear classifiers' trained weights to reduce model size.		default=0.1"<<endl;
	cerr<<"-tce = param.centroid_threshold			: Threshold value for sparsifying label centroids to speed up label clustering.		default=0"<<endl;
	cerr<<"-e = param.clustering_eps			: Eps value for terminating balanced spherical 2-Means clustering algorithm. Algorithm is terminated when successive iterations decrease objective by less than this value.	default=0.0001"<<endl;
	cerr<<"-n = param.classifier_maxiter			: Maximum iterations of algorithm for training linear classifiers			default=20"<<endl;
	cerr<<"-k = param.classifier_kind			: Kind of linear classifier to use. 0=L2R_L2LOSS_SVC, 1=L2R_LR (Refer to Liblinear)	default=L2R_L2LOSS_SVC"<<endl;
	cerr<<"-q = param.quiet				: Quiet option to restrict the output for reporting progress and debugging purposes 0=no quiet, 1=quiet		default=0"<<endl<<endl;

	cerr<<"The feature and label input files are expected to be in sparse matrix text format. Refer to README.txt for more details."<<endl;
	exit(1);
}

Param parse_param(_int argc, char* argv[])
{
	Param param;
	string opt;
	string sval;
	_float val;

	for(_int i=0; i<argc; i+=2)
	{
		opt = string(argv[i]);
		sval = string(argv[i+1]);
		val = stof(sval);

		if( opt == "-T" ) 
			param.num_thread = (_int)val;
		else if( opt == "-s" )
			param.start_tree = (_int)val;
		else if( opt == "-t" )
			param.num_tree = (_int)val;
		else if( opt == "-b" )
			param.bias_feat = (_float)val;
		else if( opt == "-c" )
			param.classifier_cost = (_float)val;
		else if( opt == "-m" )
			param.max_leaf = (_int)val;
		else if( opt == "-tcl" )
			param.classifier_threshold = (_float)val;
		else if( opt == "-tce" )
			param.centroid_threshold = (_float)val;
		else if( opt == "-e" )
			param.clustering_eps = (_float)val;
		else if( opt == "-n" )
			param.classifier_maxitr = (_int)val;
		else if( opt == "-k" )
			param.classifier_kind = (_Classifier_Kind)((_int)val);
		else if( opt == "-q" )
			param.quiet = (_bool)(val);
	}

	return param;
}

int main(int argc, char* argv[])
{
	std::ios_base::sync_with_stdio(false);

	if(argc < 4)
		help();

	string ft_file = string( argv[1] );
	string lbl_file = string( argv[2] );
	string model_dir = string( argv[3] );
	check_valid_foldername( model_dir );

	Param param = parse_param( argc-4, argv+4 );

	train(ft_file, lbl_file, model_dir, param.num_thread, param.start_tree, \
		param.num_tree, param.bias_feat, param.classifier_cost, param.max_leaf, \
		param.classifier_threshold, param.centroid_threshold, param.clustering_eps, \
		param.classifier_maxitr, 0, param.quiet);
}
