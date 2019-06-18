** Please make sure that you read and agree to the terms of license (License.pdf) and copyright (liblinear_COPYRIGHT) before using this software. **

This is the code for the algorithm proposed in our research paper "Parabel: Partitioned Label Trees for Extreme Classification with Appplication to Dynamic Search Advertising" authored by Yashoteja Prabhu, Anil Kag, Shrutendra Harsola, Rahul Agrawal and Manik Varma and published at The Web Conference-2018. The code is authored by Yashoteja Prabhu (yashoteja.prabhu@gmail.com).

About Parabel
=============
The objective in extreme multi-label learning is to learn a classifier that can automatically tag a datapoint with the most relevant subset of labels from an extremely large label space. Parabel is an efficient tree ensemble based extreme classifier that achieves close to state-of-the-art accuracies while being significantly faster to train and predict than most other extreme classifiers. Parabel can train on millions of labels and datapoints within a few hours on a single core of a desktop and make predictions in milliseconds per test point. Parabel's model sizes are significantly smaller than other tree based methods such as FastXML/PfastreXML. Please refer to the research paper for more details.

This code is made available as is for non-commercial research purposes. Please make sure that you have read the license agreement in LICENSE.doc/pdf. Please do not install or use Parabel unless you agree to the terms of the license.

The code for Parabel is written in C++ and should compile on 64 bit Windows/Linux machines using a C++11 enabled compiler. Matlab wrappers have also been provided with the code. Installation and usage instructions are provided below. The default parameters provided in the Usage Section work reasonably on the benchmark datasets in the Extreme Classification Repository (http://manikvarma.org/downloads/XC/XMLRepository.html). 

Please contact Yashoteja Prabhu (yashoteja.prabhu@gmail.com) and Manik Varma (manik@microsoft.com) if you have any questions or feedback.

Experimental Results and Datasets
=================================
Please visit the Extreme Classification Repository (http://manikvarma.org/downloads/XC/XMLRepository.html) to download the benchmark datasets and compare Parabel's performance to baseline algorithms.

Usage
=====
Linux/Windows makefiles for compiling Parabel have been provided with the source code. To compile, run "make" (Linux) or "nmake -f Makefile.win" (Windows) in the topmost folder. Run the following commands from inside Parabel folder for training and testing. To use Matlab scripts, compile the mex files in 'Tools/matlab' folder by running "make".

Training
--------

C++:
	./parabel_train [input feature file name] [input label file name] [output model folder name] -T 1 -s 0 -t 3 -b 1.0 -c 1.0 -m 100 -tcl 0.1 -tce 0 -e 0.0001 -n 20 -k 0 -q 0

Matlab:
	parabel_train([input feature matrix], [input label matrix], [output model folder name], param)

where:
	-T = param.num_thread				: Number of threads									default=1
	-s = param.start_tree				: Starting index of the trees								default=0
	-t = param.num_tree				: Number of trees to be grown								default=3
	-b = param.bias_feat				: Additional feature value to be appended to datapoint's features. Used for bias in linear separators similar to Liblinear.	default=1.0
	-c = param.classifier_cost			: Cost co-efficient for linear classifiers						default=1.0
	-m = param.max_leaf				: Maximum no. of labels in a leaf node. Larger nodes will be split into 2 balanced child nodes.		default=100
	-tcl = param.classifier_threshold			: Threshold value for sparsifying linear classifiers' trained weights to reduce model size.		default=0.1
	-tce = param.centroid_threshold			: Threshold value for sparsifying label centroids to speed up label clustering.		default=0
	-e = param.clustering_eps			: Eps value for terminating balanced spherical 2-Means clustering algorithm. Algorithm is terminated when successive iterations decrease objective by less than this value.	default=0.0001
	-n = param.classifier_maxitr			: Maximum iterations of algorithm for training linear classifiers			default=20
	-k = param.classifier_kind			: Kind of linear classifier to use. 0=L2R_L2LOSS_SVC, 1=L2R_LR (Refer to Liblinear)	default=L2R_L2LOSS_SVC
	-q = param.quiet				: Quiet option to restrict the output for reporting progress and debugging purposes 0=no quiet, 1=quiet		default=0

	For C++, the feature and label input files are expected to be in sparse matrix text format (refer to Miscellaneous section). For Matlab, the feature and label matrices are Matlab's sparse matrices.

Testing
-------

C++:
	./parabel_predict [input feature file name] [input model folder name] [output score file name] -T 1 -s 0 -t 3 -B 10 -q 0

Matlab:
	output_score_mat = parabel_predict( [input feature matrix], [input model folder name], param )

where:
	-T = param.num_thread				: Number of threads									default=[value saved in trained model]
	-s = param.start_tree				: Starting index of the trees for prediction								default=[value saved in trained model]
	-t = param.num_tree				: Number of trees to be used for prediction								default=[value saved in trained model]
	-B = param.beam_width				: Beam search width for fast, approximate prediction					default=10
	-q = param.quiet				: Quiet option to restrict the output for reporting progress and debugging purposes 0=no quiet, 1=quiet		default=[value saved in trained model]

	For C++, the feature and score files are expected to be in sparse matrix text format (refer to Miscellaneous section). For Matlab, the feature and score matrices are Matlab's sparse matrices.

Performance Evaluation
----------------------

Scripts for performance evaluation are only available in Matlab. To compile these scripts, execute "make" in the topmost folder from the Matlab terminal.
Following command is executed from Tools/metrics folder and outputs a struct containing all the metrics:

	[metrics] = get_all_metrics([test score matrix], [test label matrix], [inverse label propensity vector])

Miscellaneous
-------------

* The data format required by Parabel for feature and label input files is different from the format used in the repository datasets. The first line contains the number of rows and columns and the subsequent lines contain one data instance per row with the field index starting from 0. To convert from the repository format to Parabel format, run the following command from 'Tools' folder:

    	perl convert_format.pl [repository data file] [output feature file name] [output label file name]

* Scripts are provided in the 'Tools' folder for sparse matrix inter conversion between Matlab .mat format and text format.
    To read a text matrix into Matlab:

    	[matrix] = read_text_mat([text matrix name]); 

    To write a Matlab matrix into text format:

    	write_text_mat([Matlab sparse matrix], [text matrix name to be written to]);

* To generate inverse label propensity weights, run the following command inside 'Tools/metrics' folder on Matlab terminal:

    	[weights vector] = inv_propensity([training label matrix],A,B); 

    A,B are the parameters of the inverse propensity model. Following values are to be used over the benchmark datasets:

    	Wikipedia-LSHTC: A=0.5,  B=0.4
    	Amazon:          A=0.6,  B=2.6
    	Other:		 A=0.55, B=1.5

Toy Example
===========

The zip file containing the source code also includes the EUR-Lex dataset as a toy example.
To run Parabel on the EUR-Lex dataset, execute "bash sample_run.sh" (Linux) or "sample_run" (Windows and Matlab) in the Parabel folder.
Read the comments provided in the above scripts for better understanding.

