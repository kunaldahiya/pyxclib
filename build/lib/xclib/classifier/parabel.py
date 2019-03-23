from .so import parabel
class Parabel:
	def __init__(self, model_dir, num_tree=3, num_thread=5,
				 start_tree=0, bias_feat=1.0, classifier_cost=1.0,
				 max_leaf=100, classifier_threshold=0.1,
				 centroid_threshold=0, clustering_eps=1e-4,
				 classifier_maxitr=20, classifier_kind=0, quiet=False):	
		self.classifier = parabel
		self.model_dir = model_dir
		self.num_tree = num_tree
		self.num_thread = num_thread
		self.start_tree = start_tree
		self.bias_feat = bias_feat
		self.classifier_cost = classifier_cost
		self.max_leaf = max_leaf
		self.classifier_threshold = classifier_threshold
		self.centroid_threshold = centroid_threshold
		self.clustering_eps = clustering_eps
		self.classifier_maxitr = classifier_maxitr
		self.classifier_kind = classifier_kind
		self.quiet = quiet

	def fit(self, X, y):
		print("yay")
		self.classifier.train(ft_file=X, lbl_file=y, model_dir=self.model_dir, num_tree=self.num_tree,
							  num_thread=self.num_thread, start_tree=self.start_tree,
							  bias_feat=self.bias_feat, classifier_cost=self.classifier_cost,
							  max_leaf=self.max_leaf, classifier_threshold=self.classifier_threshold,
							  centroid_threshold=self.centroid_threshold, clustering_eps=self.clustering_eps,
							  classifier_maxitr=self.classifier_maxitr,
							  classifier_kind=self.classifier_kind, quiet=self.quiet)

	def predict(self, X):
		return self.classifier.predict(ft_file=X, model_dir=self.model_dir, num_tree=self.num_tree,
									   num_thread=self.num_thread, start_tree=self.start_tree,quiet=self.quiet)
