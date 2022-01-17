from ..utils import shortlist, sparse
from .base import BaseClassifier
import logging
import time
from ..data import data_loader


class KNNClassifier(BaseClassifier):
    """KNN classifier
    * brute or HNSW algorithm for search
    Parameters
    ----------
    method: str, optional, default='hnsw'
        brute or hnsw
    num_neighbours: int
        number of neighbors (same as efS)
        * may be useful if the NN search retrieve less number of labels
        * typically doesn't happen with HNSW etc.
    M: int, optional, default=100
        HNSW M (Usually 100)
    efC: int, optional, default=300
        construction parameter (Usually 300)
    efS: int, optional, default=300
        search parameter (Usually 300)
    num_threads: int, optional, default=18
        use multiple threads to cluster
    space: str, optional, default='cosine'
        metric to use while quering
    verbose: boolean, optional, default=True
        print progress
    norm: str or None, optional, default=None
        normalize features
    use_sparse: boolean. optional, default=False
        Not used; kept for future implementation
    """

    def __init__(self, method='hnsw', num_neighbours=300, M=100, efC=300,
                 efS=300, num_threads=12, space='cosine', verbose=False,
                 norm=None, use_sparse=False):
        super().__init__(verbose, False, use_sparse)
        self.norm = None
        self.num_labels = None
        self.num_labels_ = None
        self.valid_labels = None
        self.shorty = shortlist.construct_shortlist(
            method=method, num_neighbours=num_neighbours,
            M=M, efC=efC, efS=efS, order='instances',
            num_threads=num_threads,
            space=space)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('KNN')
        self.logger.info("Parameters:: {}".format(str(self)))

    def get_data_loader(self, data_dir, dataset, feat_fname,
                        label_fname, mode, batch_order):
        """Data loader
        - batch_order: 'label' during training
        - batch_order: 'instances' during prediction
        """
        return data_loader.Dataloader(
            batch_size=1,  # Dummy
            data_dir=data_dir,
            dataset=dataset,
            feat_fname=feat_fname,
            label_fname=label_fname,
            feature_type=self.feature_type,
            mode=mode,
            batch_order=batch_order,
            norm=self.norm,
            start_index=0,
            end_index=-1)

    def fit(self, data_dir, dataset, feat_fname, label_fname,
            model_dir, **kwargs):
        """Train the classifier
        Will create batches on labels and then parallelize
        - Not very efficient when training time per classifier is too low
        - Will not train for labels without any datapoints
          A list will be maintained which will used to remap labels
          to original ids
        Arguments:
        ---------
        data_dir: str
            data directory with all files
        dataset: str
            Name of the dataset; like EURLex-4K
        feat_fname: str
            File name of training feature file
            Should be in sparse format with header
        label_fname: str
            File name of training label file
            Should be in sparse format with header
        model_dir: str
            dump checkpoints in this directory
            based on save_after
        """
        data = self.get_data_loader(
            data_dir, dataset, feat_fname, label_fname, 'train', 'labels')
        self.num_labels = data.num_labels  # valid labels
        self.num_labels_ = data.num_labels_  # number of original labels
        self.valid_labels = data.valid_labels
        start_time = time.time()
        self.shorty.fit(data.features.data, data.labels.data)
        end_time = time.time()
        self.logger.info(
            "Train time (sec): {}, Model size (MB): {}".format(
                end_time-start_time, self.model_size))

    def predict(self, data_dir, dataset, feat_fname, label_fname, top_k=10):
        """Predict for given instances
        Will create batches on instance and then parallelize
        Arguments:
        ---------
        data_dir: str
            data directory with all files
        dataset: str
            Name of the dataset; like EURLex-4K
        feat_fname: str
            File name of training feature file
            Should be in sparse format with header
        label_fname: str
            File name of training label file
            Should be in sparse format with header
            TODO: Avoid sending labels as they are not used
        top_k: int, optional (default=10)
            retain only top_k values
        """
        data = self.get_data_loader(
            data_dir, dataset, feat_fname, label_fname, 'predict', 'instances')
        start_time = time.time()
        indices, scores = self.shorty.query(data.features.data)
        predicted = sparse.csr_from_arrays(
            indices, scores, shape=(data.num_instances, self.num_labels+1))
        predicted = sparse.retain_topk(predicted, k=top_k)
        end_time = time.time()
        self.logger.info(
            "Prediction time/sample (ms): {}".format(
                (end_time-start_time)*1000/data.num_instances))
        return self._map_to_original(predicted[:, :-1])

    def save(self, fname):
        super().save(fname)
        self.shorty.save(fname)

    def load(self, fname):
        super().load(fname)
        self.shorty.load(fname)

    def __repr__(self):
        return f"{self.shorty}"

    @property
    def model_size(self):
        return self.shorty.model_size

    def _map_to_original(self, X):
        """Some labels were removed during training as training data was
        not availale; remap to original mapping
        - Assumes documents need not be remapped
        """
        shape = (X.shape[0], self.num_labels_)
        return sparse._map_cols(X, self.valid_labels, shape)
