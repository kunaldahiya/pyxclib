import os
import xclib.classifier.parameters as parameters
from xclib.classifier import ova, slice, kcentroid, knn
from xclib.data import data_utils


def _get_fname(params):
    os.makedirs(params.model_dir, exist_ok=True)
    _ext = ""
    if params.start_index != 0 or params.end_index != -1:
        _ext = "_{}_{}".format(params.start_index, params.end_index)
    model_fname = os.path.join(params.model_dir, params.model_fname+_ext)
    return model_fname


def create_classifier(params):
    if params.clf_type == 'ova':
        return ova.OVAClassifier(solver='liblinear',
                                 loss='squared_hinge',
                                 C=params.C,
                                 tol=params.tol,
                                 verbose=0,
                                 batch_size=params.batch_size,
                                 norm=params.norm,
                                 num_threads=params.num_threads,
                                 max_iter=params.max_iter,
                                 threshold=params.threshold,
                                 feature_type=params.feature_type,
                                 dual=params.dual)
    if params.clf_type == 'slice':
        if params.feature_type == 'sparse':
            raise NotImplementedError("Slice not tested with sparse features.")
        return slice.Slice(solver='liblinear',
                           loss='squared_hinge',
                           C=params.C,
                           tol=params.tol,
                           verbose=0,
                           batch_size=params.batch_size,
                           norm=params.norm,
                           order='centroids',
                           M=params.M,
                           efC=params.efC,
                           efS=params.efS,
                           num_neighbours=params.num_neighbours,
                           num_threads=params.num_threads,
                           max_iter=params.max_iter,
                           feature_type=params.feature_type,
                           dual=params.dual)
    if params.clf_type == 'kcentroid':
        if params.feature_type == 'sparse':
            raise NotImplementedError("KCentroid not tested with sparse features.")
        return kcentroid.KCentroidClassifier(
                           M=params.M,
                           efC=params.efC,
                           efS=params.efS,
                           num_neighbours=params.num_neighbours,
                           num_threads=params.num_threads)
    if params.clf_type == 'knn':
        if params.feature_type == 'sparse':
            raise NotImplementedError("KNN not tested with sparse features.")
        return knn.KNNClassifier(
                           M=params.M,
                           efC=params.efC,
                           efS=params.efS,
                           num_neighbours=params.num_neighbours,
                           num_threads=params.num_threads)
    else:
        raise NotImplementedError("Unknown classifier!")


def main(params):
    clf = create_classifier(params)
    # args.save('train_parameters.json')
    if params.mode == 'train':
        model_fname = _get_fname(params)
        clf.fit(data_dir=params.data_dir,
                dataset=params.dataset,
                model_dir=params.model_dir,
                feat_fname=params.tr_feat_fname,
                label_fname=params.tr_label_fname,
                save_after=1000)
        clf.save(model_fname)
    elif params.mode == 'predict':
        clf.load(os.path.join(params.model_dir, params.model_fname))
        predicted_labels = clf.predict(
            data_dir=params.data_dir,
            dataset=params.dataset,
            feat_fname=params.ts_feat_fname,
            label_fname=params.ts_label_fname)
        data_utils.write_sparse_file(
            predicted_labels,
            os.path.join(params.result_dir, 'score.txt'))
    else:
        raise NotImplementedError("Mode not implemented!")


if __name__ == '__main__':
    args = parameters.Parameters("Parameters")
    args.parse_args()
    main(args.params)
