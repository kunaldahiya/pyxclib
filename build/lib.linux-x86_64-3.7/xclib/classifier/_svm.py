from sklearn.svm import LinearSVC
import numpy as np


def apply_threshold(data, threshold):
    data[np.where(np.abs(data) < threshold)] = 0

def train_one(data, loss, C, verbose, max_iter, threshold, dual, tol):
    def _get_features(obj):
        # Index samples iff they are required
        # Helful in reducing memory footprint
        if obj['ind'] is None:
            return obj['data']
        else:
            return obj['data'].take(obj['ind'], axis=0)
    X, y = _get_features(data), data['Y']
    clf = LinearSVC(tol=tol,
                    loss=loss,
                    dual=dual,
                    C=C,
                    multi_class='ovr',
                    fit_intercept=True,
                    intercept_scaling=1,
                    class_weight=None,
                    verbose=verbose,
                    random_state=0,
                    max_iter=max_iter)
    try:
        clf.fit(X, y)
        weight, bias = clf.coef_, clf.intercept_
    except ValueError:
        # TODO Find a solution for this; choose randomly may be?
        weight, bias = np.zeros((1, X.shape[1]), dtype=np.float32), np.zeros(
            (1), dtype=np.float32)
    del clf
    apply_threshold(weight, threshold)
    return weight, bias
    