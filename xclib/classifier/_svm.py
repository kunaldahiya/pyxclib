from sklearn.svm import LinearSVC
import numpy as np
from sklearn.svm._liblinear import train_wrap, set_verbosity_wrap
import scipy.sparse as sp
import warnings


def apply_threshold(data, threshold):
    data[np.where(np.abs(data) < threshold)] = 0


def _get_sample_weight(num_instances, dtype=np.float64):
    return np.ones(num_instances, dtype=dtype)


def _get_class_weight(num_classes=2):
    return np.ones(num_classes, dtype=np.float64, order='C')


def _get_random_state():
    return np.random.mtrand._rand


def train_one_safe(data, loss, C, verbose, max_iter, threshold, dual, tol):
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


def train_one(data, solver_type, C, verbose, max_iter, threshold, tol,
              intercept_scaling, fit_intercept, epsilon):
    def _get_features(obj):
        # Index samples iff they are required
        # Helful in reducing memory footprint
        if obj['ind'] is None:
            return obj['data']
        else:
            return obj['data'].take(obj['ind'], axis=0)
    set_verbosity_wrap(verbose)
    rnd = _get_random_state()
    if verbose:
        print('[LibLinear]', end='')
    bias = intercept_scaling if fit_intercept else -1.0
    X, y = _get_features(data), np.asarray(data["Y"], dtype=np.float64).ravel()
    # FIXME: libnear gives weird warning with 1/-1
    # WARNING: class label 0 specified in weight is not found
    y[y == -1] = 0
    y = np.require(y, requirements="W")
    classes = np.unique(y)
    class_weight = _get_class_weight(len(classes))
    sample_weight = _get_sample_weight(X.shape[0])

    if not is_single_class(classes):
        raw_coef_, n_iter_ = train_wrap(
            X, y, sp.isspmatrix(X), solver_type, tol, bias, C,
            class_weight, max_iter, rnd.randint(np.iinfo('i').max),
            epsilon, sample_weight)
    else:
        if verbose:
            warnings.warn("Liblinear requires atleast 2 classes; "
                          "using default values.")
        raw_coef_ = np.zeros((1, X.shape[1]+1), dtype=np.float32)
        n_iter_ = [-1]
        raw_coef_[:, -1] = -1e9  # highly negative value; don't predict it
    if verbose and max(n_iter_) >= max_iter:
        warnings.warn("Liblinear failed to converge, increase "
                      "the number of iterations.")
    if fit_intercept:
        weight = raw_coef_[:, :-1]
        bias = intercept_scaling * raw_coef_[:, -1]
    else:
        weight = raw_coef_
        bias = 0.
    apply_threshold(weight, threshold)
    return weight, bias


def _get_liblinear_solver_type(multi_class, penalty, loss, dual):
    """Find the liblinear magic number for the solver.
    This number depends on the values of the following attributes:
      - multi_class
      - penalty
      - loss
      - dual
    The same number is also internally used by LibLinear to determine
    which solver to use.
    Original source:
    https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/svm/_base.py
    """
    # nested dicts containing level 1: available loss functions,
    # level2: available penalties for the given loss function,
    # level3: whether the dual solver is available for the specified
    # combination of loss function and penalty
    _solver_type_dict = {
        'logistic_regression': {
            'l1': {False: 6},
            'l2': {False: 0, True: 7}},
        'hinge': {
            'l2': {True: 3}},
        'squared_hinge': {
            'l1': {False: 5},
            'l2': {False: 2, True: 1}},
        'epsilon_insensitive': {
            'l2': {True: 13}},
        'squared_epsilon_insensitive': {
            'l2': {False: 11, True: 12}},
        'crammer_singer': 4
    }

    if multi_class == 'crammer_singer':
        return _solver_type_dict[multi_class]
    elif multi_class != 'ovr':
        raise ValueError("`multi_class` must be one of `ovr`, "
                         "`crammer_singer`, got %r" % multi_class)

    _solver_pen = _solver_type_dict.get(loss, None)
    if _solver_pen is None:
        error_string = ("loss='%s' is not supported" % loss)
    else:
        _solver_dual = _solver_pen.get(penalty, None)
        if _solver_dual is None:
            error_string = ("The combination of penalty='%s' "
                            "and loss='%s' is not supported"
                            % (penalty, loss))
        else:
            solver_num = _solver_dual.get(dual, None)
            if solver_num is None:
                error_string = ("The combination of penalty='%s' and "
                                "loss='%s' are not supported when dual=%s"
                                % (penalty, loss, dual))
            else:
                return solver_num
    raise ValueError('Unsupported set of arguments: %s, '
                     'Parameters: penalty=%r, loss=%r, dual=%r'
                     % (error_string, penalty, loss, dual))


def is_single_class(classes):
    return len(classes) == 1
