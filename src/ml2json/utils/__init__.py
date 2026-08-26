# -*- coding: utf-8 -*-

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import KDTree, BallTree
from sklearn.decomposition import SparseCoder

import scipy as sp

def is_model_fitted(model):
    # 1) Models that are not estimators (no fit method)
    #   1.1 Models that depend on optional librairies
    try:
        from pynndescent import NNDescent
        if isinstance(model, NNDescent):
            return True
    except:
        pass
    #   1.2 mlchemad applicability domains are not sklearn estimators and thus
    #       lack __sklearn_tags__, which sklearn's check_is_fitted requires since 1.6
    try:
        from mlchemad.base import ApplicabilityDomain
        if isinstance(model, ApplicabilityDomain):
            return model.fitted_
    except ImportError:
        pass
    #   1.3 Scikit-Learn or SciPy objects
    if isinstance(model, (sp.sparse.csr_matrix, KDTree, BallTree, SparseCoder)):
        return True
    #   1.4 Any other non-estimator object (e.g. a scipy.stats distribution
    #       nested inside a param_distributions dict): these have no
    #       fitted/unfitted state and, since sklearn 1.6, check_is_fitted's
    #       tag-based check crashes with AttributeError on non-BaseEstimator
    #       objects rather than a catchable TypeError, so this must be
    #       checked before calling it at all.
    if not isinstance(model, BaseEstimator):
        return True
    # 2) Models that are estimators
    try:
        check_is_fitted(model)
        return True
    except NotFittedError:
        return False
    except TypeError as e:
        if str(e).endswith('is not an estimator instance.'):
            return True
        raise e


def recursive_inspection(iterable):
    if isinstance(iterable, (int, str, float)) or iterable is None:
        return
    elif isinstance(iterable, dict):
        for key, value in iterable.items():
            ret = recursive_inspection(value)
            if ret is not None:
                return f'dict({key} -> {ret})'
    elif isinstance(iterable, (list, tuple)):
        for value in iterable:
            ret = recursive_inspection(value)
            if ret is not None:
                return f'{type(iterable).__name__}({ret})'
    else:
        return type(iterable)
