# -*- coding: utf-8 -*-

from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import KDTree
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
    #   1.2 Scikit-Learn or SciPy objects
    if isinstance(model, (sp.sparse.csr_matrix, KDTree, SparseCoder)):
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
