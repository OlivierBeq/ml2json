# -*- coding: utf-8 -*-

import inspect

from . import _base

# Allow additional dependencies to be optional
__optionals__ = []
try:
    from sklearn_extra.kernel_methods import EigenProRegressor, EigenProClassifier
    __optionals__.extend(['EigenProRegressor', 'EigenProClassifier'])

    # scikit-learn-extra 0.3.0 (its latest release) was written against a scipy
    # version whose linalg.eigh() still accepted eigvals=(lo, hi); newer scipy
    # renamed that to subset_by_index=, and against a scikit-learn version whose
    # check_X_y() still accepted force_all_finite= rather than ensure_all_finite=.
    # Both are called internally by EigenPro*.fit(), so shim them at the module
    # level rather than leaving the estimators permanently unusable on any
    # scipy/scikit-learn newer than the ones scikit-learn-extra shipped against.
    import sklearn_extra.kernel_methods._eigenpro as _eigenpro_mod

    if 'eigvals' not in inspect.signature(_eigenpro_mod.eigh).parameters:
        _orig_eigh = _eigenpro_mod.eigh

        def _eigh_shim(*args, **kwargs):
            if 'eigvals' in kwargs:
                kwargs['subset_by_index'] = kwargs.pop('eigvals')
            return _orig_eigh(*args, **kwargs)

        _eigenpro_mod.eigh = _eigh_shim

    if 'force_all_finite' not in inspect.signature(_eigenpro_mod.check_X_y).parameters:
        _orig_check_X_y = _eigenpro_mod.check_X_y

        def _check_X_y_shim(*args, **kwargs):
            if 'force_all_finite' in kwargs:
                kwargs['ensure_all_finite'] = kwargs.pop('force_all_finite')
            return _orig_check_X_y(*args, **kwargs)

        _eigenpro_mod.check_X_y = _check_X_y_shim
except:
    pass


if 'EigenProRegressor' in __optionals__:
    def serialize_eigenpro_regressor(model):
        return _base.serialize_model_generic(model)


    def deserialize_eigenpro_regressor(model_dict):
        return _base.deserialize_model_generic(model_dict)


if 'EigenProClassifier' in __optionals__:
    def serialize_eigenpro_classifier(model):
        return _base.serialize_model_generic(model)


    def deserialize_eigenpro_classifier(model_dict):
        return _base.deserialize_model_generic(model_dict)
