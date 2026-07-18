# -*- coding: utf-8 -*-

from . import _base

# Allow additional dependencies to be optional
__optionals__ = []


def serialize_isotonic_regression(model):
    # model.f_ is a scipy.interpolate.interp1d built on top of a __slots__-based
    # base class (_Interpolator1D): its state lives outside __dict__ (e.g.
    # _y_extra_shape is a slot, not an instance attribute), so a plain
    # object.__new__() + __dict__ restore silently drops it and predict()
    # blows up on the reconstructed copy. It's fully derived from
    # X_thresholds_/y_thresholds_/out_of_bounds (all already serialized here),
    # so just exclude it and rebuild it via model._build_f() on deserialize.
    f_ = model.__dict__.pop('f_', None)
    try:
        return _base.serialize_model_generic(model)
    finally:
        if f_ is not None:
            model.f_ = f_


def deserialize_isotonic_regression(model_dict):
    model = _base.deserialize_model_generic(model_dict)
    if hasattr(model, 'X_thresholds_') and hasattr(model, 'y_thresholds_'):
        model._build_f(model.X_thresholds_, model.y_thresholds_)
    return model
