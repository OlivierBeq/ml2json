# -*- coding: utf-8 -*-

from . import _base

# Allow additional dependencies to be optional
__optionals__ = []


def serialize_elliptic_envelope(model):
    return _base.serialize_model_generic(model)


def deserialize_elliptic_envelope(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_empirical_covariance(model):
    return _base.serialize_model_generic(model)


def deserialize_empirical_covariance(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_graphical_lasso(model):
    return _base.serialize_model_generic(model)


def deserialize_graphical_lasso(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_graphical_lasso_cv(model):
    return _base.serialize_model_generic(model)


def deserialize_graphical_lasso_cv(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_ledoit_wolf(model):
    return _base.serialize_model_generic(model)


def deserialize_ledoit_wolf(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_min_cov_det(model):
    return _base.serialize_model_generic(model)


def deserialize_min_cov_det(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_oas(model):
    return _base.serialize_model_generic(model)


def deserialize_oas(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_shrunk_covariance(model):
    return _base.serialize_model_generic(model)


def deserialize_shrunk_covariance(model_dict):
    return _base.deserialize_model_generic(model_dict)
