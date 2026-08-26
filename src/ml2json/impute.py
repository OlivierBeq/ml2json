# -*- coding: utf-8 -*-

from . import _base

# Allow additional dependencies to be optional
__optionals__ = []


def serialize_simple_imputer(model):
    return _base.serialize_model_generic(model)


def deserialize_simple_imputer(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_missing_indicator(model):
    return _base.serialize_model_generic(model)


def deserialize_missing_indicator(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_knn_imputer(model):
    return _base.serialize_model_generic(model)


def deserialize_knn_imputer(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_iterative_imputer(model):
    return _base.serialize_model_generic(model)


def deserialize_iterative_imputer(model_dict):
    return _base.deserialize_model_generic(model_dict)
