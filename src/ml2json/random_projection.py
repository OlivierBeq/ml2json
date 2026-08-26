# -*- coding: utf-8 -*-

from . import _base

# Allow additional dependencies to be optional
__optionals__ = []


def serialize_gaussian_random_projection(model):
    return _base.serialize_model_generic(model)


def deserialize_gaussian_random_projection(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_sparse_random_projection(model):
    return _base.serialize_model_generic(model)


def deserialize_sparse_random_projection(model_dict):
    return _base.deserialize_model_generic(model_dict)
