# -*- coding: utf-8 -*-

from . import _base

# Allow additional dependencies to be optional
__optionals__ = []


def serialize_gaussian_mixture(model):
    return _base.serialize_model_generic(model)


def deserialize_gaussian_mixture(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_bayesian_gaussian_mixture(model):
    return _base.serialize_model_generic(model)


def deserialize_bayesian_gaussian_mixture(model_dict):
    return _base.deserialize_model_generic(model_dict)
