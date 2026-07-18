# -*- coding: utf-8 -*-

from . import _base

# Allow additional dependencies to be optional
__optionals__ = []


def serialize_additive_chi2_sampler(model):
    return _base.serialize_model_generic(model)


def deserialize_additive_chi2_sampler(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_nystroem(model):
    return _base.serialize_model_generic(model)


def deserialize_nystroem(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_polynomial_count_sketch(model):
    return _base.serialize_model_generic(model)


def deserialize_polynomial_count_sketch(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_rbf_sampler(model):
    return _base.serialize_model_generic(model)


def deserialize_rbf_sampler(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_skewed_chi2_sampler(model):
    return _base.serialize_model_generic(model)


def deserialize_skewed_chi2_sampler(model_dict):
    return _base.deserialize_model_generic(model_dict)
