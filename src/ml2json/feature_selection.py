# -*- coding: utf-8 -*-

from . import _base

# Allow additional dependencies to be optional
__optionals__ = []


def serialize_select_from_model(model):
    return _base.serialize_model_generic(model)


def deserialize_select_from_model(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_rfe(model):
    return _base.serialize_model_generic(model)


def deserialize_rfe(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_rfecv(model):
    return _base.serialize_model_generic(model)


def deserialize_rfecv(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_sequential_feature_selector(model):
    return _base.serialize_model_generic(model)


def deserialize_sequential_feature_selector(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_generic_univariate_select(model):
    return _base.serialize_model_generic(model)


def deserialize_generic_univariate_select(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_select_fdr(model):
    return _base.serialize_model_generic(model)


def deserialize_select_fdr(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_select_fpr(model):
    return _base.serialize_model_generic(model)


def deserialize_select_fpr(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_select_fwe(model):
    return _base.serialize_model_generic(model)


def deserialize_select_fwe(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_select_kbest(model):
    return _base.serialize_model_generic(model)


def deserialize_select_kbest(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_select_percentile(model):
    return _base.serialize_model_generic(model)


def deserialize_select_percentile(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_variance_threshold(model):
    return _base.serialize_model_generic(model)


def deserialize_variance_threshold(model_dict):
    return _base.deserialize_model_generic(model_dict)
