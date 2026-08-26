# -*- coding: utf-8 -*-

from . import _base

# Allow additional dependencies to be optional
__optionals__ = []


def serialize_one_vs_one_classifier(model):
    return _base.serialize_model_generic(model)


def deserialize_one_vs_one_classifier(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_one_vs_rest_classifier(model):
    return _base.serialize_model_generic(model)


def deserialize_one_vs_rest_classifier(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_output_code_classifier(model):
    return _base.serialize_model_generic(model)


def deserialize_output_code_classifier(model_dict):
    return _base.deserialize_model_generic(model_dict)
