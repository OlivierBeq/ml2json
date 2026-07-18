# -*- coding: utf-8 -*-

from . import _base

# Allow additional dependencies to be optional
__optionals__ = []


def serialize_dummy_classifier(model):
    return _base.serialize_model_generic(model)


def deserialize_dummy_classifier(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_dummy_regressor(model):
    return _base.serialize_model_generic(model)


def deserialize_dummy_regressor(model_dict):
    return _base.deserialize_model_generic(model_dict)
