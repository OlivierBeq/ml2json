# -*- coding: utf-8 -*-

from . import _base

# Allow additional dependencies to be optional
__optionals__ = []


def serialize_classifier_chain(model):
    return _base.serialize_model_generic(model)


def deserialize_classifier_chain(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_multioutput_classifier(model):
    return _base.serialize_model_generic(model)


def deserialize_multioutput_classifier(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_multioutput_regressor(model):
    return _base.serialize_model_generic(model)


def deserialize_multioutput_regressor(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_regressor_chain(model):
    return _base.serialize_model_generic(model)


def deserialize_regressor_chain(model_dict):
    return _base.deserialize_model_generic(model_dict)
