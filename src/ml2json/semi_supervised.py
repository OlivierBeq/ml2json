# -*- coding: utf-8 -*-

from . import _base

# Allow additional dependencies to be optional
__optionals__ = []


def serialize_label_propagation(model):
    return _base.serialize_model_generic(model)


def deserialize_label_propagation(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_label_spreading(model):
    return _base.serialize_model_generic(model)


def deserialize_label_spreading(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_self_training_classifier(model):
    return _base.serialize_model_generic(model)


def deserialize_self_training_classifier(model_dict):
    return _base.deserialize_model_generic(model_dict)
