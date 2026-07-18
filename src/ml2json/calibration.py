# -*- coding: utf-8 -*-

from . import _base

# Allow additional dependencies to be optional
__optionals__ = []


def serialize_calibrated_classifier_cv(model):
    return _base.serialize_model_generic(model)


def deserialize_calibrated_classifier_cv(model_dict):
    return _base.deserialize_model_generic(model_dict)
