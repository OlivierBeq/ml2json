# -*- coding: utf-8 -*-

from . import _base

# Allow additional dependencies to be optional
__optionals__ = []


def serialize_kernel_ridge(model):
    return _base.serialize_model_generic(model)


def deserialize_kernel_ridge(model_dict):
    return _base.deserialize_model_generic(model_dict)
