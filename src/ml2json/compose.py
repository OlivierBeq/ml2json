# -*- coding: utf-8 -*-

from . import _base

# Allow additional dependencies to be optional
__optionals__ = []


def serialize_column_transformer(model):
    return _base.serialize_model_generic(model)


def deserialize_column_transformer(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_transformed_target_regressor(model):
    return _base.serialize_model_generic(model)


def deserialize_transformed_target_regressor(model_dict):
    return _base.deserialize_model_generic(model_dict)
