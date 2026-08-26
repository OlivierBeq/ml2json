# -*- coding: utf-8 -*-

from . import _base

# Allow additional dependencies to be optional
__optionals__ = []


def serialize_kfold(model):
    return _base.serialize_model_generic(model)


def deserialize_kfold(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_stratified_kfold(model):
    return _base.serialize_model_generic(model)


def deserialize_stratified_kfold(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_group_kfold(model):
    return _base.serialize_model_generic(model)


def deserialize_group_kfold(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_stratified_group_kfold(model):
    return _base.serialize_model_generic(model)


def deserialize_stratified_group_kfold(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_repeated_kfold(model):
    return _base.serialize_model_generic(model)


def deserialize_repeated_kfold(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_repeated_stratified_kfold(model):
    return _base.serialize_model_generic(model)


def deserialize_repeated_stratified_kfold(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_leave_one_out(model):
    return _base.serialize_model_generic(model)


def deserialize_leave_one_out(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_leave_p_out(model):
    return _base.serialize_model_generic(model)


def deserialize_leave_p_out(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_leave_one_group_out(model):
    return _base.serialize_model_generic(model)


def deserialize_leave_one_group_out(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_leave_p_groups_out(model):
    return _base.serialize_model_generic(model)


def deserialize_leave_p_groups_out(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_shuffle_split(model):
    return _base.serialize_model_generic(model)


def deserialize_shuffle_split(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_stratified_shuffle_split(model):
    return _base.serialize_model_generic(model)


def deserialize_stratified_shuffle_split(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_group_shuffle_split(model):
    return _base.serialize_model_generic(model)


def deserialize_group_shuffle_split(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_time_series_split(model):
    return _base.serialize_model_generic(model)


def deserialize_time_series_split(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_predefined_split(model):
    return _base.serialize_model_generic(model)


def deserialize_predefined_split(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_parameter_grid(model):
    return _base.serialize_model_generic(model)


def deserialize_parameter_grid(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_parameter_sampler(model):
    return _base.serialize_model_generic(model)


def deserialize_parameter_sampler(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_grid_search_cv(model):
    return _base.serialize_model_generic(model)


def deserialize_grid_search_cv(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_randomized_search_cv(model):
    return _base.serialize_model_generic(model)


def deserialize_randomized_search_cv(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_halving_grid_search_cv(model):
    return _base.serialize_model_generic(model)


def deserialize_halving_grid_search_cv(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_halving_randomized_search_cv(model):
    return _base.serialize_model_generic(model)


def deserialize_halving_randomized_search_cv(model_dict):
    return _base.deserialize_model_generic(model_dict)
