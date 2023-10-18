# -*- coding: utf-8 -*-

import inspect
import importlib

import numpy as np
from sklearn.preprocessing import (LabelEncoder, LabelBinarizer, MultiLabelBinarizer, MinMaxScaler, StandardScaler,
                                   KernelCenterer, OneHotEncoder)


def serialize_label_binarizer(label_binarizer):
    serialized_label_binarizer = {
        'meta': 'label-binarizer',
        'neg_label': label_binarizer.neg_label,
        'pos_label': label_binarizer.pos_label,
        'sparse_output': label_binarizer.sparse_output,
        'y_type_': label_binarizer.y_type_,
        'sparse_input_': label_binarizer.sparse_input_,
        'classes_': label_binarizer.classes_.tolist()
    }

    return serialized_label_binarizer


def deserialize_label_binarizer(label_binarizer_dict):
    label_binarizer = LabelBinarizer()
    label_binarizer.neg_label = label_binarizer_dict['neg_label']
    label_binarizer.pos_label = label_binarizer_dict['pos_label']
    label_binarizer.sparse_output = label_binarizer_dict['sparse_output']
    label_binarizer.y_type_ = label_binarizer_dict['y_type_']
    label_binarizer.sparse_input_ = label_binarizer_dict['sparse_input_']
    label_binarizer.classes_ = np.array(label_binarizer_dict['classes_'])

    return label_binarizer


def serialize_multilabel_binarizer(model):
    serialized_model = {
        'meta': 'multilabel-binarizer',
        'classes': sorted(list(model.classes_)),
        'sparse_output': str(model.sparse_output),
    }

    return serialized_model


def deserialize_multilabel_binarizer(model_dict):
    model = MultiLabelBinarizer()

    model.classes_ = np.array(model_dict['classes'])
    model.sparse_output = model_dict['sparse_output'] == 'True'
    model._cached_dict = dict(zip(model.classes_, range(len(model.classes_))))

    return model


def serialize_minmax_scaler(model):
    serialized_model = {
        'meta': 'minmax-scaler',
        'min_': model.min_.tolist(),
        'scale_': model.scale_.tolist(),
        'data_min_': model.data_min_.tolist(),
        'data_max_': model.data_max_.tolist(),
        'data_range_': model.data_range_.tolist(),
        'n_features_in_': model.n_features_in_,
        'n_samples_seen_': model.n_samples_seen_,
        'params': model.get_params(),
    }

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist(),

    return serialized_model


def deserialize_minmax_scaler(model_dict):
    model_dict['params']['feature_range'] = tuple(model_dict['params']['feature_range'])

    model = MinMaxScaler(**model_dict['params'])

    model.min_ = np.array(model_dict['min_'])
    model.scale_ = np.array(model_dict['scale_'])
    model.data_min_ = np.array(model_dict['data_min_'])
    model.data_max_ = np.array(model_dict['data_max_'])
    model.data_range_ = np.array(model_dict['data_range_'])
    model.n_features_in_ = model_dict['n_features_in_']
    model.n_samples_seen_ = model_dict['n_samples_seen_']

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'])

    return model

def serialize_standard_scaler(model):
    serialized_model = {
        'meta': 'standard-scaler',
        'n_features_in_': model.n_features_in_,
        'params': model.get_params(),
    }

    if model.var_ is None:
        serialized_model['var_'] = model.var_
    else:
        serialized_model['var_'] = model.var_.tolist()
    if model.mean_ is None:
        serialized_model['mean_'] = model.mean_
    else:
        serialized_model['mean_'] = model.mean_.tolist()
    if isinstance(model.scale_, np.ndarray):
        serialized_model['scale_'] = model.scale_.tolist()
    else:
        serialized_model['scale_'] = model.scale_,
    if isinstance(model.n_samples_seen_, (int, float)):
        serialized_model['n_samples_seen_'] = model.n_samples_seen_
    else:
        serialized_model['n_samples_seen_'] = model.n_samples_seen_.tolist()

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist(),

    return serialized_model


def deserialize_standard_scaler(model_dict):
    model = StandardScaler(**model_dict['params'])
    model.n_features_in_ = model_dict['n_features_in_']

    if isinstance(model_dict['mean_'], list):
        model.mean_ = np.array(model_dict['mean_'])
    else:
        model.mean_ = model_dict['mean_']
    if isinstance(model_dict['var_'], list):
        model.var_ = np.array(model_dict['var_'])
    else:
        model.var_ = model_dict['var_']
    if isinstance(model_dict['scale_'], list):
        model.scale_ = np.array(model_dict['scale_'])
    else:
        model.scale_ = model_dict['scale_']
    if isinstance(model_dict['n_samples_seen_'], list):
        model.n_samples_seen_ = np.array(model_dict['n_samples_seen_'])
    else:
        model.n_samples_seen_ = model_dict['n_samples_seen_']

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'])

    return model


def serialize_label_encoder(model):
    serialized_model = {
        'meta': 'label-encoder',
        'classes_': model.classes_.tolist(),
    }

    return serialized_model

def deserialize_label_encoder(model_dict):
    model = LabelEncoder()
    model.classes_ = np.array(model_dict['classes_'])

    return model


def serialize_kernel_centerer(model):
    serialized_model = {
        'meta': 'kernel-centerer',
        'K_fit_all_': model.K_fit_all_.astype(float).tolist(),
        'K_fit_rows_': model.K_fit_rows_.tolist(),
        'n_features_in_': model.n_features_in_,
    }

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist(),

    return serialized_model


def deserialize_kernel_centerer(model_dict):
    model = KernelCenterer()

    model.K_fit_all_ = np.array(model_dict['K_fit_all_'], dtype=np.float64)
    model.K_fit_rows_ = np.array(model_dict['K_fit_rows_'])
    model.n_features_in_ = model_dict['n_features_in_']

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'])

    return model

def serialize_onehot_encoder(model):
    serialized_model = {
        'meta': 'onehot-encoder',
        'categories_': [category.tolist() for category in model.categories_],
        'categories_dtype': [f"np.dtype('{str(category.dtype)}')" for category in model.categories_],
        'drop_idx_': model.drop_idx_ if model.drop_idx_ is None else model.drop_idx_.tolist(),
        '_infrequent_enabled': model._infrequent_enabled,
        'n_features_in_': model.n_features_in_,
        '_n_features_outs': model._n_features_outs,
        'params': model.get_params(),
    }

    serialized_model['params']['dtype'] = (inspect.getmodule(serialized_model['params']['dtype']).__name__,
                                           serialized_model['params']['dtype'].__name__)

    return serialized_model

def deserialize_onehot_encoder(model_dict):

    model_dict['params']['dtype'] = getattr(importlib.import_module(model_dict['params']['dtype'][0]),
                                            model_dict['params']['dtype'][1])

    model = OneHotEncoder(**model_dict['params'])

    model.categories_ = [np.array(category, dtype=eval(dtype))
                         for category, dtype in zip(model_dict['categories_'], model_dict['categories_dtype'])]
    model.drop_idx_ = model_dict['drop_idx_'] if model_dict['drop_idx_'] is None else np.array(model_dict['drop_idx_'])
    model._infrequent_enabled = model_dict['_infrequent_enabled']
    model.n_features_in_ = model_dict['n_features_in_']
    model._n_features_outs = model_dict['_n_features_outs']

    return model
