# -*- coding: utf-8 -*-

import inspect
import importlib

import numpy as np
from sklearn.preprocessing import (LabelEncoder, LabelBinarizer, MultiLabelBinarizer, MinMaxScaler, StandardScaler,
                                   KernelCenterer, OneHotEncoder, RobustScaler, MaxAbsScaler, OrdinalEncoder,
                                   Normalizer)


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
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()

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
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

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
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()

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
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

    return model


def serialize_robust_scaler(model):
    serialized_model = {
        'meta': 'robust-scaler',
        'n_features_in_': model.n_features_in_,
        'params': model.get_params(),
    }
    if hasattr(model, 'scale_'):
        serialized_model['scale_'] = (model.scale_.tolist()
                                      if isinstance(model.scale_, np.ndarray)
                                      else model.scale_)
    if hasattr(model, 'center_'):
        serialized_model['center_'] = (model.center_.tolist()
                                       if isinstance(model.center_, np.ndarray)
                                       else model.center_)
    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()


    return serialized_model


def deserialize_robust_scaler(model_dict):
    model_dict['params']['quantile_range'] = tuple(model_dict['params']['quantile_range'])

    model = RobustScaler(**model_dict['params'])
    model.n_features_in_ = model_dict['n_features_in_']

    if 'scale_' in model_dict.keys():
        if isinstance(model_dict['scale_'], list):
            model.scale_ = np.array(model_dict['scale_'])
        else:
            model.scale_ = model_dict['scale_']
    if 'center_' in model_dict.keys():
        if isinstance(model_dict['center_'], list):
            model.center_ = np.array(model_dict['center_'])
        else:
            model.center_ = model_dict['center_']

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

    return model


def serialize_maxabs_scaler(model):
    serialized_model = {
        'meta': 'maxabs-scaler',
        'params': model.get_params(),
    }

    if 'n_features_in_' in model.__dict__:
        serialized_model['n_features_in_'] = model.n_features_in_
    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()
    if 'n_samples_seen_' in model.__dict__:
        serialized_model['n_samples_seen_'] = model.n_samples_seen_
    if 'max_abs_' in model.__dict__:
        serialized_model['max_abs_'] = model.max_abs_.tolist()
    if 'scale_' in model.__dict__:
        serialized_model['scale_'] = model.scale_.tolist()

    return serialized_model


def deserialize_maxabs_scaler(model_dict):
    model = MaxAbsScaler(**model_dict['params'])

    if 'n_features_in_' in model_dict.keys():
        model.n_features_in_ = model_dict['n_features_in_']
    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])
    if 'n_samples_seen_' in model_dict.keys():
        model.n_samples_seen_ = model_dict['n_samples_seen_']
    if 'max_abs_' in model_dict.keys():
        model.max_abs_ = np.array(model_dict['max_abs_'])
    if 'scale_' in model_dict.keys():
        model.scale_ = np.array(model_dict['scale_'])

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
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

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

    if inspect.getmodule(serialized_model['params']['dtype']) is None:
        dtype = type(serialized_model['params']['dtype'])
        name = serialized_model['params']['dtype'].name
    else:
        dtype = serialized_model['params']['dtype']
        name = dtype.__name__

    serialized_model['params']['dtype'] = (inspect.getmodule(dtype).__name__, name)

    if '_drop_idx_after_grouping' in model.__dict__:
        serialized_model['_drop_idx_after_grouping'] = model._drop_idx_after_grouping.tolist() if model._drop_idx_after_grouping is not None else None

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

    if '_drop_idx_after_grouping' in model_dict.keys():
        model._drop_idx_after_grouping = np.array(model_dict['_drop_idx_after_grouping']) if model_dict['_drop_idx_after_grouping'] is not None else None

    return model


def serialize_ordinal_encoder(model):
    serialized_model = {
        'meta': 'ordinal-encoder',
        'categories_': [category.tolist() for category in model.categories_],
        'categories_dtype': [f"np.dtype('{str(category.dtype)}')" for category in model.categories_],
        '_infrequent_enabled': model._infrequent_enabled,
        'n_features_in_': model.n_features_in_,
        'params': model.get_params(),
    }

    if inspect.getmodule(serialized_model['params']['dtype']) is None:
        dtype = type(serialized_model['params']['dtype'])
        name = serialized_model['params']['dtype'].name
    else:
        dtype = serialized_model['params']['dtype']
        name = dtype.__name__

    serialized_model['params']['dtype'] = (inspect.getmodule(dtype).__name__, name)

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()
    if 'infrequent_categories_' in model.__dict__:
        serialized_model['infrequent_categories_'] = model.infrequent_categories_.tolist()
    if '_infrequent_indices' in model.__dict__:
        serialized_model['_infrequent_indices'] = [x.tolist() for x in model._infrequent_indices]
    if '_default_to_infrequent_mappings' in model.__dict__:
        serialized_model['_default_to_infrequent_mappings'] = [x.tolist() for x in model._default_to_infrequent_mappings]
    if '_missing_indices' in model.__dict__:
        serialized_model['_missing_indices'] = model._missing_indices

    return serialized_model


def deserialize_ordinal_encoder(model_dict):

    model_dict['params']['dtype'] = getattr(importlib.import_module(model_dict['params']['dtype'][0]),
                                            model_dict['params']['dtype'][1])

    model = OrdinalEncoder(**model_dict['params'])

    model.categories_ = [np.array(category, dtype=eval(dtype))
                         for category, dtype in zip(model_dict['categories_'], model_dict['categories_dtype'])]
    model.n_features_in_ = model_dict['n_features_in_']

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])
    if 'infrequent_categories_' in model_dict.keys():
        model.infrequent_categories_ = np.array(model_dict['infrequent_categories_'])
    if '_infrequent_enabled' in model_dict.keys():
        model._infrequent_enabled = model_dict['_infrequent_enabled']
    if '_infrequent_indices' in model_dict.keys():
        model._infrequent_indices = [np.array(x) for x in model_dict['_infrequent_indices']]
    if '_default_to_infrequent_mappings' in model_dict.keys():
        model._default_to_infrequent_mappings = [np.array(x) for x in model_dict['_default_to_infrequent_mappings']]
    if '_missing_indices' in model_dict.keys():
        model._missing_indices = {int(param): int(value) for param, value in model_dict['_missing_indices'].items()}

    return model


def serialize_normalizer(model):
    serialized_model = {
        'meta': 'normalizer',
        'params': model.get_params(),
    }

    if 'n_features_in_' in model.__dict__:
        serialized_model['n_features_in_'] = model.n_features_in_
    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()

    return serialized_model


def deserialize_normalizer(model_dict):

    model = Normalizer(**model_dict['params'])

    if 'n_features_in_' in model_dict.keys():
        model.n_features_in_ = model_dict['n_features_in_']
    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

    return model
