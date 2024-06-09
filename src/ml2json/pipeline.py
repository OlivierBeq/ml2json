# -*- coding: utf-8 -*-

import os
import uuid
import inspect
import importlib

import numpy as np
import scipy as sp
import sklearn
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.utils import Bunch

from .utils.memory import serialize_memory, deserialize_memory
from .utils.bunch import serialize_bunch, deserialize_bunch


# Allow additional dependencies to be optional
__optionals__ = []

try:
    from imblearn.pipeline import Pipeline as ImblearnPipeline
    __optionals__.append('imblearn')
except:
    pass


def serialize_pipeline(model):
    from .ml2json import serialize_model

    serialized_model = {
        'meta': 'pipeline',
        'verbose': model.verbose,
        'params': {param: value
                   for param, value in model.get_params().items()
                   if param in ['steps', 'memory', 'verbose']},
        'named_steps': serialize_bunch(model.named_steps)
    }
    serialized_model['params']['steps'] = [(name, serialize_model(estimator)) for name, estimator in model.steps]
    if not isinstance(serialized_model['params']['memory'], str) and serialized_model['params']['memory'] is not None:
        serialized_model['params']['memory'] = serialize_memory(serialized_model['params']['memory'])
    if 'classes_' in model.__dict__:
        serialized_model['classes_'] = model.classes_.tolist()
    if 'n_features_in_' in model.__dict__:
        serialized_model['n_features_in_'] = model.n_features_in_
    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()
    return serialized_model


def deserialize_pipeline(model_dict):
    from .ml2json import deserialize_model

    model_dict['params']['steps'] = [(name, deserialize_model(estimator)) for name, estimator in model_dict['params']['steps']]
    if model_dict['params']['memory'] is not None and isinstance(model_dict['params']['memory'], dict):
        model_dict['params']['memory'] = deserialize_memory(model_dict['params']['memory'])
    model = Pipeline(**model_dict['params'])
    if 'classes_' in model_dict.keys():
        model.classes_ = np.array(model_dict['classes_'])
    if 'n_feature_in_' in model_dict.keys():
        model.n_feature_in_ = np.array(model_dict['n_feature_in_'])
    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

    # model.named_steps = deserialize_bunch(model_dict['named_steps'])
    return model


if 'imblearn' in __optionals__:
    def serialize_imblearn_pipeline(model):
        from .ml2json import serialize_model

        serialized_model = {
            'meta': 'imblearn-pipeline',
            'verbose': model.verbose,
            'params': {param: value
                       for param, value in model.get_params().items()
                       if param in ['steps', 'memory', 'verbose']},
            'named_steps': serialize_bunch(model.named_steps)
        }
        serialized_model['params']['steps'] = [(name, serialize_model(estimator)) for name, estimator in model.steps]
        if not isinstance(serialized_model['params']['memory'], str) and serialized_model['params']['memory'] is not None:
            serialized_model['params']['memory'] = serialize_memory(serialized_model['params']['memory'])
        if 'classes_' in model.__dict__:
            serialized_model['classes_'] = model.classes_.tolist()
        if 'n_features_in_' in model.__dict__:
            serialized_model['n_features_in_'] = model.n_features_in_
        if 'feature_names_in_' in model.__dict__:
            serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()
        return serialized_model


    def deserialize_imblearn_pipeline(model_dict):
        from .ml2json import deserialize_model

        model_dict['params']['steps'] = [(name, deserialize_model(estimator)) for name, estimator in model_dict['params']['steps']]
        if model_dict['params']['memory'] is not None and isinstance(model_dict['params']['memory'], dict):
            model_dict['params']['memory'] = deserialize_memory(model_dict['params']['memory'])
        model = Pipeline(**model_dict['params'])
        if 'classes_' in model_dict.keys():
            model.classes_ = np.array(model_dict['classes_'])
        if 'n_feature_in_' in model_dict.keys():
            model.n_feature_in_ = np.array(model_dict['n_feature_in_'])
        if 'feature_names_in_' in model_dict.keys():
            model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

        # model.named_steps = deserialize_bunch(model_dict['named_steps'])
        return model
