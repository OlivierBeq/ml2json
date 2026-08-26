# -*- coding: utf-8 -*-

import os
import uuid
import inspect
import importlib

import numpy as np
import scipy as sp
import sklearn
from sklearn.pipeline import FeatureUnion, Pipeline

from . import _base
from .utils.memory import serialize_memory, deserialize_memory


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
        'verbose': model.verbose,
        'params': {param: value
                   for param, value in model.get_params().items()
                   if param in ['steps', 'memory', 'verbose']},
    }
    # A step can be the literal sentinel 'passthrough' (or None) rather than a
    # fitted estimator - not JSON-serializable via serialize_model, which
    # expects an actual model instance.
    serialized_model['params']['steps'] = [(name, estimator if estimator is None or isinstance(estimator, str)
                                            else serialize_model(estimator))
                                           for name, estimator in model.steps]
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

    model_dict['params']['steps'] = [(name, estimator if estimator is None or isinstance(estimator, str)
                                      else deserialize_model(estimator))
                                     for name, estimator in model_dict['params']['steps']]
    if model_dict['params']['memory'] is not None and isinstance(model_dict['params']['memory'], dict):
        model_dict['params']['memory'] = deserialize_memory(model_dict['params']['memory'])
    model = Pipeline(**model_dict['params'])
    if 'classes_' in model_dict.keys():
        model.classes_ = np.array(model_dict['classes_'])
    if 'n_feature_in_' in model_dict.keys():
        model.n_feature_in_ = np.array(model_dict['n_feature_in_'])
    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

    return model


# Structurally similar to Pipeline (a list of named sub-transformers), but
# with no exotic state of its own (no memory/verbose caching dance) - the
# generic engine already recurses through transformer_list's nested,
# already-supported transformers on its own.
def serialize_feature_union(model):
    return _base.serialize_model_generic(model)


def deserialize_feature_union(model_dict):
    return _base.deserialize_model_generic(model_dict)


if 'imblearn' in __optionals__:
    def serialize_imblearn_pipeline(model):
        from .ml2json import serialize_model

        serialized_model = {
            'verbose': model.verbose,
            'params': {param: value
                       for param, value in model.get_params().items()
                       if param in ['steps', 'memory', 'verbose']},
        }
        serialized_model['params']['steps'] = [(name, estimator if estimator is None or isinstance(estimator, str)
                                                else serialize_model(estimator))
                                               for name, estimator in model.steps]
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

        model_dict['params']['steps'] = [(name, estimator if estimator is None or isinstance(estimator, str)
                                          else deserialize_model(estimator))
                                         for name, estimator in model_dict['params']['steps']]
        if model_dict['params']['memory'] is not None and isinstance(model_dict['params']['memory'], dict):
            model_dict['params']['memory'] = deserialize_memory(model_dict['params']['memory'])
        model = ImblearnPipeline(**model_dict['params'])
        if 'classes_' in model_dict.keys():
            model.classes_ = np.array(model_dict['classes_'])
        if 'n_feature_in_' in model_dict.keys():
            model.n_feature_in_ = np.array(model_dict['n_feature_in_'])
        if 'feature_names_in_' in model_dict.keys():
            model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

        return model
