# -*- coding: utf-8 -*-

import inspect
import importlib

import numpy as np
from sklearn.neighbors import NearestNeighbors, KDTree


def serialize_kdtree(model):
    state = model.__getstate__()
    serialized_model = {
        'meta': 'kdtree',
        'data': np.array(model.data).tolist(),
        'data_arr': state[0].tolist(),
        'idx_data_arr': state[1].astype(int).tolist(),
        'node_data_arr': state[2].tolist(),
        'node_data_arr_dtype': f"np.dtype({str(state[2].dtype)})",
        'node_bounds_arr': state[3].tolist(),
        'leaf_size': state[4],
        'n_levels': state[5],
        'n_nodes': state[6],
        'n_trims': state[7],
        'n_leaves': state[8],
        'n_splits': state[9],
        'n_calls': state[10],
        'dist_metric': (inspect.getmodule(type(state[11])).__name__,
                         type(state[11]).__name__),
    }

    if state[12] is not None:
        serialized_model['sample_weight_arr'] = state[12].tolist()
    else:
        serialized_model['sample_weight_arr'] = state[12]

    return serialized_model


def deserialize_kdtree(model_dict):
    model = KDTree(np.array(model_dict['data']))

    params = [
        np.array(model_dict['data_arr']),
        np.array(model_dict['idx_data_arr'], dtype=np.int64),
        np.array(list(map(tuple, model_dict['node_data_arr'])), dtype=eval(model_dict['node_data_arr_dtype'])),
        np.array(model_dict['node_bounds_arr']),
        model_dict['leaf_size'],
        model_dict['n_levels'],
        model_dict['n_nodes'],
        model_dict['n_trims'],
        model_dict['n_leaves'],
        model_dict['n_splits'],
        model_dict['n_calls'],
        getattr(importlib.import_module(model_dict['dist_metric'][0]), model_dict['dist_metric'][1])(),
    ]

    if model_dict['sample_weight_arr'] is not None:
        params.append(np.array(model_dict['sample_weight_arr']))
    else:
        params.append(model_dict['sample_weight_arr'])

    params = tuple(params)
    print(len(params))

    model.__setstate__(params)

    return model
