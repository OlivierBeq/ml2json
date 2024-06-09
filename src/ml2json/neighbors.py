# -*- coding: utf-8 -*-

import inspect
import importlib

import numpy as np
import scipy.sparse
from sklearn.neighbors import NearestNeighbors, KDTree, KernelDensity

# Allow additional dependencies to be optional
__optionals__ = []
try:
    from pynndescent import NNDescent
    __optionals__.append('NNDescent')
except:
    pass

from .utils.csr import serialize_csr_matrix, deserialize_csr_matrix


def serialize_nearest_neighbors(model):
    serialized_model = {
        'meta': 'nearest-neighbors',
        'effective_metric_params_': model.effective_metric_params_,
        '_fit_method': model._fit_method,
        '_fit_X': model._fit_X.tolist() if not scipy.sparse.issparse(model._fit_X) else serialize_csr_matrix(model._fit_X),
        'n_samples_fit_': model.n_samples_fit_,
        'effective_metric_': model.effective_metric_,
        'n_features_in_': model.n_features_in_,
        'params': model.get_params(),
    }

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()
    if model._tree is not None:
        serialized_model['_tree'] = serialize_kdtree(model._tree)
    else:
        serialized_model['_tree'] = model._tree

    return serialized_model


def deserialize_nearest_neighbors(model_dict):
    model = NearestNeighbors(**model_dict['params'])

    model.effective_metric_params_ = model_dict['effective_metric_params_']
    model._fit_method = model_dict['_fit_method']
    model._fit_X = np.array(model_dict['_fit_X']) if isinstance(model_dict['_fit_X'], list) else deserialize_csr_matrix(model_dict['_fit_X'])
    model.n_samples_fit_ = model_dict['n_samples_fit_']
    model.effective_metric_ = model_dict['effective_metric_']
    model.n_features_in_ = model_dict['n_features_in_']

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])
    if model_dict['_tree'] is not None:
        model._tree = deserialize_kdtree(model_dict['_tree'])
    else:
        model._tree = model_dict['_tree']

    return model


def serialize_kernel_density(model):
    serialized_model = {
        'meta': 'kernel-density',
        'bandwidth_': model.bandwidth_,
        'n_features_in_': model.n_features_in_,
        'params': model.get_params(),
    }
    
    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()
    if model.tree_ is not None:
        serialized_model['tree_'] = serialize_kdtree(model.tree_)
    else:
        serialized_model['tree_'] = model.tree_

    return serialized_model

def deserialize_kernel_density(model_dict):
    model = KernelDensity(**model_dict['params'])

    model.bandwidth_ = model_dict['bandwidth_']
    model.n_features_in_ = model_dict['n_features_in_']

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])
    if model_dict['tree_'] is not None:
        model.tree_ = deserialize_kdtree(model_dict['tree_'])
    else:
        model.tree_ = model_dict['tree_']

    return model
    

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

    model.__setstate__(params)

    return model

if 'NNDescent' in __optionals__:
    def serialize_nndescent(model):
        state = model.__getstate__()

        del state['_distance_func'], state['_tree_search']
        del state['_search_function'], state['_deheap_function']
        del state['_distance_correction']

        state['_raw_data'] = state['_raw_data'].astype(float).tolist()
        state['rng_state'] = state['rng_state'].astype(int).tolist()
        state['search_rng_state'] = state['search_rng_state'].astype(int).tolist()
        state['_search_graph'] = serialize_csr_matrix(state['_search_graph'])
        state['_visited'] = state['_visited'].astype(int).tolist()
        state['_vertex_order'] = state['_vertex_order'].astype(int).tolist()
        state['_neighbor_graph'] = (state['_neighbor_graph'][0].tolist(),
                                    state['_neighbor_graph'][1].astype(float).tolist())
        state['_search_forest'] = ((state['_search_forest'][0][0].astype(float).tolist(),
                                    state['_search_forest'][0][1].astype(float).tolist(),
                                    state['_search_forest'][0][2].astype(int).tolist(),
                                    state['_search_forest'][0][3].astype(int).tolist(),
                                    state['_search_forest'][0][4]),)

        serialized_model = {
            'meta': 'nn-descent',
            'params': state
        }

        return serialized_model


    def deserialize_nndescent(model_dict):

        params = model_dict['params']

        params['_raw_data'] = np.array(params['_raw_data'], dtype=np.float32)
        params['rng_state'] = np.array(params['rng_state'], dtype=np.int64)
        params['search_rng_state'] = np.array(params['search_rng_state'], dtype=np.int64)
        params['_search_graph'] = deserialize_csr_matrix(params['_search_graph'])
        params['_visited'] = np.array(params['_visited'], dtype=np.uint8)
        params['_vertex_order'] = np.array(params['_vertex_order'], dtype=np.int32)
        params['_neighbor_graph'] = (np.array(params['_neighbor_graph'][0]),
                                     np.array(params['_neighbor_graph'][1], dtype=np.float32))
        params['_search_forest'] = ((np.array(params['_search_forest'][0][0], dtype=np.float32),
                                     np.array(params['_search_forest'][0][1], dtype=np.float32),
                                     np.array(params['_search_forest'][0][2], dtype=np.int32),
                                     np.array(params['_search_forest'][0][3], dtype=np.int32),
                                     params['_search_forest'][0][4]),)

        model = NNDescent(params['_raw_data'], metric=params['metric'], metric_kwds=params['metric_kwds'])

        params['_distance_func'] = model._distance_func
        params['_distance_correction'] = model._distance_correction

        model.__setstate__(params)

        return model
