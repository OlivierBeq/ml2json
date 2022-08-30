# -*- coding: utf-8 -*-
import importlib
import inspect

import numpy as np
from sklearn.cluster import (AffinityPropagation, AgglomerativeClustering,
                             Birch, DBSCAN, FeatureAgglomeration, KMeans,
                             MiniBatchKMeans, MeanShift, OPTICS, SpectralClustering,
                             SpectralBiclustering, SpectralCoclustering)
from sklearn.cluster._birch import _CFNode


def serialize_kmeans(model):
    serialized_model = {
        'meta': 'kmeans',
        'cluster_centers_': model.cluster_centers_.tolist(),
        'labels_': model.labels_.tolist(),
        'inertia_': model.inertia_,
        '_tol': model._tol,
        '_n_init': model._n_init,
        '_n_threads': model._n_threads,
        'n_iter_': model.n_iter_,
        'n_features_in_': model.n_features_in_,
        '_n_features_out': model._n_features_out,
        '_algorithm': model._algorithm,
        'params': model.get_params(),
    }

    if 'feature_names_in' in model.__dict__:
        serialized_model['feature_names_in'] = model.feature_names_in.tolist(),

    return serialized_model


def deserialize_kmeans(model_dict):
    model = KMeans(**model_dict['params'])

    model.cluster_centers_ = np.array(model_dict['cluster_centers_'])
    model.labels_ = np.array(model_dict['labels_'])
    model.inertia_ = model_dict['inertia_']
    model._tol = model_dict['_tol']
    model._n_init = model_dict['_n_init']
    model._n_threads = model_dict['_n_threads']
    model.n_iter_ = model_dict['n_iter_']
    model.n_features_in_ = model_dict['n_features_in_']
    model._n_features_out = model_dict['_n_features_out']
    model._algorithm = model_dict['_algorithm']

    if 'feature_names_in' in model_dict.keys():
        model.feature_names_in = np.array(model_dict['feature_names_in'])

    return model


def serialize_affinity_propagation(model):
    serialized_model = {
        'meta': 'affinity-propagation',
        'cluster_centers_indices_': model.cluster_centers_indices_.tolist(),
        'cluster_centers_': model.cluster_centers_.tolist(),
        'labels_': model.labels_.tolist(),
        'affinity_matrix_': model.affinity_matrix_.tolist(),
        'n_iter_': model.n_iter_,
        'n_features_in_': model.n_features_in_,
        'params': model.get_params(),
    }

    if 'feature_names_in' in model.__dict__:
        serialized_model['feature_names_in'] = model.feature_names_in.tolist(),

    return serialized_model


def deserialize_affinity_propagation(model_dict):
    model = AffinityPropagation(**model_dict['params'])

    model.cluster_centers_indices_ = np.array(model_dict['cluster_centers_indices_'], dtype=np.int64)
    model.cluster_centers_ = np.array(model_dict['cluster_centers_'])
    model.labels_ = np.array(model_dict['labels_'], dtype=np.int64)
    model.affinity_matrix_ = np.array(model_dict['affinity_matrix_'])
    model.n_iter_ = model_dict['n_iter_']
    model.n_features_in_ = model_dict['n_features_in_']

    if 'feature_names_in' in model_dict.keys():
        model.feature_names_in = np.array(model_dict['feature_names_in'])

    return model


def serialize_agglomerative_clustering(model):
    serialized_model = {
        'meta': 'agglomerative-clustering',
        'n_clusters_': model.n_clusters_,
        'labels_': model.labels_.tolist(),
        'n_leaves_': model.n_leaves_,
        'n_connected_components_': model.n_connected_components_,
        'n_features_in_': model.n_features_in_,
        'children_': model.children_.tolist(),
        'params': model.get_params(),
    }

    if 'feature_names_in' in model.__dict__:
        serialized_model['feature_names_in'] = model.feature_names_in.tolist()
    if 'distances_' in model.__dict__:
        serialized_model['distances_'] = model.distances_.tolist()

    return serialized_model


def deserialize_agglomerative_clustering(model_dict):
    model = AgglomerativeClustering(**model_dict['params'])

    model.n_clusters_ = model_dict['n_clusters_']
    model.labels_ = np.array(model_dict['labels_'])
    model.n_leaves_ = model_dict['n_leaves_']
    model.n_connected_components_ = model_dict['n_connected_components_']
    model.n_features_in_ = model_dict['n_features_in_']
    model.children_ = np.array(model_dict['children_'])

    if 'feature_names_in' in model_dict.keys():
        model.feature_names_in = np.array(model_dict['feature_names_in'])
    if 'distances_' in model_dict.keys():
        model.distances_ = np.array(model_dict['distances_'])

    return model

def serialize_dbscan(model):
    serialized_model = {
        'meta': 'dbscan',
        'components_': model.components_.tolist(),
        'core_sample_indices_': model.core_sample_indices_.tolist(),
        'labels_': model.labels_.tolist(),
        'n_features_in_': model.n_features_in_,
        '_estimator_type': model._estimator_type,
        'params': model.get_params()
    }

    if 'feature_names_in' in model.__dict__:
        serialized_model['feature_names_in'] = model.feature_names_in.tolist()

    return serialized_model


def deserialize_dbscan(model_dict):
    model = DBSCAN(**model_dict['params'])

    model.components_ = np.array(model_dict['components_'])
    model.labels_ = np.array(model_dict['labels_'])
    model.core_sample_indices_ = model_dict['core_sample_indices_']
    model.n_features_in_ = model_dict['n_features_in_']
    model._estimator_type = model_dict['_estimator_type']

    if 'feature_names_in' in model_dict.keys():
        model.feature_names_in = np.array(model_dict['feature_names_in'])

    return model


def serialize_optics(model):
    serialized_model = {
        'meta': 'optics',
        'labels_': model.labels_.tolist(),
        'reachability_': model.reachability_.tolist(),
        'ordering_': model.ordering_.tolist(),
        'core_distances_': model.core_distances_.tolist(),
        'predecessor_': model.predecessor_.tolist(),
        'cluster_hierarchy_': model.cluster_hierarchy_.tolist(),
        'n_features_in_': model.n_features_in_,
        'params': model.get_params()
    }

    if 'feature_names_in' in model.__dict__:
        serialized_model['feature_names_in'] = model.feature_names_in.tolist()

    return serialized_model


def deserialize_optics(model_dict):
    model = OPTICS(**model_dict['params'])

    model.labels_ = np.array(model_dict['labels_'])
    model.reachability_ = np.array(model_dict['reachability_'])
    model.ordering_ = np.array(model_dict['ordering_'])
    model.core_distances_ = np.array(model_dict['core_distances_'])
    model.predecessor_ = np.array(model_dict['predecessor_'])
    model.cluster_hierarchy_ = np.array(model_dict['cluster_hierarchy_'])
    model.n_features_in_ = model_dict['n_features_in_']

    if 'feature_names_in' in model_dict.keys():
        model.feature_names_in = np.array(model_dict['feature_names_in'])

    return model


def serialize_spectral_clustering(model):
    serialized_model = {
        'meta': 'spectral-clustering',
        'affinity_matrix_': model.affinity_matrix_.tolist(),
        'labels_': model.labels_.tolist(),
        'n_features_in_': model.n_features_in_,
        'params': model.get_params()
    }

    if 'feature_names_in' in model.__dict__:
        serialized_model['feature_names_in'] = model.feature_names_in.tolist()

    return serialized_model


def deserialize_spectral_clustering(model_dict):
    model = SpectralClustering(**model_dict['params'])

    model.affinity_matrix_ = np.array(model_dict['affinity_matrix_'])
    model.labels_ = np.array(model_dict['labels_'])
    model.n_features_in_ = model_dict['n_features_in_']

    if 'feature_names_in' in model_dict.keys():
        model.feature_names_in = np.array(model_dict['feature_names_in'])

    return model

def serialize_feature_agglomeration(model):
    params = model.get_params()
    serialized_model = {
        'meta': 'feature-agglomeration',
        'n_clusters_': model.n_clusters_,
        'labels_': model.labels_.tolist(),
        'n_leaves_': model.n_leaves_,
        'n_features_in_': model.n_features_in_,
        'children_': model.children_.tolist(),
        'pooling_func': (inspect.getmodule(params['pooling_func']).__name__,
                         params['pooling_func'].__name__),
        '_n_features_out': model._n_features_out,
        'n_connected_components_': model.n_connected_components_,
        'params': {key: value for key, value in params.items() if key != 'pooling_func'}
    }

    if 'feature_names_in' in model.__dict__:
        serialized_model['feature_names_in'] = model.feature_names_in.tolist()
    if 'distances_' in model.__dict__:
        serialized_model['distances_'] = model.feature_names_in.tolist()

    return serialized_model

def deserialize_feature_agglomeration(model_dict):
    params = model_dict['params']
    params['pooling_func'] = getattr(importlib.import_module(model_dict['pooling_func'][0]), model_dict['pooling_func'][1])
    model = FeatureAgglomeration(**params)

    model.n_clusters_ = model_dict['n_clusters_']
    model.labels_ = np.array(model_dict['labels_'])
    model.n_leaves_ = model_dict['n_leaves_']
    model.n_features_in_ = model_dict['n_features_in_']
    model.children_ = np.array(model_dict['children_'])
    model._n_features_out = model_dict['_n_features_out']
    model.n_connected_components_ = model_dict['n_connected_components_']

    if 'feature_names_in' in model_dict.keys():
        model.feature_names_in = np.array(model_dict['feature_names_in'])
    if 'distances_' in model_dict.keys():
        model.feature_names_in = np.array(model_dict['distances_'])

    return model
