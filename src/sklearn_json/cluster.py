# -*- coding: utf-8 -*-

import numpy as np
from sklearn.cluster import AffinityPropagation, KMeans


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
