# -*- coding: utf-8 -*-

import importlib
import inspect

import numpy as np
import sklearn
from sklearn.cluster import (AffinityPropagation, AgglomerativeClustering,
                             Birch, DBSCAN, FeatureAgglomeration, KMeans,
                             BisectingKMeans, MiniBatchKMeans, MeanShift, OPTICS,
                             SpectralClustering, SpectralBiclustering, SpectralCoclustering)
from sklearn.cluster._bisect_k_means import _BisectingTree

# Allow additional dependencies to be optional
__optionals__ = []
try:
    from kmodes.kmodes import KModes
    from kmodes.kprototypes import KPrototypes
    __optionals__.extend(['KModes', 'KPrototypes'])
except:
    pass
try:
    from hdbscan import HDBSCAN
    __optionals__.append('HDBSCAN')
except:
    pass


from . import _base
from .utils.random_state import serialize_random_state, deserialize_random_state
from .utils.memory import serialize_memory, deserialize_memory


def serialize_kmeans(model):
    serialized_model = {
        'meta': 'kmeans',
        'cluster_centers_': model.cluster_centers_.tolist(),
        'labels_': model.labels_.tolist(),
        'inertia_': model.inertia_,
        '_tol': float(model._tol),
        '_n_init': model._n_init,
        '_n_threads': model._n_threads,
        'n_iter_': model.n_iter_,
        'n_features_in_': model.n_features_in_,
        '_n_features_out': model._n_features_out,
        '_algorithm': model._algorithm,
        'params': model.get_params(),
    }

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()
    serialized_model['params']['n_clusters'] = int(serialized_model['params']['n_clusters'])

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

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

    return model


def serialize_minibatch_kmeans(model):
    serialized_model = {
        'meta': 'minibatch-kmeans',
        'cluster_centers_': model.cluster_centers_.tolist(),
        'labels_': model.labels_.tolist(),
        'inertia_': model.inertia_,
        '_ewa_inertia': model._ewa_inertia,
        '_ewa_inertia_min': model._ewa_inertia_min,
        '_counts': model._counts.tolist(),
        '_tol': float(model._tol),
        '_n_init': model._n_init,
        '_init_size': model._init_size,
        '_n_threads': model._n_threads,
        '_batch_size': model._batch_size,
        'n_iter_': model.n_iter_,
        'n_steps_': model.n_steps_,
        'n_features_in_': model.n_features_in_,
        '_n_features_out': model._n_features_out,
        '_n_since_last_reassign': model._n_since_last_reassign,
        '_no_improvement': model._no_improvement,
        'params': model.get_params(),
    }

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()
    serialized_model['params']['n_clusters'] = int(serialized_model['params']['n_clusters'])

    return serialized_model


def deserialize_minibatch_kmeans(model_dict):
    model = MiniBatchKMeans(**model_dict['params'])

    model.cluster_centers_ = np.array(model_dict['cluster_centers_'])
    model.labels_ = np.array(model_dict['labels_'])
    model.inertia_ = model_dict['inertia_']
    model._ewa_inertia = model_dict['_ewa_inertia']
    model._ewa_inertia_min = model_dict['_ewa_inertia_min']
    model._counts = np.array(model_dict['_counts'])
    model._tol = model_dict['_tol']
    model._n_init = model_dict['_n_init']
    model._init_size = model_dict['_init_size']
    model._n_threads = model_dict['_n_threads']
    model._batch_size = model_dict['_batch_size']
    model.n_iter_ = model_dict['n_iter_']
    model.n_steps_ = model_dict['n_steps_']
    model._n_since_last_reassign = model_dict['_n_since_last_reassign']
    model._no_improvement = model_dict['_no_improvement']
    model.n_features_in_ = model_dict['n_features_in_']
    model._n_features_out = model_dict['_n_features_out']

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

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

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist(),

    return serialized_model


def deserialize_affinity_propagation(model_dict):
    model = AffinityPropagation(**model_dict['params'])

    model.cluster_centers_indices_ = np.array(model_dict['cluster_centers_indices_'], dtype=np.int64)
    model.cluster_centers_ = np.array(model_dict['cluster_centers_'])
    model.labels_ = np.array(model_dict['labels_'], dtype=np.int64)
    model.affinity_matrix_ = np.array(model_dict['affinity_matrix_'])
    model.n_iter_ = model_dict['n_iter_']
    model.n_features_in_ = model_dict['n_features_in_']

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

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

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()
    if 'distances_' in model.__dict__:
        serialized_model['distances_'] = model.distances_.tolist()
    if '_metric' in model.__dict__:
        serialized_model['_metric'] = model._metric

    return serialized_model


def deserialize_agglomerative_clustering(model_dict):
    model = AgglomerativeClustering(**model_dict['params'])

    model.n_clusters_ = model_dict['n_clusters_']
    model.labels_ = np.array(model_dict['labels_'])
    model.n_leaves_ = model_dict['n_leaves_']
    model.n_connected_components_ = model_dict['n_connected_components_']
    model.n_features_in_ = model_dict['n_features_in_']
    model.children_ = np.array(model_dict['children_'])

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])
    if 'distances_' in model_dict.keys():
        model.distances_ = np.array(model_dict['distances_'])
    if '_metric' in model_dict:
        model._metric = model_dict['_metric']

    return model


def serialize_birch(model):
    return _base.serialize_model_generic(model, meta='birch')


def deserialize_birch(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_dbscan(model):
    serialized_model = {
        'meta': 'dbscan',
        'components_': model.components_.tolist(),
        'core_sample_indices_': model.core_sample_indices_.tolist(),
        'labels_': model.labels_.tolist(),
        'n_features_in_': model.n_features_in_,
        '_estimator_type': getattr(model, '_estimator_type', model.__sklearn_tags__().estimator_type),
        'params': model.get_params()
    }

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()

    return serialized_model


def deserialize_dbscan(model_dict):
    model = DBSCAN(**model_dict['params'])

    model.components_ = np.array(model_dict['components_'])
    model.labels_ = np.array(model_dict['labels_'])
    model.core_sample_indices_ = model_dict['core_sample_indices_']
    model.n_features_in_ = model_dict['n_features_in_']
    if not hasattr(model, '_estimator_type'):
        model._estimator_type = model_dict['_estimator_type']

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

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

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()

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

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

    return model


def serialize_spectral_clustering(model):
    serialized_model = {
        'meta': 'spectral-clustering',
        'affinity_matrix_': model.affinity_matrix_.tolist(),
        'labels_': model.labels_.tolist(),
        'n_features_in_': model.n_features_in_,
        'params': model.get_params()
    }

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()

    return serialized_model


def deserialize_spectral_clustering(model_dict):
    model = SpectralClustering(**model_dict['params'])

    model.affinity_matrix_ = np.array(model_dict['affinity_matrix_'])
    model.labels_ = np.array(model_dict['labels_'])
    model.n_features_in_ = model_dict['n_features_in_']

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

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

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()
    if 'distances_' in model.__dict__:
        serialized_model['distances_'] = model.feature_names_in_.tolist()
    if '_metric' in model.__dict__:
        serialized_model['_metric'] = model._metric

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

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])
    if 'distances_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['distances_'])
    if '_metric' in model_dict:
        model._metric = model_dict['_metric']

    return model


def serialize_meanshift(model):
    serialized_model = {
        'meta': 'meanshift',
        'cluster_centers_': model.cluster_centers_.tolist(),
        'labels_': model.labels_.tolist(),
        'n_iter_': model.n_iter_,
        'n_features_in_': model.n_features_in_,
        'params': model.get_params()
    }

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()

    return serialized_model


def deserialize_meanshift(model_dict):
    model = MeanShift(**model_dict['params'])

    model.cluster_centers_ = np.array(model_dict['cluster_centers_'])
    model.labels_ = np.array(model_dict['labels_'])
    model.n_iter_ = model_dict['n_iter_']
    model.n_features_in_ = model_dict['n_features_in_']

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

    return model


def serialize_spectral_biclustering(model):
    serialized_model = {
        'meta': 'spectral-biclustering',
        'rows_': model.rows_.tolist(),
        'columns_': model.columns_.tolist(),
        'row_labels_': model.row_labels_.tolist(),
        'n_features_in_': model.n_features_in_,
        'params': model.get_params()
    }

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()
    if 'columns_labels_' in model.__dict__:
        serialized_model['columns_labels_'] = model.columns_labels_.tolist()

    return serialized_model


def deserialize_spectral_biclustering(model_dict):
    model = SpectralBiclustering(**model_dict['params'])

    model.rows_ = np.array(model_dict['rows_'])
    model.columns_ = np.array(model_dict['columns_'])
    model.row_labels_ = np.array(model_dict['row_labels_'])
    model.n_features_in_ = model_dict['n_features_in_']

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])
    if 'columns_labels_' in model_dict.keys():
        model.columns_labels_ = np.array(model_dict['columns_labels_'])

    return model


def serialize_spectral_coclustering(model):
    serialized_model = {
        'meta': 'spectral-coclustering',
        'rows_': model.rows_.tolist(),
        'columns_': model.columns_.tolist(),
        'row_labels_': model.row_labels_.tolist(),
        'n_features_in_': model.n_features_in_,
        'params': model.get_params()
    }

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()
    if 'columns_labels_' in model.__dict__:
        serialized_model['columns_labels_'] = model.columns_labels_.tolist()

    return serialized_model


def deserialize_spectral_coclustering(model_dict):
    model = SpectralCoclustering(**model_dict['params'])

    model.rows_ = np.array(model_dict['rows_'])
    model.columns_ = np.array(model_dict['columns_'])
    model.row_labels_ = np.array(model_dict['row_labels_'])
    model.n_features_in_ = model_dict['n_features_in_']

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])
    if 'columns_labels_' in model_dict.keys():
        model.columns_labels_ = np.array(model_dict['columns_labels_'])

    return model


if 'KModes' in __optionals__:
    def serialize_kmodes(model):
        params = model.get_params()
        params['cat_dissim'] = (inspect.getmodule(params['cat_dissim']).__name__,
                                params['cat_dissim'].__name__)

        serialized_model = {
            'meta': 'kmodes',
            '_enc_cluster_centroids': model._enc_cluster_centroids.astype(int).tolist(),
            'labels_': model.labels_.tolist(),
            'cost_': float(model.cost_),
            'n_iter_': model.n_iter_,
            'epoch_costs_': [float(x) for x in model.epoch_costs_],
            '_enc_map': [{float(key): value for key, value in subdict.items()} for subdict in model._enc_map],
            'params': params
        }

        return serialized_model


    def deserialize_kmodes(model_dict):
        params = model_dict['params']
        params['cat_dissim'] = getattr(importlib.import_module(params['cat_dissim'][0]),
                                       params['cat_dissim'][1])

        model = KModes(**params)

        model._enc_cluster_centroids = np.array(model_dict['_enc_cluster_centroids'])
        model.labels_ = np.array(model_dict['labels_'])
        model.cost_ = model_dict['cost_']
        model.n_iter_ = model_dict['n_iter_']
        model.epoch_costs_ = model_dict['epoch_costs_']
        model._enc_map = [{np.float64(key): value for key, value in subdict.items()} for subdict in model_dict['_enc_map']]

        return model


if 'KPrototypes' in __optionals__:
    def serialize_kprototypes(model):
        params = model.get_params()

        params['cat_dissim'] = (inspect.getmodule(params['cat_dissim']).__name__,
                                params['cat_dissim'].__name__)
        params['num_dissim'] = (inspect.getmodule(params['num_dissim']).__name__,
                                params['num_dissim'].__name__)

        params['gamma'] = float(params['gamma'])

        serialized_model = {
            'meta': 'kprototypes',
            '_enc_cluster_centroids': np.array(model._enc_cluster_centroids).astype(float).tolist(),
            'labels_': model.labels_.tolist(),
            'cost_': float(model.cost_),
            'n_iter_': model.n_iter_,
            'epoch_costs_': [float(x) for x in model.epoch_costs_],
            '_enc_map': [{int(key): value for key, value in subdict.items()} for subdict in model._enc_map],
            'params': params
        }

        return serialized_model


    def deserialize_kprototypes(model_dict):
        params = model_dict['params']
        params['cat_dissim'] = getattr(importlib.import_module(params['cat_dissim'][0]),
                                       params['cat_dissim'][1])
        params['num_dissim'] = getattr(importlib.import_module(params['num_dissim'][0]),
                                       params['num_dissim'][1])
        params['gamma'] = np.float64(params['gamma'])

        model = KPrototypes(**params)

        model._enc_cluster_centroids = np.array(model_dict['_enc_cluster_centroids'])
        model.labels_ = np.array(model_dict['labels_'])
        model.cost_ = model_dict['cost_']
        model.n_iter_ = model_dict['n_iter_']
        model.epoch_costs_ = model_dict['epoch_costs_']
        model._enc_map = [{np.int32(key): value for key, value in subdict.items()} for subdict in model_dict['_enc_map']]

        return model


def serialize_bisecting_tree(model):
    if model is None:
        return None

    serialized_model = {
        'meta': 'bisecting-tree',
        'center': model.center.tolist() if isinstance(model.center, np.ndarray) else model.center,
        'indices': model.indices.tolist() if isinstance(model.indices, np.ndarray) else model.indices,
        'score': model.score,
        'left': serialize_bisecting_tree(model.left),
        'right': serialize_bisecting_tree(model.right),
    }

    if hasattr(model, 'label'):
        serialized_model['label'] = model.label

    return serialized_model


def deserialize_bisecting_tree(model_dict):
    if model_dict is None:
        return None

    model = _BisectingTree(np.array(model_dict['center']) if isinstance(model_dict['center'], list) else model_dict['center'],
                           np.array(model_dict['indices']) if isinstance(model_dict['indices'], list) else model_dict['indices'],
                           model_dict['score'])

    model.left = deserialize_bisecting_tree(model_dict['left'])
    model.right = deserialize_bisecting_tree(model_dict['right'])

    if 'label' in model_dict:
        model.label = model_dict['label']

    return model


def serialize_bisecting_kmeans(model):
    serialized_model = {
        'meta': 'bisecting-kmeans',
        'cluster_centers_': model.cluster_centers_.tolist(),
        'labels_': model.labels_.tolist(),
        'inertia_': model.inertia_,
        '_tol': float(model._tol),
        '_n_init': model._n_init,
        '_n_threads': model._n_threads,
        'n_features_in_': model.n_features_in_,
        '_n_features_out': model._n_features_out,
        '_X_mean': model._X_mean.tolist(),
        '_bisecting_tree': serialize_bisecting_tree(model._bisecting_tree),
        '_kmeans_single': (inspect.getmodule(model._kmeans_single).__name__,
                           model._kmeans_single.__name__),
        '_random_state': serialize_random_state(model._random_state),
        'params': model.get_params(),
    }

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()
    serialized_model['params']['n_clusters'] = int(serialized_model['params']['n_clusters'])

    return serialized_model


def deserialize_bisecting_kmeans(model_dict):
    model = BisectingKMeans(**model_dict['params'])
    model.cluster_centers_ = np.array(model_dict['cluster_centers_'])
    model.labels_ = np.array(model_dict['labels_'])
    model.inertia_ = model_dict['inertia_']
    model._tol = model_dict['_tol']
    model._n_init = model_dict['_n_init']
    model._n_threads = model_dict['_n_threads']
    model.n_features_in_ = model_dict['n_features_in_']
    model._n_features_out = model_dict['_n_features_out']
    model._X_mean = np.array(model_dict['_X_mean'])
    model._bisecting_tree = deserialize_bisecting_tree(model_dict['_bisecting_tree'])
    model._kmeans_single = getattr(importlib.import_module(model_dict['_kmeans_single'][0]),
                                   model_dict['_kmeans_single'][1])
    model._random_state = deserialize_random_state(model_dict['_random_state'])

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

    return model


if 'HDBSCAN' in __optionals__:
    def serialize_hdbscan(model):
        serialized_model = {
            'meta': 'hdbscan',
            '_metric_kwargs': model._metric_kwargs,
            '_condensed_tree': model._condensed_tree.tolist(),
            '_condensed_tree_dtype': f"np.dtype({str(model._condensed_tree.dtype)})",
            '_single_linkage_tree': model._single_linkage_tree.tolist(),
            '_raw_data': model._raw_data.tolist(),
            '_all_finite': bool(model._all_finite),
            'labels_': model.labels_.tolist(),
            'probabilities_': model.probabilities_.tolist(),
            'cluster_persistence_': model.cluster_persistence_.tolist(),
            'params': model.get_params()
        }

        if 'feature_names_in_' in model.__dict__:
            serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()
        if '_min_spanning_tree' in model.__dict__ and model._min_spanning_tree is not None:
            serialized_model['_min_spanning_tree'] = model._min_spanning_tree.tolist()
        else:
            serialized_model['_min_spanning_tree'] = model._min_spanning_tree
        if '_outlier_scores' in model.__dict__:
            serialized_model['_outlier_scores'] = model._outlier_scores
        if '_prediction_data' in model.__dict__:
            serialized_model['_prediction_data'] = model._prediction_data
        if '_relative_validity' in model.__dict__:
            serialized_model['_relative_validity'] = model._relative_validity

        if serialized_model['params']['memory'] is not None:
            serialized_model['params']['memory'] = serialize_memory(serialized_model['params']['memory'])

        return serialized_model


    def deserialize_hdbscan(model_dict):

        if model_dict['params']['memory'] is not None:
            model_dict['params']['memory'] = deserialize_memory(model_dict['params']['memory'])

        model = HDBSCAN(**model_dict['params'])

        model._metric_kwargs = model_dict['_metric_kwargs']
        model._condensed_tree = np.array(list(map(tuple, model_dict['_condensed_tree'])),
                                         dtype=eval(model_dict['_condensed_tree_dtype']))
        model._single_linkage_tree = np.array(model_dict['_single_linkage_tree'])
        model._raw_data = np.array(model_dict['_raw_data'])
        model._all_finite = model_dict['_all_finite']
        model.labels_ = np.array(model_dict['labels_'])
        model.probabilities_ = np.array(model_dict['probabilities_'])
        model.cluster_persistence_ = np.array(model_dict['cluster_persistence_'])

        if 'feature_names_in_' in model_dict.keys():
            model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])
        if '_min_spanning_tree' in model_dict:
            model._min_spanning_tree = np.array(model_dict['_min_spanning_tree'])
        if '_outlier_scores' in model_dict:
            model._outlier_scores = np.array(model_dict['_outlier_scores'])
        if '_prediction_data' in model_dict:
            model._prediction_data = np.array(model_dict['_prediction_data'])
        if '_relative_validity' in model_dict:
            model._relative_validity = np.array(model_dict['_relative_validity'])

        return model
