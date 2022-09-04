# -*- coding: utf-8 -*-

import scipy
import numpy as np
from sklearn.manifold import (Isomap, LocallyLinearEmbedding,
                              MDS, SpectralEmbedding, TSNE)
from sklearn.utils import check_random_state
from umap import UMAP

from .decomposition import serialize_kernel_pca, deserialize_kernel_pca
from .neighbors import (serialize_nearest_neighbors, deserialize_nearest_neighbors,
                        serialize_nndescent, deserialize_nndescent)
from .utils.csr import serialize_csr_matrix, deserialize_csr_matrix
from .utils.random_state import serialize_random_state, deserialize_random_state


def serialize_tsne(model):
    serialized_model = {
        'meta': 'tsne',
        'embedding_': model.embedding_.tolist(),
        'kl_divergence_': model.kl_divergence_,
        'n_features_in_': model.n_features_in_,
        'n_iter_': model.n_iter_,
        '_init': model._init,
        '_learning_rate': model._learning_rate,
        'params': model.get_params()
    }

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()

    return serialized_model


def deserialize_tsne(model_dict):
    model = TSNE(**model_dict['params'])

    model.embedding_ = np.array(model_dict['embedding_'])
    model.kl_divergence_ = model_dict['kl_divergence_']
    model.n_features_in_ = model_dict['n_features_in_']
    model.n_iter_ = model_dict['n_iter_']
    model._init = model_dict['_init']
    model._learning_rate = model_dict['_learning_rate']

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'])

    return model


def serialize_mds(model):
    serialized_model = {
        'meta': 'mds',
        'dissimilarity_matrix_': model.dissimilarity_matrix_.tolist(),
        'embedding_': model.embedding_.tolist(),
        'n_features_in_': model.n_features_in_,
        'n_iter_': model.n_iter_,
        'stress_': float(model.stress_),
        'params': model.get_params()
    }

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()

    return serialized_model


def deserialize_mds(model_dict):
    model = MDS(**model_dict['params'])

    model.dissimilarity_matrix_ = np.array(model_dict['dissimilarity_matrix_'])
    model.embedding_ = np.array(model_dict['embedding_'])
    model.n_features_in_ = model_dict['n_features_in_']
    model.n_iter_ = model_dict['n_iter_']
    model.stress_ = np.float64(model_dict['stress_'])

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'])

    return model


def serialize_isomap(model):
    serialized_model = {
        'meta': 'isomap',
        'embedding_': model.embedding_.tolist(),
        'dist_matrix_': model.dist_matrix_.tolist(),
        'n_features_in_': model.n_features_in_,
        '_n_features_out': model._n_features_out,
        'kernel_pca_': serialize_kernel_pca(model.kernel_pca_),
        'nbrs_': serialize_nearest_neighbors(model.nbrs_),
        'params': model.get_params()
    }

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()

    return serialized_model


def deserialize_isomap(model_dict):
    model = Isomap(**model_dict['params'])

    model.embedding_ = np.array(model_dict['embedding_'])
    model.dist_matrix_ = np.array(model_dict['dist_matrix_'])
    model.n_features_in_ = model_dict['n_features_in_']
    model._n_features_out = model_dict['_n_features_out']
    model.kernel_pca_ = deserialize_kernel_pca(model_dict['kernel_pca_'])
    model.nbrs_ = deserialize_nearest_neighbors(model_dict['nbrs_'])

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'])

    return model


def serialize_locally_linear_embedding(model):
    serialized_model = {
        'meta': 'locally-linear-embedding',
        'embedding_': model.embedding_.tolist(),
        'n_features_in_': model.n_features_in_,
        '_n_features_out': model._n_features_out,
        'reconstruction_error_': float(model.reconstruction_error_),
        'nbrs_': serialize_nearest_neighbors(model.nbrs_),
        'params': model.get_params()
    }

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()

    return serialized_model


def deserialize_locally_linear_embedding(model_dict):
    model = LocallyLinearEmbedding(**model_dict['params'])

    model.embedding_ = np.array(model_dict['embedding_'])
    model.n_features_in_ = model_dict['n_features_in_']
    model._n_features_out = model_dict['_n_features_out']
    model.reconstruction_error_ = np.float64(model_dict['reconstruction_error_'])
    model.nbrs_ = deserialize_nearest_neighbors(model_dict['nbrs_'])

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'])

    return model


def serialize_spectral_embedding(model):
    serialized_model = {
        'meta': 'spectral-embedding',
        'embedding_': model.embedding_.tolist(),
        'n_features_in_': model.n_features_in_,
        'params': model.get_params()
    }

    if 'n_neighbors_' in model.__dict__:
        serialized_model['n_neighbors_'] = model.n_neighbors_
    if scipy.sparse.issparse(model.affinity_matrix_):
        serialized_model['affinity_matrix_'] = serialize_csr_matrix(model.affinity_matrix_)
        serialized_model['affinity_matrix_type'] = 'sparse'
    else:
        serialized_model['affinity_matrix_'] = model.affinity_matrix_.tolist()
        serialized_model['affinity_matrix_type'] = 'dense'
    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()

    return serialized_model


def deserialize_spectral_embedding(model_dict):
    model = SpectralEmbedding(**model_dict['params'])

    model.embedding_ = np.array(model_dict['embedding_'])
    model.n_features_in_ = model_dict['n_features_in_']

    if model_dict['affinity_matrix_type'] == 'sparse':
        model.affinity_matrix_ = deserialize_csr_matrix(model_dict['affinity_matrix_'])
    else:
        model.affinity_matrix_ = np.array(model_dict['affinity_matrix_'])
    if 'n_neighbors_' in model_dict.keys():
        model.n_neighbors_ = model_dict['n_neighbors_']
    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'])

    return model

def serialize_umap(model):
    serialized_model = {
        'meta': 'umap',
        'graph_': serialize_csr_matrix(model.graph_),
        '_small_data': model._small_data,
        '_initial_alpha': model._initial_alpha,
        '_raw_data': model._raw_data.astype(float).tolist(),
        '_original_n_threads': model._original_n_threads,
        '_sparse_data': model._sparse_data,
        '_sigmas': model._sigmas.astype(float).tolist(),
        '_rhos': model._rhos.astype(float).tolist(),
        '_disconnection_distance': model._disconnection_distance,
        '_a': float(model._a),
        '_b': float(model._b),
        '_input_hash': model._input_hash,
        '_n_neighbors': model._n_neighbors,
        '_supervised': model._supervised,
        'embedding_': model.embedding_.astype(float).tolist(),
        'params': model.get_params()
    }

    if serialized_model['params']['precomputed_knn'] is not None and serialized_model['params']['precomputed_knn'][0] is not None:
        serialized_model['params']['precomputed_knn'] = (
            serialized_model['params']['precomputed_knn'][0].astype(float).tolist(),
            serialized_model['params']['precomputed_knn'][1].astype(float).tolist(),
            serialize_nndescent(serialized_model['params']['precomputed_knn'][2])
        )

    if model.graph_dists_ is not None:
        serialized_model['graph_dists_'] = serialize_csr_matrix(model.graph_dists_.tocsr())
    else:
        serialized_model['graph_dists_'] = None
    if model.knn_indices is not None:
        serialized_model['knn_indices'] = model.knn_indices.astype(float).tolist()
    else:
        serialized_model['knn_indices'] = None
    if model.knn_dists is not None:
        serialized_model['knn_dists'] = model.knn_dists.astype(float).tolist()
    else:
        serialized_model['knn_dists'] = None
    if model.knn_search_index is not None:
        serialized_model['knn_search_index'] = serialize_nndescent(model.knn_search_index)
    else:
        serialized_model['knn_search_index'] = None
    if 'rad_emb_' in model.__dict__:
        serialized_model['rad_emb_'] = model.rad_emb_.tolist()
    if 'rad_orig_' in model.__dict__:
        serialized_model['rad_orig_'] = model.rad_orig_.tolist()
    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()

    return serialized_model


def deserialize_umap(model_dict):

    if model_dict['params']['precomputed_knn'] is not None and model_dict['params']['precomputed_knn'][0] is not None:
        model_dict['params']['precomputed_knn'] = (
            np.array(model_dict['params']['precomputed_knn'][0], dtype=np.float32),
            np.array(model_dict['params']['precomputed_knn'][1], dtype=np.float32),
            deserialize_nndescent(model_dict['params']['precomputed_knn'][2])
        )

    model = UMAP(**model_dict['params'])

    model.graph_ = deserialize_csr_matrix(model_dict['graph_'])
    model._small_data = model_dict['_small_data']
    model._initial_alpha = model_dict['_initial_alpha']
    model._raw_data = np.array(model_dict['_raw_data'], dtype=np.float32)
    model._original_n_threads = model_dict['_original_n_threads']
    model._sparse_data = model_dict['_sparse_data']
    model._sigmas = np.array(model_dict['_sigmas'], dtype=np.float32)
    model._rhos = np.array(model_dict['_rhos'], dtype=np.float32)
    model._disconnection_distance = model_dict['_disconnection_distance']
    model._a = np.float64(model_dict['_a'])
    model._b = np.float64(model_dict['_b'])
    model._input_hash = model_dict['_input_hash']
    model._n_neighbors = model_dict['_n_neighbors']
    model._supervised = model_dict['_supervised']
    model.embedding_ = np.array(model_dict['embedding_'], dtype=np.float32)

    if model_dict['graph_dists_'] is not None:
        model.graph_dists_ = deserialize_csr_matrix(model_dict['graph_dists_']).todok()
    else:
        model.graph_dists_ = None


    if model_dict['knn_indices'] is not None:
        model.knn_indices = np.array(model_dict['knn_indices'], dtype=np.float32)
    else:
        model.knn_indices = None
    if model_dict['knn_dists'] is not None:
        model.knn_dists = np.array(model_dict['knn_dists'], dtype=np.float32)
    else:
        model.knn_dists = None
    if model_dict['knn_search_index'] is not None:
        model.knn_search_index = deserialize_nndescent(model_dict['knn_search_index'])
    else:
        model.knn_search_index = None

    if 'rad_emb_' in model_dict.keys():
        model.rad_emb_ = np.array(model_dict['rad_emb_'], dtype=np.float32)
    if 'rad_orig_' in model_dict.keys():
        model.rad_orig_ = np.array(model_dict['rad_orig_'], dtype=np.float32)
    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'])

    return model
