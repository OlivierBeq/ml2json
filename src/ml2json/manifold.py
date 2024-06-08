# -*- coding: utf-8 -*-

import scipy
import numpy as np
from sklearn.manifold import (Isomap, LocallyLinearEmbedding,
                              MDS, SpectralEmbedding, TSNE)
from sklearn.utils import check_random_state

# Allow additional dependencies to be optional
__optionals__ = []
try:
    from umap import UMAP
    __optionals__.append('UMAP')
except:
    pass

try:
    from openTSNE import (TSNE as OpenTSNE, TSNEEmbedding as OpenTNSEEmbedding,
                          PartialTSNEEmbedding  as OpenPartialTSNEEmbedding)
    from openTSNE.sklearn import TSNE as OpenTSNEsklearn
    from openTSNE.affinity import (PerplexityBasedNN, FixedSigmaNN,
                                   Multiscale, MultiscaleMixture,
                                   Uniform, PrecomputedAffinities)
    from openTSNE.nearest_neighbors import (Sklearn as OpentTSNESklearnNN,
                                            NNDescent as OpentTSNENNDescentNN, HNSW as OpentTSNEHNSWNN,
                                            PrecomputedDistanceMatrix as OpentTSNEPrecomputedDistanceMatrix,
                                            PrecomputedNeighbors as OpentTSNEPrecomputedNeighbors)
    from openTSNE.tsne import gradient_descent as OpenTSNEGradientDescentOptimizer
    __optionals__.append('OpenTSNE')
except:
    pass

from .decomposition import serialize_kernel_pca, deserialize_kernel_pca
from .neighbors import (serialize_nearest_neighbors, deserialize_nearest_neighbors,
                        __optionals__ as __neig_optionals__)
from .utils.csr import serialize_csr_matrix, deserialize_csr_matrix
from .utils.random_state import serialize_random_state, deserialize_random_state

if 'NNDescent' in __neig_optionals__:
    from .neighbors import serialize_nndescent, deserialize_nndescent


def serialize_tsne(model):
    serialized_model = {
        'meta': 'tsne',
        'embedding_': model.embedding_.tolist(),
        'kl_divergence_': model.kl_divergence_,
        'n_features_in_': model.n_features_in_,
        'n_iter_': model.n_iter_,
        'params': model.get_params()
    }

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()
    if '_init' in model.__dict__:
        serialized_model['_init'] = model._init.tolist()
    if '_learning_rate' in model.__dict__:
        serialized_model['_learning_rate'] = model._learning_rate.tolist()
    if 'learning_rate_' in model.__dict__:
        serialized_model['learning_rate_'] = model.learning_rate_.tolist()

    return serialized_model


def deserialize_tsne(model_dict):
    model = TSNE(**model_dict['params'])

    model.embedding_ = np.array(model_dict['embedding_'])
    model.kl_divergence_ = model_dict['kl_divergence_']
    model.n_features_in_ = model_dict['n_features_in_']
    model.n_iter_ = model_dict['n_iter_']

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])
    if '_init' in model_dict.keys():
        model._init = np.array(model_dict['_init'])
    if '_learning_rate' in model_dict.keys():
        model._learning_rate = np.array(model_dict['_learning_rate'])
    if 'learning_rate_' in model_dict.keys():
        model.learning_rate_ = np.array(model_dict['learning_rate_'])

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
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

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
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

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
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

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
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

    return model

if 'UMAP' in __optionals__ and 'NNDescent' in __neig_optionals__:
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
            model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

        return model


if 'OpenTSNE' in __optionals__:
    def serialize_opentsne(model):
        serialized_model = {
            'meta': 'openTSNE',
            'params': model.get_params()
        }
        if hasattr(model, 'embedding_'):
            serialized_model['embedding_'] = serialize_opentsne_embedding(model.embedding_)

        return serialized_model


    def deserialize_opentsne(model_dict):
        if 'embedding_' in model_dict:
            model = OpenTSNEsklearn(**model_dict['params'])
            model.embedding_ = deserialize_opentsne_embedding(model_dict['embedding_'])
        else:
            model = OpenTSNE(**model_dict['params'])
        return model


    def serialize_opentsne_embedding(model):
        serialized_model = {
            'meta': 'openTSNEEmbedding',
            'value': model.__array__().tolist(),
            'affinities': serialize_opentsne_affinities(model.affinities),
            'optimizer': serialize_opentsne_optimizer(model.optimizer),
            'params': {key: value for key, value in model.__dict__.items()
                       if key not in ['affinities', 'optimizer']}
        }
        serialized_model['params']['kl_divergence'] = float(serialized_model['params']['kl_divergence'])
        return serialized_model


    def deserialize_opentsne_embedding(model_dict):
        model = OpenTNSEEmbedding(embedding=np.array(model_dict['value']),
                                  affinities=deserialize_opentsne_affinities(model_dict['affinities']),
                                  random_state=model_dict['params']['random_state'],
                                  optimizer=deserialize_opentsne_optimizer(model_dict['optimizer']),
                                  **model_dict['params']['gradient_descent_params']
                                  )
        if 'kl_divergence' in model_dict:
            model.kl_divergence = model_dict['kl_divergence']
        if 'interp_coeffs' in model_dict:
            model.interp_coeffs = model_dict['interp_coeffs']
        if 'box_x_lower_bounds' in model_dict:
            model.box_x_lower_bounds = model_dict['box_x_lower_bounds']
        if 'box_y_lower_bounds' in model_dict:
            model.box_y_lower_bounds = model_dict['box_y_lower_bounds']
        return model


    def serialize_opentsne_partial_embedding(model):
        serialized_model = {
            'meta': 'openTSNEPartialEmbedding',
            'value': model.__array__().tolist(),
            'reference_embedding': model.reference_embedding.tolist(),
            'P': serialize_csr_matrix(model.P),
            'optimizer': serialize_opentsne_optimizer(model.optimizer),
            'gradient_descent_params': {key: value for key, value in model.gradient_descent_params.items()},
            'kl_divergence': model.kl_divergence
        }
        return serialized_model


    def deserialize_opentsne_partial_embedding(model_dict):
        model = OpenPartialTSNEEmbedding(embedding=model_dict['value'],
                                         reference_embedding=model_dict['reference_embedding'],
                                         P=model_dict['P'],
                                         optimizer=deserialize_opentsne_optimizer(model_dict['optimizer']),
                                         **model_dict['gradient_descent_params'])
        model.kl_divergence = model_dict['kl_divergence']
        return model


    def serialize_opentsne_affinities(model):
        affinity_type = type(model).__name__
        serialized_model = {
            'meta': 'openTSNEAffinities',
            'type': affinity_type,
            'P': serialize_csr_matrix(model.P),
            'verbose': model.verbose,
            'knn_index': serialize_opentsne_knnindex(model.knn_index),
            'n_jobs': model.n_jobs
        }

        if affinity_type == 'PerplexityBasedNN':
            serialized_model['_PerplexityBasedNN__neighbors'] = model._PerplexityBasedNN__neighbors.tolist()
            serialized_model['_PerplexityBasedNN__distances'] = model._PerplexityBasedNN__distances.tolist()
            serialized_model['perplexity'] = model.perplexity
            serialized_model['effective_perplexity_'] = model.effective_perplexity_
            serialized_model['symmetrize'] = model.symmetrize
        elif affinity_type == 'FixedSigmaNN':
            serialized_model['sigma'] = model.sigma
        elif affinity_type in ['MultiscaleMixture', 'Multiscale']:
            serialized_model['_MultiscaleMixture__neighbors'] = getattr(model, '_MultiscaleMixture__neighbors').tolist()
            serialized_model['_MultiscaleMixture__distances'] = getattr(model, '_MultiscaleMixture__distances').tolist()
            serialized_model['perplexities'] = model.perplexities
            serialized_model['effective_perplexities_'] = model.effective_perplexities_
            serialized_model['symmetrize'] = model.symmetrize
        elif affinity_type in ['Uniform', 'PrecomputedAffinities']:
            pass
        else:
            raise NotImplementedError(f'Affinity serialization of the type {type(model)} is not supported')

        return serialized_model


    def get_opentsne_nn_method(knn_index_type) -> str:
        if knn_index_type  == 'HNSW':
            return 'hnsw'
        elif knn_index_type == 'Sklearn':
            return 'exact'
        elif knn_index_type == 'NNDescent':
            return "pynndescent"
        raise TypeError(f'Unknown KNNIndex type: {knn_index_type}')


    def deserialize_opentsne_affinities(model_dict):
        if model_dict['type'] == 'MultiscaleMixture':
            model = MultiscaleMixture(data=None,
                                      perplexities=model_dict['perplexities'],
                                      method=get_opentsne_nn_method(model_dict['knn_index']['type']),
                                      metric=model_dict['knn_index']['metric'],
                                      metric_params=model_dict['knn_index']['metric_params'],
                                      symmetrize=model_dict['symmetrize'],
                                      n_jobs=model_dict['n_jobs'],
                                      random_state=model_dict['knn_index']['random_state'],
                                      verbose=model_dict['verbose'],
                                      knn_index=deserialize_opentsne_knnindex(model_dict['knn_index']))
            model._MultiscaleMixture__neighbors = np.array(model_dict['_MultiscaleMixture__neighbors'])
            model._MultiscaleMixture__distances = np.array(model_dict['_MultiscaleMixture__distances'])
            model.effective_perplexities_ = model_dict['effective_perplexities_']
        elif model_dict['type'] == 'Multiscale':
            model = Multiscale(data=None,
                               perplexities=model_dict['perplexities'],
                               method=get_opentsne_nn_method(model_dict['knn_index']['type']),
                               metric=model_dict['knn_index']['metric'],
                               metric_params=model_dict['knn_index']['metric_params'],
                               symmetrize=model_dict['symmetrize'],
                               n_jobs=model_dict['n_jobs'],
                               random_state=model_dict['knn_index']['random_state'],
                               verbose=model_dict['verbose'],
                               knn_index=deserialize_opentsne_knnindex(model_dict['knn_index']))
            model._MultiscaleMixture__neighbors = np.array(model_dict['_MultiscaleMixture__neighbors'])
            model._MultiscaleMixture__distances = np.array(model_dict['_MultiscaleMixture__distances'])
            model.effective_perplexities_ = model_dict['effective_perplexities_']
        elif model_dict['type'] == 'FixedSigmaNN':
            model = FixedSigmaNN(data=None,
                                 sigma=model_dict['sigma'],
                                 k=model_dict['knn_index']['k'],
                                 method=get_opentsne_nn_method(model_dict['knn_index']['type']),
                                 metric=model_dict['knn_index']['metric'],
                                 metric_params=model_dict['knn_index']['metric_params'],
                                 symmetrize=model_dict['symmetrize'],
                                 n_jobs=model_dict['n_jobs'],
                                 random_state=model_dict['knn_index']['random_state'],
                                 verbose=model_dict['verbose'],
                                 knn_index=deserialize_opentsne_knnindex(model_dict['knn_index'])
                                 )
        elif model_dict['type'] == 'PerplexityBasedNN':
            model = PerplexityBasedNN(data=None,
                                      perplexity=model_dict['perplexity'],
                                      method=get_opentsne_nn_method(model_dict['knn_index']['type']),
                                      metric=model_dict['knn_index']['metric'],
                                      metric_params=model_dict['knn_index']['metric_params'],
                                      symmetrize=model_dict['symmetrize'],
                                      n_jobs=model_dict['n_jobs'],
                                      random_state=model_dict['knn_index']['random_state'],
                                      verbose=model_dict['verbose'],
                                      k_neighbors=model_dict['effective_perplexity_'],
                                      knn_index=deserialize_opentsne_knnindex(model_dict['knn_index']))
        elif model_dict['type'] == 'Uniform':
            model = Uniform(data=None,
                            k_neighbors=model_dict['knn_index']['k'],
                            method=get_opentsne_nn_method(model_dict['knn_index']['type']),
                            metric=model_dict['knn_index']['metric'],
                            metric_params=model_dict['knn_index']['metric_params'],
                            symmetrize=model_dict['symmetrize'],
                            n_jobs=model_dict['n_jobs'],
                            random_state=model_dict['knn_index']['random_state'],
                            verbose=model_dict['verbose'],
                            knn_index=deserialize_opentsne_knnindex(model_dict['knn_index'])
                            )
        elif model_dict['type'] == PrecomputedAffinities:
            model = PrecomputedAffinities(deserialize_csr_matrix(model_dict['P']),
                                          normalize=False)
        else:
            raise TypeError(f'OpenTSNE affinity type is not supported: {model_dict["type"]}')

        if model_dict['type'] != PrecomputedAffinities:
            model.P = deserialize_csr_matrix(model_dict['P'])
        return model

    def serialize_opentsne_optimizer(model):
        serialized_model = {
            'meta': 'openTSNEGradientDescentOptimizer',
            'gains': model.gains.tolist(),
            'update': model.update.tolist(),
        }
        return serialized_model


    def deserialize_opentsne_optimizer(model_dict):
        model = OpenTSNEGradientDescentOptimizer()
        model.gains = np.array(model_dict['gains'])
        model.update = np.array(model_dict['update'])
        return model


    def serialize_opentsne_knnindex(model):
        index_type = type(model).__name__

        if index_type == 'Annoy':
            raise TypeError('openTSNE objects relying on the Annoy library are not supported. '
                            'Using PyNNDescent instead is recommended.')

        serialized_model = {
            'meta': 'openTSNEKnnIndex',
            'type': index_type,
            'data': model.data.tolist()
        }
        for param, value in model.__dict__.items():
            if param not in ['index', 'data', '_tmp_dirs']:
                serialized_model[param] = value

        if index_type == 'HNSW':
            serialized_model['state'] = model.__getstate__()
            serialized_model['state']['data'] = serialized_model['state']['data'].tolist()
            serialized_model['state']['b64_index'] = serialized_model['state']['b64_index'].decode()
        elif index_type  == 'Sklearn':
            serialized_model['state'] = serialize_nearest_neighbors(model.index)
        elif index_type == 'NNDescent':
            serialized_model['state'] = serialize_nndescent(model.index)
        return serialized_model

    def deserialize_opentsne_knnindex(model_dict):
        params = dict(data=np.array(model_dict['data']),
                      k=model_dict['k'],
                      metric=model_dict['metric'],
                      metric_params=model_dict['metric_params'],
                      n_jobs=model_dict['n_jobs'],
                      random_state=model_dict['random_state'],
                      verbose=model_dict['verbose'])
        if model_dict['type'] == 'Sklearn':
            model = OpentTSNESklearnNN(**params)
            model.index = deserialize_nearest_neighbors(model_dict['state'])
        elif model_dict['type'] == 'NNDescent':
            model = OpentTSNENNDescentNN(**params)
            model.index = deserialize_nndescent(model_dict['state'])
        elif model_dict['type'] == 'HNSW':
            model = OpentTSNEHNSWNN(**params)
            model_dict['state']['data'] = np.array(model_dict['state']['data'])
            model_dict['state']['b64_index'] = model_dict['state']['b64_index'].encode()
            model.__setstate__(model_dict['state'])
        elif model_dict['type'] == 'PrecomputedDistanceMatrix':
            # Load from dummy distance matrix
            model = OpentTSNEPrecomputedDistanceMatrix(distance_matrix=np.array([[1, 1], [1, 1]]),
                                                       k=model_dict['k'])
        elif model_dict['type'] == 'PrecomputedNeighbors':
            model = OpentTSNEPrecomputedNeighbors(neighbors=model_dict['indices'],
                                                  distances=model_dict['distances'])

        # Load other parameters
        for param, value in model_dict.items():
            if param not in list(params.keys()) + ['state']:
                setattr(model, param, value)

        return model
