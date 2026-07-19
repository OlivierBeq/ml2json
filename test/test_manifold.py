# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np
from sklearn.datasets import load_iris, load_digits, fetch_california_housing
from sklearn.manifold import (Isomap, LocallyLinearEmbedding,
                              MDS, SpectralEmbedding, TSNE)
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import kneighbors_graph

# Allow testing of additional optional dependencies
__optionals__ = []
try:
    from umap import UMAP
    from umap.umap_ import nearest_neighbors
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
    from openTSNE.nearest_neighbors import (Sklearn as OpentTSNESklearnNN, Annoy as OpentTSNEAnnoyNN,
                                            NNDescent as OpentTSNENNDescentNN, HNSW as OpentTSNEHNSWNN,
                                            PrecomputedDistanceMatrix as OpentTSNEPrecomputedDistanceMatrix,
                                            PrecomputedNeighbors as OpentTSNEPrecomputedNeighbors)
    from openTSNE.tsne import gradient_descent as OpenTSNEGradientDescentOptimizer
    __optionals__.append('OpenTSNE')
except:
    pass

from src import ml2json


class TestAPI(unittest.TestCase):

    def setUp(self):
        self.iris_data, _ = load_iris(return_X_y=True)
        self.digit_data, _ = load_digits(return_X_y=True)
        self.calhouse_data, _ = fetch_california_housing(return_X_y=True)
        self.calhouse_data = self.calhouse_data[:5000, :]

    def check_model(self, model, model_name, data):
        expected_ft = model.fit_transform(data)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_ft = deserialized_model.fit_transform(data)

            if not isinstance(actual_ft, tuple):
                np.testing.assert_array_almost_equal(expected_ft, actual_ft)
            else:
                for x, y in zip(expected_ft, actual_ft):
                    np.testing.assert_array_almost_equal(x, y)

    def test_tsne(self):
        self.check_model(TSNE(init='pca', learning_rate='auto'), 'tsne.json', self.iris_data)
        with self.assertRaises(AssertionError):
            self.check_model(TSNE(init='random', learning_rate='auto'), 'tsne.json', self.iris_data)

    def test_mds(self):
        self.check_model(MDS(random_state=1234), 'mds.json', self.iris_data)

    def test_isomap(self):
        self.check_model(Isomap(n_neighbors=50, n_components=10, neighbors_algorithm='kd_tree'), 'isomap.json', self.iris_data)
        self.check_model(Isomap(n_neighbors=50, n_components=10, neighbors_algorithm='brute'), 'isomap.json', self.iris_data)
        self.check_model(Isomap(n_neighbors=50, n_components=10, neighbors_algorithm='ball_tree'), 'isomap.json', self.iris_data)

    def test_locally_linear_embedding(self):
        self.check_model(LocallyLinearEmbedding(neighbors_algorithm='kd_tree'), 'locally-linear-embedding.json', self.iris_data)
        self.check_model(LocallyLinearEmbedding(neighbors_algorithm='brute'), 'locally-linear-embedding.json', self.iris_data)
        self.check_model(LocallyLinearEmbedding(neighbors_algorithm='ball_tree'), 'locally-linear-embedding.json', self.iris_data)

    def test_spectral_embedding(self):
        self.check_model(SpectralEmbedding(affinity='nearest_neighbors', random_state=1234, n_jobs=-1), 'spectral-embedding.json', self.digit_data)
        self.check_model(SpectralEmbedding(affinity='rbf', random_state=1234, n_jobs=-1), 'spectral-embedding.json', self.iris_data)

    def test_isomap_more_params(self):
        self.check_model(Isomap(n_neighbors=20, n_components=3, eigen_solver='dense',
                                path_method='D', p=1), 'isomap.json', self.iris_data)
        self.check_model(Isomap(n_neighbors=20, n_components=3, eigen_solver='arpack',
                                path_method='FW', metric='manhattan'), 'isomap.json', self.iris_data)
        dist = pairwise_distances(self.iris_data)
        self.check_model(Isomap(n_neighbors=20, n_components=3, metric='precomputed'),
                         'isomap.json', dist)

    def test_locally_linear_embedding_methods(self):
        for method in ['standard', 'hessian', 'modified', 'ltsa']:
            self.check_model(LocallyLinearEmbedding(n_neighbors=15, n_components=2, method=method,
                                                     eigen_solver='dense'),
                             'locally-linear-embedding.json', self.iris_data)
        self.check_model(LocallyLinearEmbedding(n_neighbors=15, n_components=2, method='standard',
                                                 eigen_solver='arpack', random_state=1234),
                         'locally-linear-embedding.json', self.iris_data)

    def test_mds_variants(self):
        self.check_model(MDS(random_state=1234, metric_mds=False), 'mds.json', self.iris_data)
        dist = pairwise_distances(self.iris_data)
        self.check_model(MDS(random_state=1234, metric='precomputed'), 'mds.json', dist)
        self.check_model(MDS(random_state=1234, normalized_stress=False), 'mds.json', self.iris_data)

    def test_spectral_embedding_more_params(self):
        affinity = rbf_kernel(self.iris_data)
        self.check_model(SpectralEmbedding(affinity='precomputed', random_state=1234),
                         'spectral-embedding.json', affinity)
        nn_graph = kneighbors_graph(self.iris_data, n_neighbors=10, mode='distance', include_self=True)
        self.check_model(SpectralEmbedding(affinity='precomputed_nearest_neighbors',
                                           random_state=1234, n_neighbors=10),
                         'spectral-embedding.json', nn_graph)
        self.check_model(SpectralEmbedding(affinity='rbf', eigen_solver='arpack', random_state=1234),
                         'spectral-embedding.json', self.iris_data)
        self.check_model(SpectralEmbedding(affinity='rbf', eigen_solver='lobpcg', random_state=1234),
                         'spectral-embedding.json', self.iris_data)

    def test_tsne_more_params(self):
        self.check_model(TSNE(init='pca', learning_rate='auto', method='exact', random_state=1234),
                         'tsne.json', self.iris_data)
        self.check_model(TSNE(init='pca', learning_rate='auto', metric='manhattan', random_state=1234),
                         'tsne.json', self.iris_data)
        dist = pairwise_distances(self.iris_data)
        self.check_model(TSNE(init='random', learning_rate='auto', metric='precomputed', random_state=1234),
                         'tsne.json', dist)
        self.check_model(TSNE(init='pca', learning_rate='auto', angle=0.8, random_state=1234),
                         'tsne.json', self.iris_data)

    def test_manifold_float32_input(self):
        data32 = self.iris_data.astype(np.float32)
        self.check_model(Isomap(n_neighbors=20, n_components=3), 'isomap.json', data32)
        self.check_model(LocallyLinearEmbedding(neighbors_algorithm='auto'), 'locally-linear-embedding.json', data32)
        self.check_model(MDS(random_state=1234), 'mds.json', data32)
        self.check_model(SpectralEmbedding(affinity='rbf', random_state=1234), 'spectral-embedding.json', data32)
        self.check_model(TSNE(init='pca', learning_rate='auto', random_state=1234), 'tsne.json', data32)

    def test_umap(self):
        if 'UMAP' in __optionals__:
            self.check_model(UMAP(random_state=1234, low_memory=False), 'umap.json', self.iris_data)
            self.check_model(UMAP(random_state=1234, output_dens=True, low_memory=False), 'umap.json', self.iris_data)
            precomputed_knn = nearest_neighbors(self.calhouse_data, 15, random_state=1234, metric='euclidean',
                                                metric_kwds={}, angular=False, verbose=False, low_memory=False)
            self.check_model(UMAP(n_neighbors=15, random_state=1234, metric='euclidean', output_dens=False,
                                  precomputed_knn=precomputed_knn, low_memory=False), 'umap.json',
                             self.calhouse_data)
            self.check_model(UMAP(n_neighbors=15, random_state=1234, metric='euclidean', output_dens=True,
                                  precomputed_knn=precomputed_knn, low_memory=False), 'umap.json',
                             self.calhouse_data)

    def check_opentsne_model(self, model, model_name, data, fit: bool = True):
        if fit:
            model = model.fit(data)
        expected_ft = model.transform(data)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_ft = deserialized_model.transform(data)

            if not isinstance(actual_ft, tuple):
                np.testing.assert_array_almost_equal(expected_ft, actual_ft)
            else:
                for x, y in zip(expected_ft, actual_ft):
                    np.testing.assert_array_almost_equal(x, y)

    def test_opentsne(self):
        if 'OpenTSNE' in __optionals__:
            # sklearn constructor
            for neighbors in ['exact', 'pynndescent', 'hnsw']:
                self.check_opentsne_model(OpenTSNEsklearn(random_state=1234, initialization='spectral',
                                                          neighbors=neighbors),
                                          'opentsne.json',
                                          self.iris_data)
                self.check_opentsne_model(OpenTSNEsklearn(random_state=1234, initialization='pca',
                                                          neighbors=neighbors),
                                          'opentsne.json',
                                          self.iris_data)
                self.check_opentsne_model(OpenTSNEsklearn(random_state=1234, initialization='random',
                                                          neighbors=neighbors),
                                          'opentsne.json',
                                          self.iris_data)
            with self.assertRaises(TypeError) as context:
                self.check_opentsne_model(OpenTSNEsklearn(random_state=1234, initialization='spectral',
                                                          neighbors='annoy'),
                                          'opentsne.json',
                                          self.iris_data)
                self.check_opentsne_model(OpenTSNEsklearn(random_state=1234, initialization='pca',
                                                          neighbors='annoy'),
                                          'opentsne.json',
                                          self.iris_data)
                self.check_opentsne_model(OpenTSNEsklearn(random_state=1234, initialization='random',
                                                          neighbors='annoy'),
                                          'opentsne.json',
                                          self.iris_data)
            # With custom affinities
            from openTSNE import initialization
            init = initialization.pca(self.iris_data, random_state=42, svd_solver='full')
            for method in ['exact', 'pynndescent', 'hnsw']:
                affinities_perplex = PerplexityBasedNN(self.iris_data, method=method, random_state=42)
                affinities_multimixt = MultiscaleMixture(self.iris_data, perplexities=[5, 10, 20, 30, 40], method=method, random_state=87)
                affinities_multi = Multiscale(self.iris_data, perplexities=[5, 10, 20, 30, 40], method=method, random_state=87)
                for affinity in [affinities_perplex, affinities_multimixt, affinities_multi]:
                    emb = OpenTNSEEmbedding(init, affinity)
                    # Early exaggeration
                    emb.optimize(n_iter=250, exaggeration=12, inplace=True)
                    # Regular optimization
                    emb.optimize(n_iter=500, inplace=True)
                    self.check_opentsne_model(emb, 'opentsne.json', self.iris_data, False)
