# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import load_iris
from sklearn.neighbors import (NearestNeighbors, KDTree, KernelDensity, BallTree, KNeighborsTransformer,
                               RadiusNeighborsTransformer, LocalOutlierFactor, NeighborhoodComponentsAnalysis)

# Allow testing of additional optional dependencies
__optionals__ = []
try:
    from pynndescent import NNDescent, PyNNDescentTransformer
    __optionals__.extend(['NNDescent', 'PyNNDescentTransformer'])
except:
    pass

from src import ml2json


class TestAPI(unittest.TestCase):

    def setUp(self):
        self.data, self.labels = load_iris(return_X_y=True)

    def check_nearest_neighbors_model(self, model, model_name):
        model.fit(self.data)

        rng = np.random.RandomState(1234)
        subset = self.data[rng.randint(self.data.shape[0], size=10)]
        expected_ft = model.kneighbors(subset)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_ft = deserialized_model.kneighbors(subset)

            np.testing.assert_array_almost_equal(expected_ft, actual_ft)

    def test_nearest_neighbors(self):
        self.check_nearest_neighbors_model(NearestNeighbors(), 'nearest-neighbors.json')

    def test_nearest_neighbors_algorithms(self):
        for algorithm in ['ball_tree', 'kd_tree', 'brute']:
            self.check_nearest_neighbors_model(NearestNeighbors(algorithm=algorithm),
                                               'nearest-neighbors-algo.json')

    def test_nearest_neighbors_metrics(self):
        for metric, params in [('manhattan', {}), ('chebyshev', {}), ('minkowski', {'p': 1}),
                               ('minkowski', {'p': 3})]:
            self.check_nearest_neighbors_model(NearestNeighbors(algorithm='brute', metric=metric,
                                                                 metric_params=params or None),
                                               'nearest-neighbors-metric.json')

    def test_nearest_neighbors_mahalanobis(self):
        cov = np.cov(self.data.T)
        model = NearestNeighbors(algorithm='brute', metric='mahalanobis', metric_params={'VI': np.linalg.inv(cov)})
        self.check_nearest_neighbors_model(model, 'nearest-neighbors-mahalanobis.json')

    def test_nearest_neighbors_precomputed(self):
        distances = squareform(pdist(self.data))
        model = NearestNeighbors(algorithm='brute', metric='precomputed')
        model.fit(distances)
        expected_ft = model.kneighbors(distances[:10])

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        model_name = 'nearest-neighbors-precomputed.json'
        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_ft = deserialized_model.kneighbors(distances[:10])
            np.testing.assert_array_almost_equal(expected_ft, actual_ft)

    def test_nearest_neighbors_dtypes(self):
        for dtype in [np.float32, np.float64]:
            data = self.data.astype(dtype)
            model = NearestNeighbors()
            model.fit(data)

            rng = np.random.RandomState(1234)
            subset = data[rng.randint(data.shape[0], size=10)]
            expected_ft = model.kneighbors(subset)

            serialized_dict_model = ml2json.to_dict(model)
            deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

            model_name = 'nearest-neighbors-dtype.json'
            ml2json.to_json(model, model_name)
            deserialized_json_model = ml2json.from_json(model_name)
            os.remove(model_name)

            for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
                actual_ft = deserialized_model.kneighbors(subset)
                np.testing.assert_array_almost_equal(expected_ft, actual_ft)

    def check_kernel_density_model(self, model, model_name):
        model.fit(self.data)

        rng = np.random.RandomState(1234)
        subset = self.data[rng.randint(self.data.shape[0], size=10)]
        expected_ft = model.score_samples(subset)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_ft = deserialized_model.score_samples(subset)

            np.testing.assert_array_almost_equal(expected_ft, actual_ft) 
        
    def test_kernel_density(self):
        self.check_kernel_density_model(KernelDensity(), 'kernel-density.json')

    def test_kernel_density_kernels(self):
        for kernel in ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']:
            self.check_kernel_density_model(KernelDensity(kernel=kernel, bandwidth=0.5), 'kernel-density-kernel.json')

    def test_kernel_density_algorithms(self):
        for algorithm in ['ball_tree', 'kd_tree', 'auto']:
            self.check_kernel_density_model(KernelDensity(algorithm=algorithm), 'kernel-density-algo.json')

    def check_kdtree_model(self, model, model_name):
        rng = np.random.RandomState(1234)
        subset = self.data[rng.randint(self.data.shape[0], size=10)]
        expected_ft_d, expected_ft_i = model.query(subset)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)


        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_ft_d, actual_ft_i = deserialized_model.query(subset)

            np.testing.assert_array_almost_equal(expected_ft_d, actual_ft_d)
            np.testing.assert_array_almost_equal(expected_ft_i, actual_ft_i)

    def test_kdtree(self):
        self.check_kdtree_model(KDTree(self.data), 'kd-tree.json')

    def test_balltree(self):
        self.check_kdtree_model(BallTree(self.data), 'ball-tree.json')

    def test_kdtree_metrics(self):
        for metric, kwargs in [('manhattan', {}), ('chebyshev', {}), ('minkowski', {'p': 3})]:
            self.check_kdtree_model(KDTree(self.data, metric=metric, **kwargs), 'kd-tree-metric.json')

    def test_balltree_metrics(self):
        for metric, kwargs in [('manhattan', {}), ('chebyshev', {}), ('minkowski', {'p': 3})]:
            self.check_kdtree_model(BallTree(self.data, metric=metric, **kwargs), 'ball-tree-metric.json')

    def test_kdtree_leaf_size(self):
        self.check_kdtree_model(KDTree(self.data, leaf_size=5), 'kd-tree-leaf.json')
        self.check_kdtree_model(KDTree(self.data, leaf_size=50), 'kd-tree-leaf.json')

    def test_balltree_leaf_size(self):
        self.check_kdtree_model(BallTree(self.data, leaf_size=5), 'ball-tree-leaf.json')
        self.check_kdtree_model(BallTree(self.data, leaf_size=50), 'ball-tree-leaf.json')

    def test_kdtree_dtypes(self):
        for dtype in [np.float32, np.float64]:
            self.check_kdtree_model(KDTree(self.data.astype(dtype)), 'kd-tree-dtype.json')

    def test_balltree_dtypes(self):
        for dtype in [np.float32, np.float64]:
            self.check_kdtree_model(BallTree(self.data.astype(dtype)), 'ball-tree-dtype.json')

    def test_nndescent(self):
        if 'NNDescent' in __optionals__:
            self.check_kdtree_model(NNDescent(self.data, random_state=1234), 'nn-descent.json')

    def check_transform_model(self, model, model_name):
        model.fit(self.data)
        expected_t = model.transform(self.data)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_t = deserialized_model.transform(self.data)

            if hasattr(expected_t, 'toarray'):
                expected_t = expected_t.toarray()
            if hasattr(actual_t, 'toarray'):
                actual_t = actual_t.toarray()

            np.testing.assert_array_almost_equal(expected_t, actual_t)

    def test_kneighbors_transformer(self):
        self.check_transform_model(KNeighborsTransformer(n_neighbors=3), 'kneighbors-transformer.json')

    def test_kneighbors_transformer_modes(self):
        for mode in ['distance', 'connectivity']:
            self.check_transform_model(KNeighborsTransformer(n_neighbors=3, mode=mode),
                                       'kneighbors-transformer-mode.json')

    def test_kneighbors_transformer_algorithms(self):
        for algorithm in ['ball_tree', 'kd_tree', 'brute']:
            self.check_transform_model(KNeighborsTransformer(n_neighbors=3, algorithm=algorithm),
                                       'kneighbors-transformer-algo.json')

    def test_radius_neighbors_transformer(self):
        self.check_transform_model(RadiusNeighborsTransformer(radius=2.0), 'radius-neighbors-transformer.json')

    def test_radius_neighbors_transformer_modes(self):
        for mode in ['distance', 'connectivity']:
            self.check_transform_model(RadiusNeighborsTransformer(radius=2.0, mode=mode),
                                       'radius-neighbors-transformer-mode.json')

    def test_pynndescent_transformer(self):
        if 'PyNNDescentTransformer' in __optionals__:
            self.check_transform_model(PyNNDescentTransformer(n_neighbors=5, random_state=1234),
                                       'pynndescent-transformer.json')

    def test_local_outlier_factor(self):
        model = LocalOutlierFactor(novelty=True)
        model.fit(self.data)
        expected_predictions = model.predict(self.data)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        model_name = 'local-outlier-factor.json'
        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_predictions = deserialized_model.predict(self.data)
            np.testing.assert_array_equal(expected_predictions, actual_predictions)

    def test_neighborhood_components_analysis(self):
        model = NeighborhoodComponentsAnalysis(n_components=2, random_state=1234)
        model.fit(self.data, self.labels)
        expected_t = model.transform(self.data)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        model_name = 'nca.json'
        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_t = deserialized_model.transform(self.data)
            np.testing.assert_array_almost_equal(expected_t, actual_t)

    def check_nca_model(self, model, model_name):
        model.fit(self.data, self.labels)
        expected_t = model.transform(self.data)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_t = deserialized_model.transform(self.data)
            np.testing.assert_array_almost_equal(expected_t, actual_t)

    def test_neighborhood_components_analysis_init(self):
        for init in ['identity', 'random', 'pca', 'lda']:
            self.check_nca_model(NeighborhoodComponentsAnalysis(n_components=2, init=init, random_state=1234),
                                 'nca-init.json')

    def test_neighborhood_components_analysis_no_reduction(self):
        self.check_nca_model(NeighborhoodComponentsAnalysis(n_components=None, random_state=1234), 'nca-full.json')

    def test_neighborhood_components_analysis_dtypes(self):
        for dtype in [np.float32, np.float64]:
            model = NeighborhoodComponentsAnalysis(n_components=2, random_state=1234)
            model.fit(self.data.astype(dtype), self.labels)
            expected_t = model.transform(self.data.astype(dtype))

            serialized_dict_model = ml2json.to_dict(model)
            deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

            model_name = 'nca-dtype.json'
            ml2json.to_json(model, model_name)
            deserialized_json_model = ml2json.from_json(model_name)
            os.remove(model_name)

            for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
                actual_t = deserialized_model.transform(self.data.astype(dtype))
                np.testing.assert_array_almost_equal(expected_t, actual_t)
