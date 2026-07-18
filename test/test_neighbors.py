# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np
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

    def test_radius_neighbors_transformer(self):
        self.check_transform_model(RadiusNeighborsTransformer(radius=2.0), 'radius-neighbors-transformer.json')

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
