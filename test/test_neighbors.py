# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import NearestNeighbors, KDTree
from pynndescent import NNDescent

from src import sklearn_json as skljson


class TestAPI(unittest.TestCase):

    def setUp(self):
        self.data, self.labels = load_iris(return_X_y=True)

    def check_nearest_neighbors_model(self, model, model_name):
        model.fit(self.data)

        rng = np.random.RandomState(1234)
        subset = self.data[rng.randint(self.data.shape[0], size=10)]
        expected_ft = model.kneighbors(subset)

        serialized_dict_model = skljson.to_dict(model)
        deserialized_dict_model = skljson.from_dict(serialized_dict_model)

        skljson.to_json(model, model_name)
        deserialized_json_model = skljson.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_ft = deserialized_model.kneighbors(subset)

            np.testing.assert_array_almost_equal(expected_ft, actual_ft)

    def test_nearest_neighbors(self):
        self.check_nearest_neighbors_model(NearestNeighbors(), 'nearest-neighbors.json')

    def check_kdtree_model(self, model, model_name):
        rng = np.random.RandomState(1234)
        subset = self.data[rng.randint(self.data.shape[0], size=10)]
        expected_ft_d, expected_ft_i = model.query(subset)

        serialized_dict_model = skljson.to_dict(model)
        deserialized_dict_model = skljson.from_dict(serialized_dict_model)


        skljson.to_json(model, model_name)
        deserialized_json_model = skljson.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_ft_d, actual_ft_i = deserialized_model.query(subset)

            np.testing.assert_array_almost_equal(expected_ft_d, actual_ft_d)
            np.testing.assert_array_almost_equal(expected_ft_i, actual_ft_i)

    def test_kdtree(self):
        self.check_kdtree_model(KDTree(self.data), 'kd-tree.json')

    def test_nndescent(self):
        self.check_kdtree_model(NNDescent(self.data), 'nn-descent.json')
