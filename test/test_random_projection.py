# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np
from sklearn.datasets import load_iris
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

from src import ml2json


class TestAPI(unittest.TestCase):

    def setUp(self):
        self.X, _ = load_iris(return_X_y=True)

    def check_model(self, model, model_name):
        expected_t = model.fit_transform(self.X)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_t = deserialized_model.transform(self.X)
            np.testing.assert_array_almost_equal(expected_t, actual_t)

    def test_gaussian_random_projection(self):
        self.check_model(GaussianRandomProjection(n_components=2, random_state=1234), 'gaussian-random-projection.json')

    def test_sparse_random_projection(self):
        self.check_model(SparseRandomProjection(n_components=2, random_state=1234), 'sparse-random-projection.json')
