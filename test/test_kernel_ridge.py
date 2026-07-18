# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np
from sklearn.datasets import make_regression
from sklearn.kernel_ridge import KernelRidge

from src import ml2json


class TestAPI(unittest.TestCase):

    def setUp(self):
        self.X, self.y = make_regression(n_samples=50, n_features=3, random_state=0)

    def test_kernel_ridge(self):
        model = KernelRidge(kernel='rbf')
        model.fit(self.X, self.y)
        expected_predictions = model.predict(self.X)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        model_name = 'kernel-ridge.json'
        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_predictions = deserialized_model.predict(self.X)
            np.testing.assert_array_almost_equal(expected_predictions, actual_predictions)
