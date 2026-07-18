# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np
from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

from src import ml2json


class TestAPI(unittest.TestCase):

    def setUp(self):
        self.X, self.y = load_iris(return_X_y=True)

    def check_model(self, model, model_name):
        model.fit(self.X)
        expected_predictions = model.predict(self.X)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_predictions = deserialized_model.predict(self.X)
            np.testing.assert_array_equal(expected_predictions, actual_predictions)

            expected_score = model.score_samples(self.X)
            actual_score = deserialized_model.score_samples(self.X)
            np.testing.assert_array_almost_equal(expected_score, actual_score)

    def test_gaussian_mixture(self):
        self.check_model(GaussianMixture(n_components=3, random_state=1234), 'gaussian-mixture.json')

    def test_bayesian_gaussian_mixture(self):
        self.check_model(BayesianGaussianMixture(n_components=3, random_state=1234), 'bayesian-gaussian-mixture.json')
