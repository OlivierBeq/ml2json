# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern

from src import ml2json


class TestAPI(unittest.TestCase):

    def setUp(self):
        self.X_clf, self.y_clf = make_classification(n_samples=40, n_features=3, n_classes=2, n_informative=3,
                                                      n_redundant=0, random_state=0)
        self.X_reg, self.y_reg = make_regression(n_samples=40, n_features=3, random_state=0)

    def check_model(self, model, model_name, X, method='predict'):
        expected_predictions = getattr(model, method)(X)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_predictions = getattr(deserialized_model, method)(X)
            np.testing.assert_array_almost_equal(expected_predictions, actual_predictions)

    def test_gaussian_process_classifier(self):
        model = GaussianProcessClassifier(kernel=1.0 * RBF(length_scale=1.0), random_state=1234)
        model.fit(self.X_clf, self.y_clf)
        self.check_model(model, 'gaussian-process-classifier.json', self.X_clf, method='predict_proba')

    def test_gaussian_process_regressor(self):
        model = GaussianProcessRegressor(kernel=RBF() + WhiteKernel(), random_state=1234)
        model.fit(self.X_reg, self.y_reg)
        self.check_model(model, 'gaussian-process-regressor.json', self.X_reg, method='predict')

    def test_gaussian_process_regressor_composite_kernel(self):
        model = GaussianProcessRegressor(kernel=(1.0 * RBF()) * Matern() + WhiteKernel(noise_level=0.1),
                                         random_state=1234)
        model.fit(self.X_reg, self.y_reg)
        self.check_model(model, 'gaussian-process-regressor-composite.json', self.X_reg, method='predict')
