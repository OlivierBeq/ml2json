# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.dummy import DummyClassifier, DummyRegressor

from src import ml2json


class TestAPI(unittest.TestCase):

    def setUp(self):
        self.X_clf, self.y_clf = make_classification(n_samples=50, n_features=3, n_classes=3, n_informative=3,
                                                      n_redundant=0, random_state=0)
        self.X_reg, self.y_reg = make_regression(n_samples=50, n_features=3, random_state=0)

    def check_model(self, model, model_name, X, y):
        model.fit(X, y)
        expected_predictions = model.predict(X)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_predictions = deserialized_model.predict(X)
            np.testing.assert_array_equal(expected_predictions, actual_predictions)

    def test_dummy_classifier(self):
        self.check_model(DummyClassifier(strategy='stratified', random_state=1234), 'dummy-classifier.json', self.X_clf, self.y_clf)
        self.check_model(DummyClassifier(strategy='most_frequent'), 'dummy-classifier.json', self.X_clf, self.y_clf)

    def test_dummy_regressor(self):
        self.check_model(DummyRegressor(strategy='mean'), 'dummy-regressor.json', self.X_reg, self.y_reg)
        self.check_model(DummyRegressor(strategy='median'), 'dummy-regressor.json', self.X_reg, self.y_reg)
