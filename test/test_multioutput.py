# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier, MultiOutputRegressor, RegressorChain

from src import ml2json


class TestAPI(unittest.TestCase):

    def setUp(self):
        X_cls, y_cls = make_classification(n_samples=100, n_features=5, n_classes=2, n_informative=3,
                                           n_redundant=0, random_state=0)
        self.X_cls = X_cls
        self.y_cls_multi = np.vstack([y_cls, y_cls[::-1]]).T

        X_reg, y_reg = make_regression(n_samples=100, n_features=5, random_state=0)
        self.X_reg = X_reg
        self.y_reg_multi = np.vstack([y_reg, y_reg[::-1]]).T

    def check_model(self, model, model_name, X):
        expected_predictions = model.predict(X)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_predictions = deserialized_model.predict(X)
            np.testing.assert_array_almost_equal(expected_predictions, actual_predictions)

    def test_classifier_chain(self):
        model = ClassifierChain(RandomForestClassifier(n_estimators=5, random_state=42))
        model.fit(self.X_cls, self.y_cls_multi)
        self.check_model(model, 'classifier-chain.json', self.X_cls)

    def test_multioutput_classifier(self):
        model = MultiOutputClassifier(RandomForestClassifier(n_estimators=5, random_state=42))
        model.fit(self.X_cls, self.y_cls_multi)
        self.check_model(model, 'multioutput-classifier.json', self.X_cls)

    def test_multioutput_regressor(self):
        model = MultiOutputRegressor(RandomForestRegressor(n_estimators=5, random_state=42))
        model.fit(self.X_reg, self.y_reg_multi)
        self.check_model(model, 'multioutput-regressor.json', self.X_reg)

    def test_regressor_chain(self):
        model = RegressorChain(RandomForestRegressor(n_estimators=5, random_state=42))
        model.fit(self.X_reg, self.y_reg_multi)
        self.check_model(model, 'regressor-chain.json', self.X_reg)

    def test_classifier_chain_order(self):
        model = ClassifierChain(RandomForestClassifier(n_estimators=5, random_state=42), order=[1, 0])
        model.fit(self.X_cls, self.y_cls_multi)
        self.check_model(model, 'classifier-chain-order.json', self.X_cls)

    def test_regressor_chain_order(self):
        model = RegressorChain(RandomForestRegressor(n_estimators=5, random_state=42), order=[1, 0])
        model.fit(self.X_reg, self.y_reg_multi)
        self.check_model(model, 'regressor-chain-order.json', self.X_reg)

    def test_classifier_chain_linear_estimator(self):
        model = ClassifierChain(LogisticRegression(), cv=3, random_state=42)
        model.fit(self.X_cls, self.y_cls_multi)
        self.check_model(model, 'classifier-chain-linear.json', self.X_cls)

    def test_regressor_chain_linear_estimator(self):
        model = RegressorChain(Ridge(), cv=3, random_state=42)
        model.fit(self.X_reg, self.y_reg_multi)
        self.check_model(model, 'regressor-chain-linear.json', self.X_reg)

    def test_multioutput_classifier_linear_estimator(self):
        model = MultiOutputClassifier(LogisticRegression())
        model.fit(self.X_cls, self.y_cls_multi)
        self.check_model(model, 'multioutput-classifier-linear.json', self.X_cls)

    def test_multioutput_regressor_linear_estimator(self):
        model = MultiOutputRegressor(Ridge())
        model.fit(self.X_reg, self.y_reg_multi)
        self.check_model(model, 'multioutput-regressor-linear.json', self.X_reg)
