# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src import ml2json


class TestAPI(unittest.TestCase):

    def setUp(self):
        self.X_cls, self.y_cls = make_classification(n_samples=100, n_features=5, n_classes=2, n_informative=3,
                                                      n_redundant=0, random_state=0)
        self.X_reg, self.y_reg = make_regression(n_samples=100, n_features=5, random_state=0)

    def check_model(self, model, model_name, X, method='predict'):
        expected = getattr(model, method)(X)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual = getattr(deserialized_model, method)(X)
            np.testing.assert_array_almost_equal(expected, actual)

    def test_column_transformer(self):
        model = ColumnTransformer([('scale', StandardScaler(), [0, 1, 2])], remainder='passthrough')
        model.fit(self.X_cls, self.y_cls)
        self.check_model(model, 'column-transformer.json', self.X_cls, method='transform')

    def test_column_transformer_multiple_transformers(self):
        model = ColumnTransformer([('scale', StandardScaler(), [0, 1]),
                                   ('minmax', MinMaxScaler(), [2, 3])],
                                  remainder='drop')
        model.fit(self.X_cls, self.y_cls)
        self.check_model(model, 'column-transformer-multi.json', self.X_cls, method='transform')

    def test_transformed_target_regressor(self):
        model = TransformedTargetRegressor(regressor=LinearRegression(), transformer=StandardScaler())
        model.fit(self.X_reg, self.y_reg)
        self.check_model(model, 'transformed-target-regressor.json', self.X_reg, method='predict')
