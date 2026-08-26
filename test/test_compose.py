# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np
import scipy.sparse as sp
from sklearn.datasets import make_classification, make_regression
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, QuantileTransformer

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
            if sp.issparse(expected) or sp.issparse(actual):
                np.testing.assert_array_almost_equal(np.asarray(expected.todense()), np.asarray(actual.todense()))
            else:
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

    def test_column_transformer_remainder_estimator(self):
        model = ColumnTransformer([('scale', StandardScaler(), [0, 1])],
                                  remainder=PCA(n_components=2, random_state=1234))
        model.fit(self.X_cls, self.y_cls)
        self.check_model(model, 'column-transformer-remainder-estimator.json', self.X_cls, method='transform')

    def test_column_transformer_transformer_weights(self):
        model = ColumnTransformer([('scale', StandardScaler(), [0, 1]),
                                   ('minmax', MinMaxScaler(), [2, 3])],
                                  transformer_weights={'scale': 0.5, 'minmax': 2.0})
        model.fit(self.X_cls, self.y_cls)
        self.check_model(model, 'column-transformer-weights.json', self.X_cls, method='transform')

    def test_column_transformer_sparse_threshold(self):
        X_mixed = np.c_[self.X_cls, np.random.RandomState(0).randint(0, 3, size=(self.X_cls.shape[0], 1))]
        model = ColumnTransformer([('onehot', OneHotEncoder(), [5]),
                                   ('scale', StandardScaler(), [0, 1])],
                                  sparse_threshold=1.0)
        model.fit(X_mixed, self.y_cls)
        self.check_model(model, 'column-transformer-sparse.json', X_mixed, method='transform')

    def test_column_transformer_mixed_categorical_numeric(self):
        X_mixed = np.c_[self.X_cls, np.random.RandomState(0).randint(0, 3, size=(self.X_cls.shape[0], 1))]
        model = ColumnTransformer([('onehot', OneHotEncoder(sparse_output=False), [5]),
                                   ('scale', StandardScaler(), [0, 1, 2])],
                                  remainder='drop')
        model.fit(X_mixed, self.y_cls)
        self.check_model(model, 'column-transformer-mixed.json', X_mixed, method='transform')

    def test_transformed_target_regressor_func(self):
        y_reg_pos = np.abs(self.y_reg) + 1
        model = TransformedTargetRegressor(regressor=LinearRegression(), func=np.log1p, inverse_func=np.expm1)
        model.fit(self.X_reg, y_reg_pos)
        self.check_model(model, 'transformed-target-regressor-func.json', self.X_reg, method='predict')

    def test_transformed_target_regressor_quantile_transformer(self):
        model = TransformedTargetRegressor(regressor=LinearRegression(),
                                           transformer=QuantileTransformer(n_quantiles=50, random_state=1234))
        model.fit(self.X_reg, self.y_reg)
        self.check_model(model, 'transformed-target-regressor-qt.json', self.X_reg, method='predict')
