# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np
from sklearn.datasets import make_classification
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, MissingIndicator, KNNImputer, IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor

from src import ml2json


class TestAPI(unittest.TestCase):

    def setUp(self):
        self.X, _ = make_classification(n_samples=100, n_features=10, random_state=0)
        rng = np.random.RandomState(0)
        mask = rng.rand(*self.X.shape) < 0.2
        self.X_missing = self.X.copy()
        self.X_missing[mask] = np.nan

    def check_model(self, model, model_name, X=None):
        X = self.X_missing if X is None else X
        model.fit(X)
        expected_transform = model.transform(X)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_transform = deserialized_model.transform(X)
            np.testing.assert_array_almost_equal(np.asarray(expected_transform, dtype=float),
                                                  np.asarray(actual_transform, dtype=float))

    def test_simple_imputer(self):
        self.check_model(SimpleImputer(), 'simple-imputer.json')
        self.check_model(SimpleImputer(strategy='median'), 'simple-imputer.json')
        self.check_model(SimpleImputer(strategy='most_frequent'), 'simple-imputer.json')
        self.check_model(SimpleImputer(strategy='constant', fill_value=-1), 'simple-imputer.json')
        self.check_model(SimpleImputer(add_indicator=True), 'simple-imputer.json')

    def test_missing_indicator(self):
        self.check_model(MissingIndicator(), 'missing-indicator.json')
        self.check_model(MissingIndicator(features='all'), 'missing-indicator.json')

    def test_knn_imputer(self):
        self.check_model(KNNImputer(), 'knn-imputer.json')
        self.check_model(KNNImputer(n_neighbors=3, weights='distance'), 'knn-imputer.json')
        self.check_model(KNNImputer(add_indicator=True), 'knn-imputer.json')

    def test_iterative_imputer(self):
        self.check_model(IterativeImputer(max_iter=5, random_state=0), 'iterative-imputer.json')
        self.check_model(IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=5, random_state=0),
                                          max_iter=5, random_state=0), 'iterative-imputer.json')

    def test_simple_imputer_dtype(self):
        self.check_model(SimpleImputer(), 'simple-imputer-f32.json', X=self.X_missing.astype(np.float32))


if __name__ == '__main__':
    unittest.main()
