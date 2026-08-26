# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np
from sklearn.datasets import load_iris, make_blobs, make_regression

# Allow testing of additional optional dependencies
__optionals__ = []
try:
    from sklearn_extra.robust import (RobustWeightedClassifier, RobustWeightedRegressor,
                                      RobustWeightedKMeans)
    __optionals__.extend(['RobustWeightedClassifier', 'RobustWeightedRegressor', 'RobustWeightedKMeans'])
except:
    pass

from src import ml2json


class TestAPI(unittest.TestCase):

    def setUp(self):
        self.X, self.y = load_iris(return_X_y=True)
        self.Xr, self.yr = make_regression(n_samples=60, n_features=5, random_state=1234)
        self.Xk, _ = make_blobs(n_samples=60, n_features=4, centers=3, random_state=1234)

    def check_predict_model(self, model, model_name, X, y=None):
        if y is None:
            model.fit(X)
        else:
            model.fit(X, y)
        expected_p = model.predict(X)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_p = deserialized_model.predict(X)
            np.testing.assert_array_almost_equal(expected_p, actual_p)

    def test_robust_weighted_regressor(self):
        if 'RobustWeightedRegressor' in __optionals__:
            self.check_predict_model(RobustWeightedRegressor(random_state=1234), 'robust-weighted-regressor.json',
                                     self.Xr, self.yr)

    def test_robust_weighted_classifier(self):
        if 'RobustWeightedClassifier' in __optionals__:
            self.check_predict_model(RobustWeightedClassifier(random_state=1234), 'robust-weighted-classifier.json',
                                     self.X, self.y)

    def test_robust_weighted_kmeans(self):
        if 'RobustWeightedKMeans' in __optionals__:
            self.check_predict_model(RobustWeightedKMeans(n_clusters=3, random_state=1234),
                                     'robust-weighted-kmeans.json', self.Xk)


if __name__ == '__main__':
    unittest.main()
