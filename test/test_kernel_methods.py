# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np
from sklearn.datasets import load_iris, make_regression

# Allow testing of additional optional dependencies
__optionals__ = []
try:
    from sklearn_extra.kernel_methods import EigenProRegressor, EigenProClassifier
    __optionals__.extend(['EigenProRegressor', 'EigenProClassifier'])
except:
    pass

from src import ml2json


class TestAPI(unittest.TestCase):

    def setUp(self):
        self.X, self.y = load_iris(return_X_y=True)
        self.Xr, self.yr = make_regression(n_samples=60, n_features=5, random_state=1234)

    def check_predict_model(self, model, model_name, X, y):
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

    def test_eigenpro_regressor(self):
        if 'EigenProRegressor' in __optionals__:
            self.check_predict_model(EigenProRegressor(random_state=1234), 'eigenpro-regressor.json', self.Xr, self.yr)

    def test_eigenpro_classifier(self):
        if 'EigenProClassifier' in __optionals__:
            self.check_predict_model(EigenProClassifier(random_state=1234), 'eigenpro-classifier.json', self.X, self.y)


if __name__ == '__main__':
    unittest.main()
