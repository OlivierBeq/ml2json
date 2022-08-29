# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

from src import sklearn_json as skljson


class TestAPI(unittest.TestCase):

    def setUp(self):
        self.data, self.labels = load_iris(return_X_y=True)

    def check_model(self, model):
        expected_ft = model.fit_transform(self.data)
        expected_t = model.transform(self.data)
        expected_it = model.inverse_transform(expected_t)

        serialized_dict_model = skljson.to_dict(model)
        deserialized_dict_model = skljson.from_dict(serialized_dict_model)

        skljson.to_json(model, 'model.json')
        deserialized_json_model = skljson.from_json('model.json')
        os.remove('model.json')

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_t = deserialized_model.transform(self.data)
            actual_ft = deserialized_model.fit_transform(self.data)
            actual_it = deserialized_model.inverse_transform(actual_ft)

            np.testing.assert_array_almost_equal(expected_t, actual_t)
            np.testing.assert_array_almost_equal(expected_ft, actual_ft)
            np.testing.assert_array_almost_equal(expected_it, actual_it)

    def test_pca(self):
        self.check_model(PCA(n_components=2, svd_solver='full'))
        self.check_model(PCA(svd_solver='arpack'))
