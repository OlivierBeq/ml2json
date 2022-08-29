# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

from src import sklearn_json as skljson


class TestAPI(unittest.TestCase):

    def setUp(self):
        centers = [[1, 1], [-1, -1], [1, -1]]
        self.n_centers = len(centers)
        self.X, self.labels = make_blobs(n_samples=3000,
                                         centers=centers,
                                         cluster_std=0.7,
                                         random_state=1234)

    def check_model(self, model):
        expected_ft = model.fit_transform(self.X)
        expected_t = model.transform(self.X)
        expected_fp = model.fit_predict(self.X)
        expected_p = model.predict(self.X)

        serialized_dict_model = skljson.to_dict(model)
        deserialized_dict_model = skljson.from_dict(serialized_dict_model)

        skljson.to_json(model, 'model.json')
        deserialized_json_model = skljson.from_json('model.json')
        os.remove('model.json')

        for deserialized_model in [deserialized_dict_model]:#, deserialized_json_model]:

            # for key in model.__dict__.keys():
            #     print(key, model.__dict__[key], deserialized_dict_model.__dict__[key])

            print(model.get_params())
            print(deserialized_dict_model.get_params())
            print([model.__dict__[key] for key in sorted(model.__dict__.keys())])
            print([deserialized_dict_model.__dict__[key] for key in sorted(deserialized_dict_model.__dict__.keys())])

            actual_ft = deserialized_model.fit_transform(self.X)
            actual_t = deserialized_model.transform(self.X)
            actual_fp = deserialized_model.fit_predict(self.X)
            actual_p = deserialized_model.predict(self.X)

            np.testing.assert_array_almost_equal(expected_ft, actual_ft)
            np.testing.assert_array_almost_equal(expected_t, actual_t)
            np.testing.assert_array_equal(expected_fp, actual_fp)
            np.testing.assert_array_equal(expected_p, actual_p)

    def test_kmeans(self):
        self.check_model(KMeans(n_clusters=self.n_centers, init='k-means++', random_state=1234, n_init=100, max_iter=10000, verbose=0, tol=1e-999))
        self.check_model(KMeans(n_clusters=self.n_centers, init='random', random_state=1234, n_init=100, max_iter=10000, verbose=0, tol=1e-999))
