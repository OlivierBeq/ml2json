# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import AffinityPropagation, KMeans

from src import sklearn_json as skljson


class TestAPI(unittest.TestCase):

    def setUp(self):
        centers = [[1, 1], [-1, -1], [1, -1]]
        self.n_centers = len(centers)
        self.X, self.labels = make_blobs(n_samples=3000,
                                         centers=centers,
                                         cluster_std=0.7,
                                         random_state=1234)
        self.simple_X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])

    def check_transform_model(self, model, data):
        expected_ft = model.fit_transform(data)
        expected_t = model.transform(data)

        serialized_dict_model = skljson.to_dict(model)
        deserialized_dict_model = skljson.from_dict(serialized_dict_model)

        skljson.to_json(model, 'model.json')
        deserialized_json_model = skljson.from_json('model.json')
        os.remove('model.json')

        for deserialized_model in [deserialized_dict_model]:#, deserialized_json_model]:

            for key in sorted(model.__dict__.keys()):
                if isinstance(model.__dict__[key], np.ndarray):
                    print(key, (model.__dict__[key] == deserialized_dict_model.__dict__[key]).all())
                else:
                    print(key, model.__dict__[key] == deserialized_dict_model.__dict__[key])

            actual_ft = deserialized_model.fit_transform(data)
            actual_t = deserialized_model.transform(data)

            np.testing.assert_array_almost_equal(expected_ft, actual_ft)
            np.testing.assert_array_almost_equal(expected_t, actual_t)

    def check_predict_model(self, model, data):
        expected_fp = model.fit_predict(data)
        expected_p = model.predict(data)

        serialized_dict_model = skljson.to_dict(model)
        deserialized_dict_model = skljson.from_dict(serialized_dict_model)

        skljson.to_json(model, 'model.json')
        skljson.from_json('model.json')
        os.remove('model.json')

        for deserialized_model in [deserialized_dict_model]:#, deserialized_json_model]:

            for key in sorted(model.__dict__.keys()):
                if isinstance(model.__dict__[key], np.ndarray):
                    print(key, (model.__dict__[key] == deserialized_dict_model.__dict__[key]).all())
                else:
                    print(key, model.__dict__[key] == deserialized_dict_model.__dict__[key])

            actual_fp = deserialized_model.fit_predict(data)
            actual_p = deserialized_model.predict(data)

            np.testing.assert_array_equal(expected_fp, actual_fp)
            np.testing.assert_array_equal(expected_p, actual_p)

    def test_kmeans(self):
        self.check_transform_model(
            KMeans(n_clusters=self.n_centers, init='k-means++', random_state=1234, n_init=100, max_iter=10000,
                   verbose=0, tol=1e-999), self.X)
        self.check_predict_model(
            KMeans(n_clusters=self.n_centers, init='k-means++', random_state=1234, n_init=100, max_iter=10000,
                   verbose=0, tol=1e-999), self.X)
        self.check_transform_model(
            KMeans(n_clusters=self.n_centers, init='random', random_state=1234, n_init=100, max_iter=10000, verbose=0,
                   tol=1e-999), self.X)
        self.check_predict_model(
            KMeans(n_clusters=self.n_centers, init='random', random_state=1234, n_init=100, max_iter=10000, verbose=0,
                   tol=1e-999), self.X)

    def test_affinity_propagation(self):
        self.check_predict_model(AffinityPropagation(), self.simple_X)