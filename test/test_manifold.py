# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np
from sklearn.datasets import load_iris, load_digits, fetch_california_housing
from sklearn.manifold import (Isomap, LocallyLinearEmbedding,
                              MDS, SpectralEmbedding, TSNE)

from src import sklearn_json as skljson


class TestAPI(unittest.TestCase):

    def setUp(self):
        self.iris_data, _ = load_iris(return_X_y=True)
        self.digit_data, _ = load_digits(return_X_y=True)
        self.calhouse_data, _ = fetch_california_housing(return_X_y=True)

    def check_model(self, model, model_name, data):
        expected_ft = model.fit_transform(data)

        serialized_dict_model = skljson.to_dict(model)
        deserialized_dict_model = skljson.from_dict(serialized_dict_model)

        skljson.to_json(model, model_name)
        deserialized_json_model = skljson.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_ft = deserialized_model.fit_transform(data)

            np.testing.assert_array_almost_equal(expected_ft, actual_ft)

    def test_tsne(self):
        self.check_model(TSNE(init='pca', learning_rate='auto'), 'tsne.json', self.iris_data)
        with self.assertRaises(AssertionError):
            self.check_model(TSNE(init='random', learning_rate='auto'), 'tsne.json', self.iris_data)

    def test_mds(self):
        self.check_model(MDS(random_state=1234), 'mds.json', self.iris_data)

    def test_isomap(self):
        self.check_model(Isomap(n_neighbors=50, n_components=10, neighbors_algorithm='kd_tree'), 'isomap.json', self.iris_data)
        self.check_model(Isomap(n_neighbors=50, n_components=10, neighbors_algorithm='brute'), 'isomap.json', self.iris_data)
        self.check_model(Isomap(n_neighbors=50, n_components=10, neighbors_algorithm='ball_tree'), 'isomap.json', self.iris_data)

    def test_locally_linear_embedding(self):
        self.check_model(LocallyLinearEmbedding(neighbors_algorithm='kd_tree'), 'locally-linear-embedding.json', self.iris_data)
        self.check_model(LocallyLinearEmbedding(neighbors_algorithm='brute'), 'locally-linear-embedding.json', self.iris_data)
        self.check_model(LocallyLinearEmbedding(neighbors_algorithm='ball_tree'), 'locally-linear-embedding.json', self.iris_data)

    def test_spectral_embedding(self):
        self.check_model(SpectralEmbedding(affinity='nearest_neighbors', random_state=1234), 'spectral-embedding.json', self.digit_data)
        self.check_model(SpectralEmbedding(affinity='rbf', random_state=1234), 'spectral-embedding.json', self.iris_data)
