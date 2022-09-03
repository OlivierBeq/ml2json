# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np
from sklearn.datasets import load_iris
from sklearn.manifold import (Isomap, LocallyLinearEmbedding,
                              MDS, SpectralEmbedding, TSNE)

from src import sklearn_json as skljson


class TestAPI(unittest.TestCase):

    def setUp(self):
        self.data, self.labels = load_iris(return_X_y=True)

    def check_model(self, model, model_name):
        expected_ft = model.fit_transform(self.data)

        serialized_dict_model = skljson.to_dict(model)
        deserialized_dict_model = skljson.from_dict(serialized_dict_model)

        skljson.to_json(model, model_name)
        deserialized_json_model = skljson.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_ft = deserialized_model.fit_transform(self.data)

            np.testing.assert_array_almost_equal(expected_ft, actual_ft)

    def test_tsne(self):
        self.check_model(TSNE(init='pca', learning_rate='auto'), 'tsne.json')
        with self.assertRaises(AssertionError):
            self.check_model(TSNE(init='random', learning_rate='auto'), 'tsne.json')

    def test_mds(self):
        self.check_model(MDS(random_state=1234), 'mds.json')

    def test_isomap(self):
        self.check_model(Isomap(n_neighbors=50, n_components=10, neighbors_algorithm='brute'), 'isomap.json')
