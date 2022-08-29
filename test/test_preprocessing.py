from sklearn.preprocessing import MultiLabelBinarizer
# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np
from sklearn.datasets import fetch_california_housing

from src import sklearn_json as skljson


class TestAPI(unittest.TestCase):

    def setUp(self):
        self.data = [
            {'action', 'drama', 'fantasy'},
            {'comedy', 'horror'},
            {'comedy', 'romance'},
            {'horror'},
            {'mystery', 'thriller'},
            {'sci-fi', 'thriller'},
        ]
        self.labels = np.array([
            [1, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 1],
        ])

    def check_model(self, model):
        expected_ft = model.fit_transform(self.data)
        expected_t = model.transform(self.data)
        expected_it = model.inverse_transform(self.labels)

        serialized_dict_model = skljson.to_dict(model)
        deserialized_dict_model = skljson.from_dict(serialized_dict_model)

        skljson.to_json(model, 'model.json')
        deserialized_json_model = skljson.from_json('model.json')

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_t = deserialized_model.transform(self.data)
            actual_ft = deserialized_model.fit_transform(self.data)
            actual_it = deserialized_model.inverse_transform(self.labels)

            if model.sparse_output:
                np.testing.assert_array_equal(expected_t.indptr, actual_t.indptr)
                np.testing.assert_array_equal(expected_t.indices, actual_t.indices)
                np.testing.assert_array_equal(expected_t.data, actual_t.data)
                np.testing.assert_array_equal(expected_ft.indptr, actual_ft.indptr)
                np.testing.assert_array_equal(expected_ft.indices, actual_ft.indices)
                np.testing.assert_array_equal(expected_ft.data, actual_ft.data)
                np.testing.assert_array_equal(expected_it, actual_it)
            else:
                np.testing.assert_array_equal(expected_t, actual_t)
                np.testing.assert_array_equal(expected_ft, actual_ft)
                np.testing.assert_array_equal(expected_it, actual_it)

    def test_multilabel_binarizer(self):
        self.check_model(MultiLabelBinarizer())
        self.check_model(MultiLabelBinarizer(sparse_output=True))
