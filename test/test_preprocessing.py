# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer, MinMaxScaler, StandardScaler

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

        self.simple_fit_data = np.array([[0, 1, 1], [1, 0, 0]])
        self.simple_test_data = [0, 1, 2, 1]
        self.simple_test_labels = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]])

        self.X = fetch_california_housing()['data']


    def check_model(self, model, data, labels):
        expected_ft = model.fit_transform(data)
        expected_t = model.transform(data)
        expected_it = model.inverse_transform(labels)

        serialized_dict_model = skljson.to_dict(model)
        deserialized_dict_model = skljson.from_dict(serialized_dict_model)

        skljson.to_json(model, 'model.json')
        deserialized_json_model = skljson.from_json('model.json')
        os.remove('model.json')

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_t = deserialized_model.transform(data)
            actual_ft = deserialized_model.fit_transform(data)
            actual_it = deserialized_model.inverse_transform(labels)

            if model.sparse_output:
                np.testing.assert_array_equal(expected_t.indptr, actual_t.indptr)
                np.testing.assert_array_equal(expected_t.indices, actual_t.indices)
                np.testing.assert_array_equal(expected_t.data, actual_t.data)
                np.testing.assert_array_equal(expected_ft.indptr, actual_ft.indptr)
                np.testing.assert_array_equal(expected_ft.indices, actual_ft.indices)
                np.testing.assert_array_equal(expected_ft.data, actual_ft.data)
                self.assertEqual(expected_it, actual_it)
            else:
                np.testing.assert_array_equal(expected_t, actual_t)
                np.testing.assert_array_equal(expected_ft, actual_ft)
                self.assertEqual(expected_it, actual_it)

    def test_label_binarizer(self):
        self.check_model(LabelBinarizer(), self.sim)
        self.check_model(LabelBinarizer(sparse_output=True))

    def test_multilabel_binarizer(self):
        self.check_model(MultiLabelBinarizer(), self.data)
        self.check_model(MultiLabelBinarizer(sparse_output=True), self.labels)

    def check_scaler(self, scaler):
        expected_ft = scaler.fit_transform(self.X)
        expected_t = scaler.transform(self.X)
        expected_it = scaler.inverse_transform(expected_t)

        serialized_dict_model = skljson.to_dict(scaler)
        deserialized_dict_model = skljson.from_dict(serialized_dict_model)

        skljson.to_json(scaler, 'model.json')
        deserialized_json_model = skljson.from_json('model.json')
        os.remove('model.json')

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_t = deserialized_model.transform(self.X)
            actual_ft = deserialized_model.fit_transform(self.X)
            actual_it = deserialized_model.inverse_transform(actual_t)

            np.testing.assert_array_equal(expected_t, actual_t)
            np.testing.assert_array_equal(expected_ft, actual_ft)
            np.testing.assert_array_equal(expected_it, actual_it)

    def test_minmax_scaler(self):
        self.check_scaler(MinMaxScaler())
        self.check_scaler(MinMaxScaler(feature_range=(10, 20)))
        self.check_scaler(MinMaxScaler(clip=True))

    def test_standard_scaler(self):
        self.check_scaler(StandardScaler())
        # self.check_scaler(StandardScaler(with_mean=False))
        # self.check_scaler(StandardScaler(with_std=False))
        # self.check_scaler(StandardScaler(with_mean=False, with_std=False))
