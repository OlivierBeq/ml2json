# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import (LabelEncoder, LabelBinarizer, MultiLabelBinarizer,
                                   MinMaxScaler, StandardScaler, KernelCenterer,
                                   OneHotEncoder, RobustScaler, MaxAbsScaler,
                                   OrdinalEncoder)
from sklearn.metrics.pairwise import pairwise_kernels

from src import ml2json


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
        self.kernel_X = pairwise_kernels(self.X[:100], metric="linear", filter_params=True, degree=3, coef0=1)

    def check_model(self, model, model_name, data, labels):
        expected_ft = model.fit_transform(data)
        expected_t = model.transform(data)
        expected_it = model.inverse_transform(labels)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_t = deserialized_model.transform(data)
            actual_ft = deserialized_model.fit_transform(data)
            actual_it = deserialized_model.inverse_transform(labels)

            if hasattr(model, 'sparse_output') and model.sparse_output:
                np.testing.assert_array_equal(expected_t.indptr, actual_t.indptr)
                np.testing.assert_array_equal(expected_t.indices, actual_t.indices)
                np.testing.assert_array_equal(expected_t.data, actual_t.data)
                np.testing.assert_array_equal(expected_ft.indptr, actual_ft.indptr)
                np.testing.assert_array_equal(expected_ft.indices, actual_ft.indices)
                np.testing.assert_array_equal(expected_ft.data, actual_ft.data)
                if isinstance(actual_it, np.ndarray):
                    np.testing.assert_array_equal(expected_it, actual_it)
                else:
                    self.assertEqual(expected_it, actual_it)
            else:
                np.testing.assert_array_equal(expected_t, actual_t)
                np.testing.assert_array_equal(expected_ft, actual_ft)
                if isinstance(actual_it, np.ndarray):
                    np.testing.assert_array_equal(expected_it, actual_it)
                else:
                    self.assertEqual(expected_it, actual_it)

    def test_label_encoder(self):
        self.check_model(LabelEncoder(), 'label-encoder.json', ["paris", "paris", "tokyo", "amsterdam"], [0, 0, 1, 2])

    def test_label_binarizer(self):
        self.check_model(LabelBinarizer(), 'label-binarizer.json', self.simple_test_data, self.simple_test_labels)
        self.check_model(LabelBinarizer(sparse_output=True), 'label-binarizer.json', self.simple_test_data, self.simple_test_labels)

    def test_multilabel_binarizer(self):
        self.check_model(MultiLabelBinarizer(), 'multilabel-binarizer.json', self.data, self.labels)
        self.check_model(MultiLabelBinarizer(sparse_output=True), 'multilabel-binarizer.json', self.data, self.labels)

    def check_scaler(self, scaler, model_name):
        expected_ft = scaler.fit_transform(self.X)
        expected_t = scaler.transform(self.X)
        expected_it = scaler.inverse_transform(expected_t)

        serialized_dict_model = ml2json.to_dict(scaler)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        ml2json.to_json(scaler, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_t = deserialized_model.transform(self.X)
            actual_ft = deserialized_model.fit_transform(self.X)
            actual_it = deserialized_model.inverse_transform(actual_t)

            np.testing.assert_array_equal(expected_t, actual_t)
            np.testing.assert_array_equal(expected_ft, actual_ft)
            np.testing.assert_array_equal(expected_it, actual_it)

    def test_minmax_scaler(self):
        self.check_scaler(MinMaxScaler(), 'minmax-scaler.json')
        self.check_scaler(MinMaxScaler(feature_range=(10, 20)), 'minmax-scaler.json')
        self.check_scaler(MinMaxScaler(clip=True), 'minmax-scaler.json')

    def test_standard_scaler(self):
        self.check_scaler(StandardScaler(), 'standard-scaler.json')
        self.check_scaler(StandardScaler(with_mean=False), 'standard-scaler.json')
        self.check_scaler(StandardScaler(with_std=False), 'standard-scaler.json')
        self.check_scaler(StandardScaler(with_mean=False, with_std=False), 'standard-scaler.json')

    def test_robust_scaler(self):
        self.check_scaler(RobustScaler(), 'robust-scaler.json')
        self.check_scaler(RobustScaler(with_centering=False), 'robust-scaler.json')
        self.check_scaler(RobustScaler(with_scaling=False), 'robust-scaler.json')
        self.check_scaler(RobustScaler(with_centering=False, with_scaling=False), 'robust-scaler.json')
        
    def test_maxabs_scaler(self):
        self.check_scaler(MaxAbsScaler(), 'maxabs-scaler.json')

    def check_centerer(self, centerer, model_name):
        expected_ft = centerer.fit_transform(self.kernel_X)
        expected_t = centerer.transform(self.kernel_X)

        serialized_dict_model = ml2json.to_dict(centerer)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        ml2json.to_json(centerer, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_t = deserialized_model.transform(self.kernel_X)
            actual_ft = deserialized_model.fit_transform(self.kernel_X)

            np.testing.assert_array_equal(expected_t, actual_t)
            np.testing.assert_array_equal(expected_ft, actual_ft)

    def test_kernel_centerer(self):
        self.check_centerer(KernelCenterer(), 'kernel-centerer.json')

    def test_onehot_encoder(self):
        model  = OneHotEncoder(handle_unknown='ignore')
        model.fit([['Male', 1], ['Female', 3], ['Female', 2]])
        expected_t = model.transform([['Female', 1], ['Male', 4]]).toarray()
        expected_it = model.inverse_transform(expected_t)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        model_name = 'onehot-encoder.json'
        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_t = deserialized_model.transform([['Female', 1], ['Male', 4]]).toarray()
            actual_it = deserialized_model.inverse_transform(actual_t)

            np.testing.assert_array_equal(expected_t, actual_t)
            np.testing.assert_array_equal(expected_it, actual_it)

    def test_ordinal_encoder(self):
        X_train = np.array([["a"] * 5 + ["b"] * 20 + ["c"] * 10 + ["d"] * 3 + [np.nan]],dtype=object).T
        model = OrdinalEncoder()
        model.fit(X_train)
        X_test = np.array([["a"], ["b"], ["c"], ["d"]], dtype=object)

        expected_t = model.transform(X_test)
        expected_it = model.inverse_transform(expected_t)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        model_name = 'onehot-encoder.json'
        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_t = deserialized_model.transform(X_test)
            actual_it = deserialized_model.inverse_transform(actual_t)

            np.testing.assert_array_equal(expected_t, actual_t)
            np.testing.assert_array_equal(expected_it, actual_it)

        model = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=3,
                               max_categories=3, encoded_missing_value=4)
        model.fit(X_train)
        X_test = np.array([["a"], ["b"], ["c"], ["d"], ["e"]], dtype=object)

        expected_t = model.transform(X_test)
        expected_it = model.inverse_transform(expected_t)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        model_name = 'onehot-encoder.json'
        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_t = deserialized_model.transform(X_test)
            np.testing.assert_array_equal(expected_t, actual_t)
