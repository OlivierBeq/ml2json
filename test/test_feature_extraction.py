# -*- coding: utf-8 -*-

import os
import re
import unittest
from collections import Counter

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import DictVectorizer, FeatureHasher

from src import ml2json


class TestAPI(unittest.TestCase):

    def setUp(self):
        newsgroup = fetch_20newsgroups(subset='train', categories=['sci.space'], remove=('headers', 'footers', 'quotes'))

        self.X = [
            Counter(tok.lower() for tok in re.findall(r"\w+", text))
            for text in newsgroup.data
        ]

    def check_model(self, model, model_name):
        expected_vectors = model.fit_transform(self.X)

        serialized_model = ml2json.to_dict(model)
        deserialized_model = ml2json.from_dict(serialized_model)

        actual_vectors = deserialized_model.fit_transform(self.X)

        if model.sparse:
            np.testing.assert_array_equal(expected_vectors.indptr, actual_vectors.indptr)
            np.testing.assert_array_equal(expected_vectors.indices, actual_vectors.indices)
            np.testing.assert_array_equal(expected_vectors.data, actual_vectors.data)
        else:
            np.testing.assert_array_equal(expected_vectors, actual_vectors)

        # JSON
        ml2json.to_json(model, model_name)
        deserialized_model = ml2json.from_json(model_name)
        os.remove(model_name)
        json_predictions = deserialized_model.transform(self.X)

        if model.sparse:
            np.testing.assert_array_equal(expected_vectors.indptr, json_predictions.indptr)
            np.testing.assert_array_equal(expected_vectors.indices, json_predictions.indices)
            np.testing.assert_array_equal(expected_vectors.data, json_predictions.data)
        else:
            np.testing.assert_array_equal(expected_vectors, json_predictions)

    def test_dict_vectorization(self):
        self.check_model(DictVectorizer(), 'dict-vectorizer.json')
        self.check_model(DictVectorizer(sparse=False), 'dict-vectorizer.json')
        self.check_model(DictVectorizer(separator=':'), 'dict-vectorizer.json')
        self.check_model(DictVectorizer(sort=False), 'dict-vectorizer.json')
        self.check_model(DictVectorizer(dtype=np.float32), 'dict-vectorizer.json')
        self.check_model(DictVectorizer(dtype=np.int64), 'dict-vectorizer.json')

    def check_feature_hasher(self, model, data):
        expected_t = model.transform(data)

        serialized_model = ml2json.to_dict(model)
        deserialized_model = ml2json.from_dict(serialized_model)

        actual_t = deserialized_model.transform(data)
        np.testing.assert_array_equal(expected_t.toarray(), actual_t.toarray())

        model_name = 'feature-hasher.json'
        ml2json.to_json(model, model_name)
        deserialized_model = ml2json.from_json(model_name)
        os.remove(model_name)

        actual_t = deserialized_model.transform(data)
        np.testing.assert_array_equal(expected_t.toarray(), actual_t.toarray())

    def test_feature_hasher(self):
        data = [{'a': 1, 'b': 2, 'c': 3}, {'a': 4, 'd': 1}, {'b': 1, 'e': 5}]
        self.check_feature_hasher(FeatureHasher(n_features=8), data)

    def test_feature_hasher_alternate_sign_false(self):
        data = [{'a': 1, 'b': 2, 'c': 3}, {'a': 4, 'd': 1}, {'b': 1, 'e': 5}]
        self.check_feature_hasher(FeatureHasher(n_features=8, alternate_sign=False), data)

    def test_feature_hasher_pair_input(self):
        data = [[('a', 1), ('b', 2), ('c', 3)], [('a', 4), ('d', 1)], [('b', 1), ('e', 5)]]
        self.check_feature_hasher(FeatureHasher(n_features=8, input_type='pair'), data)

    def test_feature_hasher_string_input(self):
        data = [['a', 'b', 'c'], ['a', 'd'], ['b', 'e']]
        self.check_feature_hasher(FeatureHasher(n_features=8, input_type='string'), data)

    def test_feature_hasher_int32_dtype(self):
        data = [{'a': 1, 'b': 2, 'c': 3}, {'a': 4, 'd': 1}, {'b': 1, 'e': 5}]
        self.check_feature_hasher(FeatureHasher(n_features=8, dtype=np.int32), data)
