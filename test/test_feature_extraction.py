# -*- coding: utf-8 -*-

import os
import re
import unittest
from collections import Counter

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import DictVectorizer

from src import sklearn_json as skljson


class TestAPI(unittest.TestCase):

    def setUp(self):
        newsgroup = fetch_20newsgroups(subset='train', categories=['sci.space'], remove=('headers', 'footers', 'quotes'))

        self.X = [
            Counter(tok.lower() for tok in re.findall(r"\w+", text))
            for text in newsgroup.data
        ]

    def check_model(self, model, model_name):
        expected_vectors = model.fit_transform(self.X)

        serialized_model = skljson.to_dict(model)
        deserialized_model = skljson.from_dict(serialized_model)

        actual_vectors = deserialized_model.fit_transform(self.X)

        if model.sparse:
            np.testing.assert_array_equal(expected_vectors.indptr, actual_vectors.indptr)
            np.testing.assert_array_equal(expected_vectors.indices, actual_vectors.indices)
            np.testing.assert_array_equal(expected_vectors.data, actual_vectors.data)
        else:
            np.testing.assert_array_equal(expected_vectors, actual_vectors)

        # JSON
        skljson.to_json(model, model_name)
        deserialized_model = skljson.from_json(model_name)
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
