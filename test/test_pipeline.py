# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np
from sklearn.datasets import make_classification
from sklearn.pipeline import FeatureUnion, Pipeline

from src import ml2json


class TestAPI(unittest.TestCase):

    def setUp(self):
        self.X, self.y = make_classification(n_samples=10_000, random_state=12340)

    def test_pipeline(self):
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC

        X_train, X_test, y_train, _ = train_test_split(self.X, self.y, random_state=1234)

        pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(C=10))])
        pipe.fit(X_train, y_train)
        expected = pipe.predict(X_test)

        serialized_dict_model = ml2json.to_dict(pipe)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        ml2json.to_json(pipe, 'pipe.json')
        deserialized_json_model = ml2json.from_json('pipe.json')
        os.remove('pipe.json')

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual = deserialized_model.predict(X_test)
            np.testing.assert_array_equal(expected, actual)
