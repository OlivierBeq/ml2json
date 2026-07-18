# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier, OutputCodeClassifier

from src import ml2json


class TestAPI(unittest.TestCase):

    def setUp(self):
        self.X, self.y = make_classification(n_samples=100, n_features=5, n_classes=3, n_informative=3,
                                              n_redundant=0, random_state=0)

    def check_model(self, model, model_name):
        model.fit(self.X, self.y)
        expected_predictions = model.predict(self.X)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_predictions = deserialized_model.predict(self.X)
            np.testing.assert_array_equal(expected_predictions, actual_predictions)

    def test_one_vs_one_classifier(self):
        self.check_model(OneVsOneClassifier(RandomForestClassifier(n_estimators=5, random_state=42)),
                          'one-vs-one-classifier.json')

    def test_one_vs_rest_classifier(self):
        self.check_model(OneVsRestClassifier(RandomForestClassifier(n_estimators=5, random_state=42)),
                          'one-vs-rest-classifier.json')

    def test_output_code_classifier(self):
        self.check_model(OutputCodeClassifier(RandomForestClassifier(n_estimators=5, random_state=42),
                                               random_state=42),
                          'output-code-classifier.json')
