# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np
from sklearn.isotonic import IsotonicRegression

from src import ml2json


class TestAPI(unittest.TestCase):

    def setUp(self):
        rng = np.random.RandomState(1234)
        self.x = np.sort(rng.rand(100)) * 10
        self.y = self.x + rng.normal(size=100)

    def test_isotonic_regression(self):
        model = IsotonicRegression()
        model.fit(self.x, self.y)
        expected_predictions = model.predict(self.x)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        model_name = 'isotonic-regression.json'
        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_predictions = deserialized_model.predict(self.x)
            np.testing.assert_array_almost_equal(expected_predictions, actual_predictions)

    def test_isotonic_regression_out_of_bounds(self):
        model = IsotonicRegression(out_of_bounds='clip')
        model.fit(self.x, self.y)

        query = np.array([-5.0, 0.0, 20.0])
        expected_predictions = model.predict(query)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        actual_predictions = deserialized_dict_model.predict(query)
        np.testing.assert_array_almost_equal(expected_predictions, actual_predictions)
