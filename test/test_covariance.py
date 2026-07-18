# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np
from sklearn.datasets import load_iris
from sklearn.covariance import (EllipticEnvelope, EmpiricalCovariance, GraphicalLasso, GraphicalLassoCV,
                                LedoitWolf, MinCovDet, OAS, ShrunkCovariance)

from src import ml2json


class TestAPI(unittest.TestCase):

    def setUp(self):
        self.X, self.y = load_iris(return_X_y=True)

    def check_model(self, model, model_name):
        model.fit(self.X)
        expected = model.mahalanobis(self.X)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual = deserialized_model.mahalanobis(self.X)
            np.testing.assert_array_almost_equal(expected, actual)

    def test_empirical_covariance(self):
        self.check_model(EmpiricalCovariance(), 'empirical-covariance.json')

    def test_graphical_lasso(self):
        self.check_model(GraphicalLasso(), 'graphical-lasso.json')

    def test_graphical_lasso_cv(self):
        self.check_model(GraphicalLassoCV(cv=3), 'graphical-lasso-cv.json')

    def test_ledoit_wolf(self):
        self.check_model(LedoitWolf(), 'ledoit-wolf.json')

    def test_min_cov_det(self):
        self.check_model(MinCovDet(random_state=1234), 'min-cov-det.json')

    def test_oas(self):
        self.check_model(OAS(), 'oas.json')

    def test_shrunk_covariance(self):
        self.check_model(ShrunkCovariance(), 'shrunk-covariance.json')

    def test_elliptic_envelope(self):
        model = EllipticEnvelope(random_state=1234)
        model.fit(self.X)
        expected_predictions = model.predict(self.X)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        model_name = 'elliptic-envelope.json'
        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_predictions = deserialized_model.predict(self.X)
            np.testing.assert_array_equal(expected_predictions, actual_predictions)
