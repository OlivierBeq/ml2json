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
        # Column 0 duplicated as a new column 4: covariance_/precision_ end up
        # exactly rank-deficient (n_samples > n_features, but two columns collinear).
        self.X_singular = np.hstack([self.X, self.X[:, [0]]])
        # n_features > n_samples: covariance_ is rank-deficient by construction.
        self.X_highdim = np.random.RandomState(0).randn(10, 20)
        self.X32 = self.X.astype(np.float32)

    def check_model(self, model, model_name, X=None):
        X = self.X if X is None else X
        model.fit(X)
        expected = model.mahalanobis(X)
        covariance = model.covariance_
        precision = model.precision_
        np.testing.assert_array_almost_equal(covariance, covariance.T)
        np.testing.assert_array_almost_equal(precision, precision.T)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            np.testing.assert_array_almost_equal(covariance, deserialized_model.covariance_)
            np.testing.assert_array_almost_equal(precision, deserialized_model.precision_)
            np.testing.assert_array_almost_equal(deserialized_model.covariance_, deserialized_model.covariance_.T)
            np.testing.assert_array_almost_equal(deserialized_model.precision_, deserialized_model.precision_.T)
            actual = deserialized_model.mahalanobis(X)
            np.testing.assert_array_almost_equal(expected, actual)

    def test_empirical_covariance(self):
        self.check_model(EmpiricalCovariance(), 'empirical-covariance.json')

    def test_empirical_covariance_singular(self):
        self.check_model(EmpiricalCovariance(), 'empirical-covariance-singular.json', X=self.X_singular)

    def test_empirical_covariance_highdim(self):
        self.check_model(EmpiricalCovariance(), 'empirical-covariance-highdim.json', X=self.X_highdim)

    def test_empirical_covariance_float32(self):
        self.check_model(EmpiricalCovariance(), 'empirical-covariance-float32.json', X=self.X32)

    def test_graphical_lasso(self):
        self.check_model(GraphicalLasso(), 'graphical-lasso.json')

    def test_graphical_lasso_singular(self):
        self.check_model(GraphicalLasso(), 'graphical-lasso-singular.json', X=self.X_singular)

    def test_graphical_lasso_highdim(self):
        self.check_model(GraphicalLasso(), 'graphical-lasso-highdim.json', X=self.X_highdim)

    def test_graphical_lasso_cv(self):
        self.check_model(GraphicalLassoCV(cv=3), 'graphical-lasso-cv.json')

    def test_graphical_lasso_cv_highdim(self):
        self.check_model(GraphicalLassoCV(cv=2), 'graphical-lasso-cv-highdim.json', X=self.X_highdim)

    def test_ledoit_wolf(self):
        self.check_model(LedoitWolf(), 'ledoit-wolf.json')

    def test_ledoit_wolf_singular(self):
        self.check_model(LedoitWolf(), 'ledoit-wolf-singular.json', X=self.X_singular)

    def test_ledoit_wolf_highdim(self):
        self.check_model(LedoitWolf(), 'ledoit-wolf-highdim.json', X=self.X_highdim)

    def test_min_cov_det(self):
        self.check_model(MinCovDet(random_state=1234), 'min-cov-det.json')

    def test_min_cov_det_singular(self):
        self.check_model(MinCovDet(random_state=1234), 'min-cov-det-singular.json', X=self.X_singular)

    def test_min_cov_det_float32(self):
        self.check_model(MinCovDet(random_state=1234), 'min-cov-det-float32.json', X=self.X32)

    def test_min_cov_det_support_fraction_low(self):
        self.check_model(MinCovDet(random_state=1234, support_fraction=0.51), 'min-cov-det-sf-low.json')

    def test_min_cov_det_support_fraction_full(self):
        self.check_model(MinCovDet(random_state=1234, support_fraction=1.0), 'min-cov-det-sf-full.json')

    def test_oas(self):
        self.check_model(OAS(), 'oas.json')

    def test_oas_singular(self):
        self.check_model(OAS(), 'oas-singular.json', X=self.X_singular)

    def test_oas_highdim(self):
        self.check_model(OAS(), 'oas-highdim.json', X=self.X_highdim)

    def test_shrunk_covariance(self):
        self.check_model(ShrunkCovariance(), 'shrunk-covariance.json')

    def test_shrunk_covariance_singular(self):
        self.check_model(ShrunkCovariance(), 'shrunk-covariance-singular.json', X=self.X_singular)

    def test_shrunk_covariance_highdim(self):
        self.check_model(ShrunkCovariance(), 'shrunk-covariance-highdim.json', X=self.X_highdim)

    def check_elliptic_envelope(self, model, model_name, X=None):
        X = self.X if X is None else X
        model.fit(X)
        expected_predictions = model.predict(X)
        covariance = model.covariance_
        precision = model.precision_
        np.testing.assert_array_almost_equal(covariance, covariance.T)
        np.testing.assert_array_almost_equal(precision, precision.T)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_predictions = deserialized_model.predict(X)
            np.testing.assert_array_equal(expected_predictions, actual_predictions)
            np.testing.assert_array_almost_equal(covariance, deserialized_model.covariance_)
            np.testing.assert_array_almost_equal(precision, deserialized_model.precision_)

    def test_elliptic_envelope(self):
        self.check_elliptic_envelope(EllipticEnvelope(random_state=1234), 'elliptic-envelope.json')

    def test_elliptic_envelope_singular(self):
        self.check_elliptic_envelope(EllipticEnvelope(random_state=1234), 'elliptic-envelope-singular.json',
                                     X=self.X_singular)

    def test_elliptic_envelope_highdim(self):
        self.check_elliptic_envelope(EllipticEnvelope(random_state=1234), 'elliptic-envelope-highdim.json',
                                     X=self.X_highdim)

    def test_elliptic_envelope_support_fraction_low(self):
        self.check_elliptic_envelope(EllipticEnvelope(random_state=1234, support_fraction=0.51),
                                     'elliptic-envelope-sf-low.json')

    def test_elliptic_envelope_support_fraction_full(self):
        self.check_elliptic_envelope(EllipticEnvelope(random_state=1234, support_fraction=1.0),
                                     'elliptic-envelope-sf-full.json')
