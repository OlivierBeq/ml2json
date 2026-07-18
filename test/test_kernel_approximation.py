# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np
from sklearn.datasets import load_iris
from sklearn.kernel_approximation import (AdditiveChi2Sampler, Nystroem, PolynomialCountSketch, RBFSampler,
                                          SkewedChi2Sampler)

from src import ml2json


class TestAPI(unittest.TestCase):

    def setUp(self):
        X, _ = load_iris(return_X_y=True)
        self.X = np.abs(X)

    def check_model(self, model, model_name):
        expected_t = model.fit_transform(self.X)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_t = deserialized_model.transform(self.X)

            if hasattr(expected_t, 'toarray'):
                expected_t = expected_t.toarray()
            if hasattr(actual_t, 'toarray'):
                actual_t = actual_t.toarray()

            np.testing.assert_array_almost_equal(expected_t, actual_t)

    def test_additive_chi2_sampler(self):
        self.check_model(AdditiveChi2Sampler(), 'additive-chi2-sampler.json')

    def test_nystroem(self):
        self.check_model(Nystroem(n_components=10, random_state=1234), 'nystroem.json')

    def test_polynomial_count_sketch(self):
        self.check_model(PolynomialCountSketch(n_components=10, random_state=1234), 'polynomial-count-sketch.json')

    def test_rbf_sampler(self):
        self.check_model(RBFSampler(n_components=10, random_state=1234), 'rbf-sampler.json')

    def test_skewed_chi2_sampler(self):
        self.check_model(SkewedChi2Sampler(n_components=10, random_state=1234), 'skewed-chi2-sampler.json')
