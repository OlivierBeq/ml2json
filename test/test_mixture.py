# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np
from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

from src import ml2json


class TestAPI(unittest.TestCase):

    def setUp(self):
        self.X, self.y = load_iris(return_X_y=True)

    def check_model(self, model, model_name):
        model.fit(self.X)
        expected_predictions = model.predict(self.X)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_predictions = deserialized_model.predict(self.X)
            np.testing.assert_array_equal(expected_predictions, actual_predictions)

            expected_score = model.score_samples(self.X)
            actual_score = deserialized_model.score_samples(self.X)
            np.testing.assert_array_almost_equal(expected_score, actual_score)

    def test_gaussian_mixture(self):
        self.check_model(GaussianMixture(n_components=3, random_state=1234), 'gaussian-mixture.json')

    def test_gaussian_mixture_covariance_types(self):
        for covariance_type in ['full', 'tied', 'diag', 'spherical']:
            self.check_model(GaussianMixture(n_components=3, covariance_type=covariance_type, random_state=1234),
                             'gaussian-mixture.json')

    def test_gaussian_mixture_init_params(self):
        for init_params in ['kmeans', 'k-means++', 'random', 'random_from_data']:
            self.check_model(GaussianMixture(n_components=3, init_params=init_params, random_state=1234),
                             'gaussian-mixture.json')

    def test_gaussian_mixture_near_singular_covariance(self):
        # More components than well-separated clusters forces per-component
        # covariance estimates toward singular (only reg_covar keeps them invertible).
        for covariance_type in ['full', 'tied', 'diag', 'spherical']:
            self.check_model(GaussianMixture(n_components=10, covariance_type=covariance_type, random_state=1234),
                             'gaussian-mixture.json')

    def test_gaussian_mixture_covariance_symmetry_roundtrip(self):
        model = GaussianMixture(n_components=3, covariance_type='full', random_state=1234)
        model.fit(self.X)

        for i in range(model.covariances_.shape[0]):
            np.testing.assert_array_almost_equal(model.covariances_[i], model.covariances_[i].T)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_model = ml2json.from_dict(serialized_dict_model)

        np.testing.assert_array_almost_equal(model.covariances_, deserialized_model.covariances_)
        for i in range(deserialized_model.covariances_.shape[0]):
            np.testing.assert_array_almost_equal(deserialized_model.covariances_[i], deserialized_model.covariances_[i].T)

    def test_bayesian_gaussian_mixture(self):
        self.check_model(BayesianGaussianMixture(n_components=3, random_state=1234), 'bayesian-gaussian-mixture.json')

    def test_bayesian_gaussian_mixture_covariance_types(self):
        for covariance_type in ['full', 'tied', 'diag', 'spherical']:
            self.check_model(BayesianGaussianMixture(n_components=3, covariance_type=covariance_type, random_state=1234),
                             'bayesian-gaussian-mixture.json')

    def test_bayesian_gaussian_mixture_weight_concentration_prior_type(self):
        for weight_concentration_prior_type in ['dirichlet_process', 'dirichlet_distribution']:
            self.check_model(BayesianGaussianMixture(n_components=3,
                                                      weight_concentration_prior_type=weight_concentration_prior_type,
                                                      random_state=1234),
                             'bayesian-gaussian-mixture.json')

    def test_gaussian_mixture_float32_input(self):
        model = GaussianMixture(n_components=3, random_state=1234)
        self.check_model(model, 'gaussian-mixture.json')
        self.X = self.X.astype(np.float32)
        self.check_model(GaussianMixture(n_components=3, random_state=1234), 'gaussian-mixture.json')
