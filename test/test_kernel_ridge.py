# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np
from sklearn.datasets import make_regression
from sklearn.kernel_ridge import KernelRidge

from src import ml2json


def _rbf_like_kernel(x, y, gamma=1.0):
    return np.exp(-gamma * np.sum((x - y) ** 2))


class TestAPI(unittest.TestCase):

    def setUp(self):
        self.X, self.y = make_regression(n_samples=50, n_features=3, random_state=0)
        self.X_nonneg = np.random.RandomState(0).uniform(0, 1, size=(50, 3))

    def check_model(self, model, model_name, X=None, y=None):
        X = self.X if X is None else X
        y = self.y if y is None else y
        model.fit(X, y)
        expected_predictions = model.predict(X)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_predictions = deserialized_model.predict(X)
            np.testing.assert_array_almost_equal(expected_predictions, actual_predictions)

    def test_kernel_ridge(self):
        self.check_model(KernelRidge(kernel='rbf'), 'kernel-ridge.json')

    def test_kernel_ridge_linear(self):
        self.check_model(KernelRidge(kernel='linear'), 'kernel-ridge-linear.json')

    def test_kernel_ridge_poly(self):
        self.check_model(KernelRidge(kernel='poly'), 'kernel-ridge-poly.json')

    def test_kernel_ridge_sigmoid(self):
        self.check_model(KernelRidge(kernel='sigmoid'), 'kernel-ridge-sigmoid.json')

    def test_kernel_ridge_laplacian(self):
        self.check_model(KernelRidge(kernel='laplacian'), 'kernel-ridge-laplacian.json')

    def test_kernel_ridge_chi2(self):
        # chi2_kernel's own `gamma=None` default multiplies K by None and
        # crashes with a numpy casting error - unrelated to ml2json, reproduces
        # identically with plain sklearn.metrics.pairwise.chi2_kernel. gamma
        # must be set explicitly to sidestep it.
        self.check_model(KernelRidge(kernel='chi2', gamma=1.0), 'kernel-ridge-chi2.json', X=self.X_nonneg)

    def test_kernel_ridge_poly_params(self):
        self.check_model(KernelRidge(kernel='poly', degree=4, gamma=0.5, coef0=2.0), 'kernel-ridge-poly-params.json')

    def test_kernel_ridge_rbf_gamma(self):
        self.check_model(KernelRidge(kernel='rbf', gamma=0.3), 'kernel-ridge-rbf-gamma.json')

    def test_kernel_ridge_sigmoid_params(self):
        self.check_model(KernelRidge(kernel='sigmoid', gamma=0.2, coef0=0.5), 'kernel-ridge-sigmoid-params.json')

    def test_kernel_ridge_precomputed(self):
        K = self.X @ self.X.T
        np.testing.assert_array_almost_equal(K, K.T)
        self.check_model(KernelRidge(kernel='precomputed'), 'kernel-ridge-precomputed.json', X=K)

    def test_kernel_ridge_alpha_array_multioutput(self):
        y_multi = np.stack([self.y, self.y * 2 + 1], axis=1)
        self.check_model(KernelRidge(kernel='linear', alpha=np.array([0.5, 2.0])),
                         'kernel-ridge-alpha-array.json', y=y_multi)

    def test_kernel_ridge_float32(self):
        self.check_model(KernelRidge(kernel='rbf'), 'kernel-ridge-float32.json',
                         X=self.X.astype(np.float32), y=self.y.astype(np.float32))

    def test_kernel_ridge_callable_kernel(self):
        self.check_model(KernelRidge(kernel=_rbf_like_kernel, kernel_params={'gamma': 0.7}),
                         'kernel-ridge-callable.json')
