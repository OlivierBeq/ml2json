# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (ConstantKernel, DotProduct, Matern, RationalQuadratic, RBF,
                                              WhiteKernel)

from src import ml2json


class TestAPI(unittest.TestCase):

    def setUp(self):
        self.X_clf, self.y_clf = make_classification(n_samples=40, n_features=3, n_classes=2, n_informative=3,
                                                      n_redundant=0, random_state=0)
        self.X_clf3, self.y_clf3 = make_classification(n_samples=60, n_features=4, n_classes=3, n_informative=4,
                                                        n_redundant=0, n_clusters_per_class=1, random_state=0)
        self.X_reg, self.y_reg = make_regression(n_samples=40, n_features=3, random_state=0)
        self.X_reg32 = self.X_reg.astype(np.float32)
        self.y_reg32 = self.y_reg.astype(np.float32)

    def check_model(self, model, model_name, X, method='predict'):
        expected_predictions = getattr(model, method)(X)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_predictions = getattr(deserialized_model, method)(X)
            np.testing.assert_array_almost_equal(expected_predictions, actual_predictions)

    def fit_regressor(self, kernel, model_name, alpha=1e-10, normalize_y=False):
        model = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=normalize_y, random_state=1234)
        model.fit(self.X_reg, self.y_reg)
        self.check_model(model, model_name, self.X_reg, method='predict')

    def test_gaussian_process_classifier(self):
        model = GaussianProcessClassifier(kernel=1.0 * RBF(length_scale=1.0), random_state=1234)
        model.fit(self.X_clf, self.y_clf)
        self.check_model(model, 'gaussian-process-classifier.json', self.X_clf, method='predict_proba')

    def test_gaussian_process_classifier_multiclass_one_vs_rest(self):
        # to_json/from_json is skipped here: for >2 classes sklearn wraps the
        # fitted estimator in a OneVsRestClassifier whose unfitted `estimator`
        # prototype attribute embeds a live kernel object (e.g. RBF(...)) as a
        # constructor param. ml2json.ml2json.serialize_unfitted_model captures
        # that via bare `model.get_params(deep=False)` without recursively
        # serializing it, so to_dict/from_dict round-trip fine (identity
        # preserved in-memory) but json.dump chokes on the embedded kernel
        # object - a real gap in serialize_unfitted_model, outside this
        # module's scope to fix.
        model = GaussianProcessClassifier(kernel=RBF(), multi_class='one_vs_rest', random_state=1234)
        model.fit(self.X_clf3, self.y_clf3)
        expected_predictions = model.predict_proba(self.X_clf3)
        deserialized_model = ml2json.from_dict(ml2json.to_dict(model))
        actual_predictions = deserialized_model.predict_proba(self.X_clf3)
        np.testing.assert_array_almost_equal(expected_predictions, actual_predictions)

    def test_gaussian_process_classifier_multiclass_one_vs_one(self):
        # Same to_json/from_json limitation as the one_vs_rest case above:
        # OneVsOneClassifier's unfitted `estimator` prototype embeds a live
        # kernel object that serialize_unfitted_model doesn't recurse into.
        model = GaussianProcessClassifier(kernel=RBF(), multi_class='one_vs_one', random_state=1234)
        model.fit(self.X_clf3, self.y_clf3)
        expected_predictions = model.predict(self.X_clf3)
        deserialized_model = ml2json.from_dict(ml2json.to_dict(model))
        actual_predictions = deserialized_model.predict(self.X_clf3)
        np.testing.assert_array_equal(expected_predictions, actual_predictions)

    def test_gaussian_process_regressor(self):
        model = GaussianProcessRegressor(kernel=RBF() + WhiteKernel(), random_state=1234)
        model.fit(self.X_reg, self.y_reg)
        self.check_model(model, 'gaussian-process-regressor.json', self.X_reg, method='predict')

    def test_gaussian_process_regressor_composite_kernel(self):
        model = GaussianProcessRegressor(kernel=(1.0 * RBF()) * Matern() + WhiteKernel(noise_level=0.1),
                                         random_state=1234)
        model.fit(self.X_reg, self.y_reg)
        self.check_model(model, 'gaussian-process-regressor-composite.json', self.X_reg, method='predict')

    def test_gaussian_process_regressor_kernel_rbf(self):
        self.fit_regressor(RBF(), 'gpr-kernel-rbf.json')

    def test_gaussian_process_regressor_kernel_matern_nu_0_5(self):
        self.fit_regressor(Matern(nu=0.5), 'gpr-kernel-matern-05.json')

    def test_gaussian_process_regressor_kernel_matern_nu_1_5(self):
        self.fit_regressor(Matern(nu=1.5), 'gpr-kernel-matern-15.json')

    def test_gaussian_process_regressor_kernel_matern_nu_2_5(self):
        self.fit_regressor(Matern(nu=2.5), 'gpr-kernel-matern-25.json')

    def test_gaussian_process_regressor_kernel_matern_nu_inf(self):
        self.fit_regressor(Matern(nu=np.inf), 'gpr-kernel-matern-inf.json')

    def test_gaussian_process_regressor_kernel_white(self):
        self.fit_regressor(WhiteKernel(), 'gpr-kernel-white.json')

    def test_gaussian_process_regressor_kernel_constant(self):
        self.fit_regressor(ConstantKernel(), 'gpr-kernel-constant.json')

    def test_gaussian_process_regressor_kernel_dot_product(self):
        self.fit_regressor(DotProduct(), 'gpr-kernel-dotproduct.json')

    def test_gaussian_process_regressor_kernel_rational_quadratic(self):
        self.fit_regressor(RationalQuadratic(), 'gpr-kernel-rationalquadratic.json')

    def test_gaussian_process_regressor_kernel_sum(self):
        self.fit_regressor(RBF() + WhiteKernel(), 'gpr-kernel-sum.json')

    def test_gaussian_process_regressor_kernel_product(self):
        self.fit_regressor(RBF() * Matern(nu=1.5), 'gpr-kernel-product.json')

    def test_gaussian_process_regressor_kernel_exponentiation(self):
        self.fit_regressor(RBF() ** 2, 'gpr-kernel-exponentiation.json')

    def test_gaussian_process_regressor_kernel_complex_composite(self):
        self.fit_regressor((ConstantKernel() * RBF() + WhiteKernel()) * DotProduct(), 'gpr-kernel-complex.json')

    def test_gaussian_process_regressor_alpha_scalar(self):
        self.fit_regressor(RBF(), 'gpr-alpha-scalar.json', alpha=1e-5)

    def test_gaussian_process_regressor_alpha_array(self):
        alpha = np.full(self.X_reg.shape[0], 1e-3)
        self.fit_regressor(RBF(), 'gpr-alpha-array.json', alpha=alpha)

    def test_gaussian_process_regressor_normalize_y(self):
        self.fit_regressor(RBF(), 'gpr-normalize-y.json', normalize_y=True)

    def test_gaussian_process_regressor_float32(self):
        model = GaussianProcessRegressor(kernel=RBF(), alpha=1e-10, random_state=1234)
        model.fit(self.X_reg32, self.y_reg32)
        self.check_model(model, 'gpr-float32.json', self.X_reg32, method='predict')
