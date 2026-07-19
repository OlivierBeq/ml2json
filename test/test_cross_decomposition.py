# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np
from sklearn.cross_decomposition import (CCA, PLSCanonical,
                                         PLSRegression, PLSSVD)

from src import ml2json


class TestAPI(unittest.TestCase):

    def setUp(self):
        n = 500
        # 2 latents vars:
        l1 = np.random.normal(size=n)
        l2 = np.random.normal(size=n)

        latents = np.array([l1, l1, l2, l2]).T
        self.X = latents + np.random.normal(size=4 * n).reshape((n, 4))
        self.y = latents + np.random.normal(size=4 * n).reshape((n, 4))

    def check_transform_model(self, model, model_name, data, labels):
        model.fit(data, labels)
        expected_t = model.transform(data)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:

            # for key in sorted(model.__dict__.keys()):
            #     if isinstance(model.__dict__[key], np.ndarray):
            #         print(key, (model.__dict__[key] == deserialized_dict_model.__dict__[key]).all())
            #     else:
            #         print(key, model.__dict__[key] == deserialized_dict_model.__dict__[key])

            actual_t = deserialized_model.transform(data)

            np.testing.assert_array_almost_equal(expected_t, actual_t)

    def check_fittransform_model(self, model, model_name, data, labels):
        expected_ft = model.fit_transform(data, labels)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:

            # for key in sorted(model.__dict__.keys()):
            #     if isinstance(model.__dict__[key], np.ndarray):
            #         print(key, (model.__dict__[key] == deserialized_dict_model.__dict__[key]).all())
            #     else:
            #         print(key, model.__dict__[key] == deserialized_dict_model.__dict__[key])

            actual_ft = deserialized_model.fit_transform(data, labels)

            np.testing.assert_array_almost_equal(expected_ft, actual_ft)

    def check_predict_model(self, model, model_name, data, labels):
        model.fit(data, labels)
        expected_p = model.predict(data)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:

            # for key in sorted(model.__dict__.keys()):
            #     if isinstance(model.__dict__[key], np.ndarray):
            #         print(key, (model.__dict__[key] == deserialized_dict_model.__dict__[key]).all())
            #     else:
            #         print(key, model.__dict__[key] == deserialized_dict_model.__dict__[key])

            actual_p = deserialized_model.predict(data)

            np.testing.assert_array_equal(expected_p, actual_p)

    def check_fitpredict_model(self, model, model_name, data, labels):
        expected_fp = model.fit_predict(data, labels)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:

            # for key in sorted(model.__dict__.keys()):
            #     if isinstance(model.__dict__[key], np.ndarray):
            #         print(key, (model.__dict__[key] == deserialized_dict_model.__dict__[key]).all())
            #     else:
            #         print(key, model.__dict__[key] == deserialized_dict_model.__dict__[key])

            actual_fp = deserialized_model.fit_predict(data, labels)

            np.testing.assert_array_equal(expected_fp, actual_fp)

    def test_cca(self):
        for model in [CCA(), CCA(n_components=1), CCA(n_components=4),
                     CCA(scale=False), CCA(max_iter=5000, tol=1e-9)]:
            self.check_transform_model(model, 'cca.json', self.X, self.y)
            self.check_fittransform_model(model, 'cca.json', self.X, self.y)
            self.check_predict_model(model, 'cca.json', self.X, self.y)

    def test_pls_canonical(self):
        for model in [PLSCanonical(), PLSCanonical(algorithm='svd'),
                     PLSCanonical(algorithm='nipals'), PLSCanonical(n_components=1),
                     PLSCanonical(n_components=4), PLSCanonical(scale=False)]:
            self.check_transform_model(model, 'pls-canonical.json', self.X, self.y)
            self.check_fittransform_model(model, 'pls-canonical.json', self.X, self.y)
            self.check_predict_model(model, 'pls-canonical.json', self.X, self.y)

    def test_pls_regression(self):
        for model in [PLSRegression(), PLSRegression(scale=False),
                     PLSRegression(n_components=1), PLSRegression(n_components=4),
                     PLSRegression(max_iter=5000, tol=1e-9)]:
            self.check_transform_model(model, 'pls-regression.json', self.X, self.y)
            self.check_fittransform_model(model, 'pls-regression.json', self.X, self.y)
            self.check_predict_model(model, 'pls-regression.json', self.X, self.y)

        # single-target y still produces 2D internal arrays but exercises the
        # coef_ shape (n_features, 1) path
        self.check_transform_model(PLSRegression(), 'pls-regression.json', self.X, self.y[:, 0])
        self.check_predict_model(PLSRegression(), 'pls-regression.json', self.X, self.y[:, 0])

    def test_pls_svd(self):
        for model in [PLSSVD(), PLSSVD(n_components=1), PLSSVD(n_components=4),
                     PLSSVD(scale=False)]:
            self.check_transform_model(model, 'pls-svd.json', self.X, self.y)
            self.check_fittransform_model(model, 'pls-svd.json', self.X, self.y)

    def test_wide_data_rank_deficient(self):
        # More features than samples forces a rank-deficient X^T Y cross-covariance
        # matrix - exercise the non-invertible-matrix path for each estimator.
        rng = np.random.RandomState(0)
        n_samples, n_features = 8, 20
        X_wide = rng.normal(size=(n_samples, n_features))
        y_wide = rng.normal(size=(n_samples, 3))

        for model, name in [(CCA(n_components=2), 'cca-wide.json'),
                            (PLSCanonical(n_components=2), 'pls-canonical-wide.json'),
                            (PLSRegression(n_components=2), 'pls-regression-wide.json')]:
            self.check_transform_model(model, name, X_wide, y_wide)
            self.check_predict_model(model, name, X_wide, y_wide)

        self.check_transform_model(PLSSVD(n_components=2), 'pls-svd-wide.json', X_wide, y_wide)

    def test_float32_input(self):
        X32 = self.X.astype(np.float32)
        y32 = self.y.astype(np.float32)

        for model, name in [(CCA(), 'cca-f32.json'),
                            (PLSCanonical(), 'pls-canonical-f32.json'),
                            (PLSRegression(), 'pls-regression-f32.json')]:
            self.check_transform_model(model, name, X32, y32)
            self.check_predict_model(model, name, X32, y32)

        self.check_transform_model(PLSSVD(), 'pls-svd-f32.json', X32, y32)
