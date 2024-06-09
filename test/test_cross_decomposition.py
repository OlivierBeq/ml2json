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

            for key in sorted(model.__dict__.keys()):
                if isinstance(model.__dict__[key], np.ndarray):
                    print(key, (model.__dict__[key] == deserialized_dict_model.__dict__[key]).all())
                else:
                    print(key, model.__dict__[key] == deserialized_dict_model.__dict__[key])

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
        self.check_transform_model(CCA(), 'cca.json', self.X, self.y)
        self.check_fittransform_model(CCA(), 'cca.json', self.X, self.y)
        self.check_predict_model(CCA(), 'cca.json', self.X, self.y)

    def test_pls_canonical(self):
        self.check_transform_model(PLSCanonical(), 'pls-canonical.json', self.X, self.y)
        self.check_fittransform_model(PLSCanonical(), 'pls-canonical.json', self.X, self.y)
        self.check_predict_model(PLSCanonical(), 'pls-canonical.json', self.X, self.y)

    def test_pls_regression(self):
        self.check_transform_model(PLSRegression(), 'pls-regression.json', self.X, self.y)
        self.check_fittransform_model(PLSRegression(), 'pls-regression.json', self.X, self.y)
        self.check_predict_model(PLSRegression(), 'pls-regression.json', self.X, self.y)

    def test_pls_svd(self):
        self.check_transform_model(PLSSVD(), 'pls-svd.json', self.X, self.y)
        self.check_fittransform_model(PLSSVD(), 'pls-svd.json', self.X, self.y)
