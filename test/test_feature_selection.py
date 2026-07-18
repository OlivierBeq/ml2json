# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (SelectFromModel, RFE, RFECV, SequentialFeatureSelector,
                                       GenericUnivariateSelect, SelectFdr, SelectFpr, SelectFwe,
                                       SelectKBest, SelectPercentile, VarianceThreshold)

from src import ml2json


class TestAPI(unittest.TestCase):

    def setUp(self):
        self.X, self.y = make_classification(n_samples=100, n_features=10, n_classes=3, n_informative=5,
                                              n_redundant=0, random_state=0)

    def check_model(self, model, model_name):
        model.fit(self.X, self.y)
        expected_transform = model.transform(self.X)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_transform = deserialized_model.transform(self.X)
            np.testing.assert_array_almost_equal(expected_transform, actual_transform)

    def test_select_from_model(self):
        self.check_model(SelectFromModel(RandomForestClassifier(n_estimators=5, random_state=42)),
                          'select-from-model.json')

    def test_rfe(self):
        self.check_model(RFE(RandomForestClassifier(n_estimators=5, random_state=42), n_features_to_select=3),
                          'rfe.json')

    def test_rfecv(self):
        self.check_model(RFECV(RandomForestClassifier(n_estimators=5, random_state=42), cv=3), 'rfecv.json')

    def test_sequential_feature_selector(self):
        self.check_model(SequentialFeatureSelector(RandomForestClassifier(n_estimators=5, random_state=42),
                                                    cv=3, n_features_to_select=3),
                          'sequential-feature-selector.json')

    def test_generic_univariate_select(self):
        self.check_model(GenericUnivariateSelect(), 'generic-univariate-select.json')

    def test_select_fdr(self):
        self.check_model(SelectFdr(), 'select-fdr.json')

    def test_select_fpr(self):
        self.check_model(SelectFpr(), 'select-fpr.json')

    def test_select_fwe(self):
        self.check_model(SelectFwe(), 'select-fwe.json')

    def test_select_kbest(self):
        self.check_model(SelectKBest(k=3), 'select-kbest.json')

    def test_select_percentile(self):
        self.check_model(SelectPercentile(), 'select-percentile.json')

    def test_variance_threshold(self):
        self.check_model(VarianceThreshold(), 'variance-threshold.json')
