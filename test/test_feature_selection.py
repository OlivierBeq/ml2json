# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (SelectFromModel, RFE, RFECV, SequentialFeatureSelector,
                                       GenericUnivariateSelect, SelectFdr, SelectFpr, SelectFwe,
                                       SelectKBest, SelectPercentile, VarianceThreshold,
                                       chi2, f_classif, f_regression, mutual_info_classif)

from src import ml2json


class TestAPI(unittest.TestCase):

    def setUp(self):
        self.X, self.y = make_classification(n_samples=100, n_features=10, n_classes=3, n_informative=5,
                                              n_redundant=0, random_state=0)
        self.X_nonneg = np.abs(self.X)
        self.X_reg, self.y_reg = make_regression(n_samples=100, n_features=10, n_informative=5, random_state=0)

    def check_model(self, model, model_name, X=None, y=None):
        X = self.X if X is None else X
        y = self.y if y is None else y
        model.fit(X, y)
        expected_transform = model.transform(X)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_transform = deserialized_model.transform(X)
            np.testing.assert_array_almost_equal(expected_transform, actual_transform)

    def test_select_from_model(self):
        self.check_model(SelectFromModel(RandomForestClassifier(n_estimators=5, random_state=42)),
                          'select-from-model.json')
        self.check_model(SelectFromModel(RandomForestClassifier(n_estimators=5, random_state=42),
                                         threshold='median'), 'select-from-model.json')
        self.check_model(SelectFromModel(RandomForestClassifier(n_estimators=5, random_state=42),
                                         threshold='mean'), 'select-from-model.json')
        self.check_model(SelectFromModel(RandomForestClassifier(n_estimators=5, random_state=42),
                                         threshold=0.05), 'select-from-model.json')
        self.check_model(SelectFromModel(RandomForestClassifier(n_estimators=5, random_state=42),
                                         max_features=3), 'select-from-model.json')

    def test_rfe(self):
        self.check_model(RFE(RandomForestClassifier(n_estimators=5, random_state=42), n_features_to_select=3),
                          'rfe.json')
        self.check_model(RFE(RandomForestClassifier(n_estimators=5, random_state=42), n_features_to_select=3,
                             step=0.5), 'rfe.json')

    def test_rfecv(self):
        self.check_model(RFECV(RandomForestClassifier(n_estimators=5, random_state=42), cv=3), 'rfecv.json')
        self.check_model(RFECV(RandomForestClassifier(n_estimators=5, random_state=42), cv=3, step=0.5),
                          'rfecv.json')

    def test_sequential_feature_selector(self):
        self.check_model(SequentialFeatureSelector(RandomForestClassifier(n_estimators=5, random_state=42),
                                                    cv=3, n_features_to_select=3),
                          'sequential-feature-selector.json')
        self.check_model(SequentialFeatureSelector(RandomForestClassifier(n_estimators=5, random_state=42),
                                                    cv=3, n_features_to_select=3, direction='backward'),
                          'sequential-feature-selector.json')

    def test_generic_univariate_select(self):
        self.check_model(GenericUnivariateSelect(), 'generic-univariate-select.json')
        self.check_model(GenericUnivariateSelect(score_func=chi2, mode='k_best', param=3),
                          'generic-univariate-select.json', X=self.X_nonneg)
        self.check_model(GenericUnivariateSelect(score_func=mutual_info_classif, mode='percentile', param=50),
                          'generic-univariate-select.json')
        self.check_model(GenericUnivariateSelect(score_func=f_classif, mode='fpr', param=0.1),
                          'generic-univariate-select.json')
        self.check_model(GenericUnivariateSelect(score_func=f_classif, mode='fdr', param=0.1),
                          'generic-univariate-select.json')
        self.check_model(GenericUnivariateSelect(score_func=f_classif, mode='fwe', param=0.1),
                          'generic-univariate-select.json')

    def test_select_fdr(self):
        self.check_model(SelectFdr(), 'select-fdr.json')
        self.check_model(SelectFdr(score_func=chi2), 'select-fdr.json', X=self.X_nonneg)

    def test_select_fpr(self):
        self.check_model(SelectFpr(), 'select-fpr.json')
        self.check_model(SelectFpr(score_func=chi2), 'select-fpr.json', X=self.X_nonneg)

    def test_select_fwe(self):
        self.check_model(SelectFwe(), 'select-fwe.json')
        self.check_model(SelectFwe(score_func=chi2), 'select-fwe.json', X=self.X_nonneg)

    def test_select_kbest(self):
        self.check_model(SelectKBest(k=3), 'select-kbest.json')
        self.check_model(SelectKBest(score_func=chi2, k=3), 'select-kbest.json', X=self.X_nonneg)
        self.check_model(SelectKBest(score_func=mutual_info_classif, k=3), 'select-kbest.json')
        self.check_model(SelectKBest(score_func=f_regression, k=3), 'select-kbest.json',
                          X=self.X_reg, y=self.y_reg)
        self.check_model(SelectKBest(k='all'), 'select-kbest.json')

    def test_select_percentile(self):
        self.check_model(SelectPercentile(), 'select-percentile.json')
        self.check_model(SelectPercentile(score_func=chi2, percentile=30), 'select-percentile.json',
                          X=self.X_nonneg)

    def test_variance_threshold(self):
        self.check_model(VarianceThreshold(), 'variance-threshold.json')
        self.check_model(VarianceThreshold(threshold=0.5), 'variance-threshold.json')

    def test_select_from_model_dtype(self):
        self.check_model(SelectFromModel(RandomForestClassifier(n_estimators=5, random_state=42)),
                          'select-from-model-f32.json', X=self.X.astype(np.float32))
