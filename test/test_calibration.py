# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np
from sklearn.datasets import make_classification
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src import ml2json


class TestAPI(unittest.TestCase):

    def setUp(self):
        self.X, self.y = make_classification(n_samples=100, n_features=5, n_classes=3, n_informative=3,
                                              n_redundant=0, random_state=0)

    def check_model(self, model, model_name):
        model.fit(self.X, self.y)
        expected_predictions = model.predict_proba(self.X)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_predictions = deserialized_model.predict_proba(self.X)
            np.testing.assert_array_almost_equal(expected_predictions, actual_predictions)

    def test_calibrated_classifier_cv(self):
        self.check_model(CalibratedClassifierCV(RandomForestClassifier(n_estimators=5, random_state=42), cv=3),
                          'calibrated-classifier-cv.json')

    def test_calibrated_classifier_cv_sigmoid(self):
        self.check_model(CalibratedClassifierCV(RandomForestClassifier(n_estimators=5, random_state=42),
                                                 method='sigmoid', cv=3),
                          'calibrated-classifier-cv-sigmoid.json')

    def test_calibrated_classifier_cv_isotonic(self):
        self.check_model(CalibratedClassifierCV(RandomForestClassifier(n_estimators=5, random_state=42),
                                                 method='isotonic', cv=3),
                          'calibrated-classifier-cv-isotonic.json')

    def test_calibrated_classifier_cv_prefit(self):
        base = RandomForestClassifier(n_estimators=5, random_state=42)
        base.fit(self.X, self.y)
        self.check_model(CalibratedClassifierCV(FrozenEstimator(base)), 'calibrated-classifier-cv-prefit.json')

    def test_calibrated_classifier_cv_splitter(self):
        self.check_model(CalibratedClassifierCV(RandomForestClassifier(n_estimators=5, random_state=42),
                                                 cv=StratifiedKFold(n_splits=3)),
                          'calibrated-classifier-cv-splitter.json')

    def test_calibrated_classifier_cv_ensemble_false(self):
        self.check_model(CalibratedClassifierCV(RandomForestClassifier(n_estimators=5, random_state=42),
                                                 cv=3, ensemble=False),
                          'calibrated-classifier-cv-ensemble-false.json')

    def test_calibrated_classifier_cv_linear_estimator(self):
        self.check_model(CalibratedClassifierCV(LogisticRegression(max_iter=1000), cv=3),
                          'calibrated-classifier-cv-linear.json')
