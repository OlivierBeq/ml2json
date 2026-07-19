# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import LabelPropagation, LabelSpreading, SelfTrainingClassifier

from src import ml2json


class TestAPI(unittest.TestCase):

    def setUp(self):
        self.X, self.y = make_classification(n_samples=100, n_features=5, n_classes=2, n_informative=3,
                                              n_redundant=0, random_state=0)
        # Semi-supervised setting: a third of the labels are unknown (-1).
        rng = np.random.RandomState(0)
        self.y_semi = self.y.copy()
        self.y_semi[rng.choice(len(self.y_semi), size=len(self.y_semi) // 3, replace=False)] = -1

    def check_model(self, model, model_name, y=None):
        model.fit(self.X, self.y if y is None else y)
        expected_predictions = model.predict(self.X)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_predictions = deserialized_model.predict(self.X)
            np.testing.assert_array_equal(expected_predictions, actual_predictions)

    def test_label_propagation(self):
        self.check_model(LabelPropagation(), 'label-propagation.json', y=self.y_semi)

    def test_label_spreading(self):
        self.check_model(LabelSpreading(), 'label-spreading.json', y=self.y_semi)

    def test_self_training_classifier(self):
        self.check_model(SelfTrainingClassifier(RandomForestClassifier(n_estimators=5, random_state=42)),
                          'self-training-classifier.json', y=self.y_semi)

    def test_label_propagation_knn(self):
        self.check_model(LabelPropagation(kernel='knn', n_neighbors=5), 'label-propagation-knn.json',
                          y=self.y_semi)

    def test_label_propagation_gamma(self):
        self.check_model(LabelPropagation(kernel='rbf', gamma=0.5), 'label-propagation-gamma.json',
                          y=self.y_semi)

    def test_label_propagation_mostly_unlabeled(self):
        rng = np.random.RandomState(1)
        y_mostly_unlabeled = self.y.copy()
        unlabeled_idx = rng.choice(len(y_mostly_unlabeled), size=int(len(y_mostly_unlabeled) * 0.9),
                                   replace=False)
        y_mostly_unlabeled[unlabeled_idx] = -1
        self.check_model(LabelPropagation(), 'label-propagation-mostly-unlabeled.json', y=y_mostly_unlabeled)

    def test_label_spreading_knn(self):
        self.check_model(LabelSpreading(kernel='knn', n_neighbors=5, alpha=0.5), 'label-spreading-knn.json',
                          y=self.y_semi)

    def test_label_spreading_alpha(self):
        self.check_model(LabelSpreading(alpha=0.8), 'label-spreading-alpha.json', y=self.y_semi)

    def test_self_training_classifier_threshold(self):
        self.check_model(SelfTrainingClassifier(RandomForestClassifier(n_estimators=5, random_state=42),
                                                 threshold=0.9, max_iter=5, verbose=True),
                          'self-training-classifier-threshold.json', y=self.y_semi)

    def test_self_training_classifier_k_best(self):
        self.check_model(SelfTrainingClassifier(RandomForestClassifier(n_estimators=5, random_state=42),
                                                 criterion='k_best', k_best=5, max_iter=3),
                          'self-training-classifier-k-best.json', y=self.y_semi)

    def test_self_training_classifier_no_samples_added(self):
        self.check_model(SelfTrainingClassifier(RandomForestClassifier(n_estimators=5, random_state=42),
                                                 threshold=0.999999, max_iter=1),
                          'self-training-classifier-no-added.json', y=self.y_semi)
