# -*- coding: utf-8 -*-

import os
import random
import unittest

import numpy as np

from sklearn.datasets import make_classification
from sklearn.feature_extraction import FeatureHasher
from sklearn import svm, discriminant_analysis
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, RandomForestClassifier, IsolationForest,
                              StackingClassifier, VotingClassifier, HistGradientBoostingClassifier,
                              RandomTreesEmbedding)
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB, ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from xgboost import XGBClassifier, XGBRFClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, Pool

from src import sklearn_json as skljson


class TestAPI(unittest.TestCase):

    def setUp(self):
        self.X, self.y = make_classification(n_samples=50, n_features=3, n_classes=3, n_informative=3, n_redundant=0, random_state=0, shuffle=False)

        feature_hasher = FeatureHasher(n_features=3)
        features = []
        for i in range(0, 100):
            features.append({'a': random.randint(0, 2), 'b': random.randint(3, 5), 'c': random.randint(6, 8)})
        self.y_sparse = [random.randint(0, 2) for i in range(0, 100)]
        self.X_sparse = feature_hasher.transform(features)

    def check_model(self, model, model_name, abs=False):
        # Given
        if abs:
            model.fit(np.absolute(self.X), self.y)
        else:
            model.fit(self.X, self.y)

        # When
        serialized_model = skljson.to_dict(model)
        deserialized_model = skljson.from_dict(serialized_model)

        # Then
        expected_predictions = model.predict(self.X)
        actual_predictions = deserialized_model.predict(self.X)

        np.testing.assert_array_equal(expected_predictions, actual_predictions)

        # When
        skljson.to_json(model, model_name)
        deserialized_model = skljson.from_json(model_name)
        os.remove(model_name)

        # JSON
        actual_predictions = deserialized_model.predict(self.X)

        np.testing.assert_array_equal(expected_predictions, actual_predictions)

    def check_sparse_model(self, model, model_name, abs=False):
        # Given
        if abs:
            model.fit(np.absolute(self.X_sparse), self.y_sparse)
        else:
            model.fit(self.X_sparse, self.y_sparse)

        # When
        serialized_model = skljson.to_dict(model)
        deserialized_model = skljson.from_dict(serialized_model)

        # Then
        expected_predictions = model.predict(self.X)
        actual_predictions = deserialized_model.predict(self.X)

        np.testing.assert_array_equal(expected_predictions, actual_predictions)

        # JSON
        skljson.to_json(model, model_name)
        deserialized_model = skljson.from_json(model_name)
        os.remove(model_name)

        # Then
        actual_predictions = deserialized_model.predict(self.X)

        np.testing.assert_array_equal(expected_predictions, actual_predictions)

    def test_bernoulli_nb(self):
        self.check_model(BernoulliNB(), 'bernoulli-nb.json')
        self.check_sparse_model(BernoulliNB(), 'bernoulli-nb.json')

    def test_guassian_nb(self):
        self.check_model(GaussianNB(), 'gaussian-nb.json')
        # No sklearn implementation for sparse matrix

    def test_multinomial_nb(self):
        self.check_model(MultinomialNB(), 'multinomial-nb.json', abs=True)
        self.check_sparse_model(MultinomialNB(), 'multinomial-nb.json', abs=True)

    def test_complement_nb(self):
        self.check_model(ComplementNB(), 'complement-nb.json', abs=True)
        # No sklearn implementation for sparse matrix

    def test_logistic_regression(self):
        self.check_model(LogisticRegression(), 'lr.json')
        self.check_sparse_model(LogisticRegression(), 'lr.json')

    def test_lda(self):
        self.check_model(discriminant_analysis.LinearDiscriminantAnalysis(), 'lda.json')
        # No sklearn implementation for sparse matrix

    def test_qda(self):
        self.check_model(discriminant_analysis.QuadraticDiscriminantAnalysis(), 'qda.json')
        # No sklearn implementation for sparse matrix

    def test_svm(self):
        self.check_model(svm.SVC(gamma=0.001, C=100., kernel='linear'), 'svm.json')
        self.check_sparse_model(svm.SVC(gamma=0.001, C=100., kernel='linear'), 'svm.json')

    def test_decision_tree(self):
        self.check_model(DecisionTreeClassifier(), 'dt.json')
        self.check_sparse_model(DecisionTreeClassifier(), 'dt.json')

    def test_extra_tree(self):
        self.check_model(ExtraTreeClassifier(), 'extra-tree.json')
        self.check_sparse_model(ExtraTreeClassifier(), 'extra-tree.json')

    def test_gradient_boosting(self):
        self.check_model(GradientBoostingClassifier(n_estimators=25, learning_rate=1.0), 'gb.json')
        self.check_sparse_model(GradientBoostingClassifier(n_estimators=25, learning_rate=1.0), 'gb.json')

    def test_random_forest(self):
        self.check_model(RandomForestClassifier(n_estimators=10, max_depth=5, random_state=0), 'rf.json')
        self.check_sparse_model(RandomForestClassifier(n_estimators=10, max_depth=5, random_state=0), 'rf.json')

    def test_perceptron(self):
        self.check_model(Perceptron(), 'perceptron.json')
        self.check_sparse_model(Perceptron(), 'perceptron.json')

    def test_mlp(self):
        self.check_model(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1), 'mlp.json')
        self.check_sparse_model(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1), 'mlp.json')

    def test_xgboost_classifier(self):
        self.check_model(XGBClassifier(), 'xgb_classifier.json')

    def test_xgboost_rf_classifier(self):
        self.check_model(XGBRFClassifier(), 'xgb_rf_classifier.json')

    def test_lightgbm_classifier(self):
        self.check_model(LGBMClassifier(), 'lightgbm_classifier.json')

    def check_catboost_model(self, model, model_name, abs=False):
        # Given
        if abs:
            model.fit(np.absolute(self.X), self.y)
        else:
            model.fit(self.X, self.y)

        pool = Pool(data=self.X, label=self.y, feature_names=list(range(self.X.shape[0])))

        # When
        serialized_model = skljson.to_dict(model, pool)
        deserialized_model = skljson.from_dict(serialized_model)

        # Then
        expected_predictions = model.predict(self.X)
        actual_predictions = deserialized_model.predict(self.X)

        np.testing.assert_array_equal(expected_predictions, actual_predictions)

        # JSON
        skljson.to_json(model, model_name)
        deserialized_model = skljson.from_json(model_name)
        os.remove(model_name)
        json_predictions = deserialized_model.predict(self.X)

        np.testing.assert_array_equal(expected_predictions, json_predictions)

    def test_catboost_classifier(self):
        self.check_model(CatBoostClassifier(allow_writing_files=False, verbose=False), 'catboost-cls.json')

    def test_adaboost_classifier(self):
        self.check_model(AdaBoostClassifier(n_estimators=25, learning_rate=1.0), 'adaboost-cls.json')
        self.check_sparse_model(AdaBoostClassifier(n_estimators=25, learning_rate=1.0), 'adaboost-cls.json')

    def test_bagging_classifier(self):
        self.check_model(BaggingClassifier(n_estimators=25), 'bagging-cls.json')
        self.check_sparse_model(BaggingClassifier(n_estimators=25), 'bagging-cls.json')

    def test_extratrees_classifier(self):
        self.check_model(ExtraTreesClassifier(n_estimators=100, max_depth=5, random_state=1234), 'extra-trees-cls.json')
        self.check_sparse_model(ExtraTreesClassifier(n_estimators=100, max_depth=5, random_state=1234), 'extra-trees-cls.json')
        self.check_model(ExtraTreesClassifier(n_estimators=100, max_depth=5, oob_score=True, bootstrap=True, random_state=1234), 'extra-trees-cls.json')
        self.check_sparse_model(ExtraTreesClassifier(n_estimators=100, max_depth=5, oob_score=True, bootstrap=True, random_state=1234), 'extra-trees-cls.json')

    def test_isolation_forest(self):
        self.check_model(IsolationForest(n_estimators=100, random_state=1234), 'isolation-forest.json')
        self.check_sparse_model(IsolationForest(n_estimators=100, random_state=1234), 'isolation-forest-cls.json')
        self.check_model(IsolationForest(n_estimators=100, bootstrap=True, random_state=1234), 'isolation-forest-cls.json')
        self.check_sparse_model(IsolationForest(n_estimators=100, bootstrap=True, random_state=1234), 'isolation-forest-cls.json')

    def check_random_trees_embedding_model(self, model, model_name):
        model.fit(self.X)

        # When
        serialized_model = skljson.to_dict(model)
        deserialized_model = skljson.from_dict(serialized_model)

        # Then
        expected_predictions = model.transform(self.X).toarray()
        actual_predictions = deserialized_model.transform(self.X).toarray()

        np.testing.assert_array_equal(expected_predictions, actual_predictions)

        # When
        skljson.to_json(model, model_name)
        deserialized_model = skljson.from_json(model_name)
        os.remove(model_name)

        # JSON
        actual_predictions = deserialized_model.transform(self.X).toarray()

        np.testing.assert_array_equal(expected_predictions, actual_predictions)

    def test_random_trees_embedding(self):
        self.check_random_trees_embedding_model(RandomTreesEmbedding(n_estimators=100, random_state=1234), 'random-trees-embedding.json')
