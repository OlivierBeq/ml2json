# -*- coding: utf-8 -*-

import os
import random
import unittest

import numpy as np
import scipy as sp

from sklearn.datasets import make_classification
from sklearn.feature_extraction import FeatureHasher
from sklearn import svm, discriminant_analysis
from sklearn.linear_model import (LogisticRegression, Perceptron, LogisticRegressionCV,
                                  PassiveAggressiveClassifier, RidgeClassifier, RidgeClassifierCV, SGDClassifier)
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, RandomForestClassifier, IsolationForest,
                              StackingClassifier, VotingClassifier, HistGradientBoostingClassifier,
                              RandomTreesEmbedding)
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB, ComplementNB, CategoricalNB
from sklearn.neural_network import MLPClassifier, BernoulliRBM
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, NearestCentroid
from sklearn.svm import LinearSVC, NuSVC, OneClassSVM
from sklearn.linear_model import SGDOneClassSVM
from sklearn.utils import shuffle

# Allow testing of additional optional dependencies
__optionals__ = []
try:
    from xgboost import XGBClassifier, XGBRFClassifier
    __optionals__.extend(['XGBClassifier', 'XGBRFClassifier'])
except:
    pass
try:
    from lightgbm import LGBMClassifier
    __optionals__.append('LGBMClassifier')
except:
    pass
try:
    from catboost import CatBoostClassifier, CatBoost, Pool
    __optionals__.extend(['CatBoostClassifier', 'CatBoost'])
except:
    pass

from src import ml2json


class TestAPI(unittest.TestCase):

    def setUp(self):
        # Python's global `random` (used below for the sparse feature-hasher
        # data) isn't reseeded per test, so its state - and thus this data -
        # depends on how many other tests already drew from it this session.
        # Seed explicitly so this test's data is reproducible regardless of
        # run order (e.g. NuSVC's libsvm solver can hit a numerically
        # infeasible fit on an unlucky draw otherwise).
        random.seed(0)
        self.X, self.y = make_classification(n_samples=50, n_features=3, n_classes=3, n_informative=3, n_redundant=0, random_state=0, shuffle=False)

        feature_hasher = FeatureHasher(n_features=3)
        features = []
        for i in range(0, 100):
            features.append({'a': random.randint(0, 2), 'b': random.randint(3, 5), 'c': random.randint(6, 8)})
        self.y_sparse = [random.randint(0, 2) for _ in range(0, 100)]
        self.X_sparse = feature_hasher.transform(features)
        self.y_multitask = np.vstack((shuffle(self.y, random_state=1), shuffle(self.y, random_state=2))).T
        self.y_multitask_sparse = sp.sparse.csr_matrix(self.y_multitask)

    def check_model(self, model, model_name, abs=False):
        # Given
        if abs:
            model.fit(np.absolute(self.X), self.y)
        else:
            model.fit(self.X, self.y)

        # When
        serialized_model = ml2json.to_dict(model)
        deserialized_model = ml2json.from_dict(serialized_model)

        # Then
        expected_predictions = model.predict(self.X)
        actual_predictions = deserialized_model.predict(self.X)

        np.testing.assert_array_equal(expected_predictions, actual_predictions)

        # When
        ml2json.to_json(model, model_name)
        deserialized_model = ml2json.from_json(model_name)
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
        serialized_model = ml2json.to_dict(model)
        deserialized_model = ml2json.from_dict(serialized_model)

        # Then
        expected_predictions = model.predict(self.X)
        actual_predictions = deserialized_model.predict(self.X)

        np.testing.assert_array_equal(expected_predictions, actual_predictions)

        # JSON
        ml2json.to_json(model, model_name)
        deserialized_model = ml2json.from_json(model_name)
        os.remove(model_name)

        # Then
        actual_predictions = deserialized_model.predict(self.X)

        np.testing.assert_array_equal(expected_predictions, actual_predictions)

    def check_multitask_model(self, model, model_name, abs=False):
        # Given
        if abs:
            model.fit(np.absolute(self.X), self.y_multitask)
        else:
            model.fit(self.X, self.y_multitask)

        # When
        serialized_model = ml2json.to_dict(model)
        deserialized_model = ml2json.from_dict(serialized_model)

        # Then
        expected_predictions = model.predict(self.X)
        actual_predictions = deserialized_model.predict(self.X)

        np.testing.assert_array_equal(expected_predictions, actual_predictions)

        # When
        ml2json.to_json(model, model_name)
        deserialized_model = ml2json.from_json(model_name)
        os.remove(model_name)

        # JSON
        actual_predictions = deserialized_model.predict(self.X)

        np.testing.assert_array_equal(expected_predictions, actual_predictions)

    def check_multitask_sparse_model(self, model, model_name, abs=False):
        # Given
        if abs:
            model.fit(np.absolute(self.X_sparse), self.y_multitask_sparse)
        else:
            model.fit(self.X_sparse, self.y_multitask_sparse)

        # When
        serialized_model = ml2json.to_dict(model)
        deserialized_model = ml2json.from_dict(serialized_model)

        # Then
        expected_predictions = model.predict(self.X)
        actual_predictions = deserialized_model.predict(self.X)

        np.testing.assert_array_equal(expected_predictions, actual_predictions)

        # JSON
        ml2json.to_json(model, model_name)
        deserialized_model = ml2json.from_json(model_name)
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
        self.check_multitask_model(RandomForestClassifier(n_estimators=10, max_depth=5, random_state=0), 'rf.json')

    def test_hist_gradient_boosting(self):
        # HistGradientBoostingClassifier holds its fitted state in compiled/binned
        # objects (a list of TreePredictor per boosting iteration/class, plus a
        # _BinMapper) that don't round-trip through a plain __dict__ walk without
        # dedicated handling - check_model's predict()-only comparison is not
        # exact enough on its own here, so predict_proba() (which depends on the
        # raw per-tree leaf values, not just the argmax) is compared too.
        model = HistGradientBoostingClassifier(max_iter=25, max_depth=3, random_state=0)
        model.fit(self.X, self.y)
        deserialized_model = ml2json.from_dict(ml2json.to_dict(model))
        np.testing.assert_array_equal(model.predict(self.X), deserialized_model.predict(self.X))
        np.testing.assert_array_equal(model.predict_proba(self.X), deserialized_model.predict_proba(self.X))
        self.assertEqual(model.n_iter_, deserialized_model.n_iter_)

        ml2json.to_json(model, 'hist-gb.json')
        deserialized_model = ml2json.from_json('hist-gb.json')
        os.remove('hist-gb.json')
        np.testing.assert_array_equal(model.predict(self.X), deserialized_model.predict(self.X))
        np.testing.assert_array_equal(model.predict_proba(self.X), deserialized_model.predict_proba(self.X))

    def test_hist_gradient_boosting_early_stopping(self):
        # Exercises train_score_/validation_score_/do_early_stopping_/_use_validation_data,
        # which are only populated (and only affect n_iter_/predictions) when
        # early stopping is enabled.
        X, y = make_classification(n_samples=150, n_features=6, n_informative=4, n_classes=3,
                                   n_clusters_per_class=1, random_state=1)
        model = HistGradientBoostingClassifier(max_iter=30, max_depth=4, random_state=1, early_stopping=True,
                                               validation_fraction=0.2, n_iter_no_change=3)
        model.fit(X, y)
        deserialized_model = ml2json.from_dict(ml2json.to_dict(model))
        np.testing.assert_array_equal(model.predict(X), deserialized_model.predict(X))
        np.testing.assert_array_equal(model.predict_proba(X), deserialized_model.predict_proba(X))
        np.testing.assert_array_equal(model.train_score_, deserialized_model.train_score_)
        np.testing.assert_array_equal(model.validation_score_, deserialized_model.validation_score_)

    def test_hist_gradient_boosting_categorical(self):
        # categorical_features makes the TreePredictor's binned_left_cat_bitsets/
        # raw_left_cat_bitsets non-empty (they're (0, 8)-shaped, and easy to get
        # silently right for the wrong reason, otherwise) and builds an internal
        # ColumnTransformer/FunctionTransformer preprocessor holding a bare numpy
        # dtype and a functools.partial - both exercised only in this scenario.
        rng = np.random.RandomState(2)
        X = rng.rand(120, 4)
        X[:, 0] = rng.randint(0, 5, size=120)
        y = (X[:, 1] + X[:, 0] / 5 > 0.7).astype(int)
        model = HistGradientBoostingClassifier(max_iter=15, max_depth=4, random_state=2, categorical_features=[0])
        model.fit(X, y)
        self.assertTrue(any(p[0].raw_left_cat_bitsets.size > 0 for p in model._predictors))
        deserialized_model = ml2json.from_dict(ml2json.to_dict(model))
        np.testing.assert_array_equal(model.predict(X), deserialized_model.predict(X))
        np.testing.assert_array_equal(model.predict_proba(X), deserialized_model.predict_proba(X))

    def test_perceptron(self):
        self.check_model(Perceptron(), 'perceptron.json')
        self.check_sparse_model(Perceptron(), 'perceptron.json')

    def test_mlp(self):
        self.check_model(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1), 'mlp.json')
        self.check_sparse_model(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1), 'mlp.json')

    def test_xgboost_classifier(self):
        if 'XGBClassifier' in __optionals__:
            self.check_model(XGBClassifier(), 'xgb_classifier.json')

    def test_xgboost_rf_classifier(self):
        if 'XGBRFClassifier' in __optionals__:
            self.check_model(XGBRFClassifier(), 'xgb_rf_classifier.json')

    def test_lightgbm_classifier(self):
        if 'LGBMClassifier' in __optionals__:
            self.check_model(LGBMClassifier(), 'lightgbm_classifier.json')

    def check_catboost_model(self, model, model_name, abs=False):
        # Given
        if abs:
            model.fit(np.absolute(self.X), self.y)
        else:
            model.fit(self.X, self.y)

        pool = Pool(data=self.X, label=self.y, feature_names=list(range(self.X.shape[1])))

        # When
        serialized_model = ml2json.to_dict(model, pool)
        deserialized_model = ml2json.from_dict(serialized_model)

        # Then
        expected_predictions = model.predict(self.X)
        actual_predictions = deserialized_model.predict(self.X)

        # almost_equal: raw (non-classifier) CatBoost.predict() returns
        # continuous floats, and the JSON round-trip of the saved model text
        # introduces float-formatting noise at ~1e-16, same as the regression/
        # ranker CatBoost helpers in test_regression.py.
        np.testing.assert_array_almost_equal(expected_predictions, actual_predictions)

        # JSON
        ml2json.to_json(model, model_name)
        deserialized_model = ml2json.from_json(model_name)
        os.remove(model_name)
        json_predictions = deserialized_model.predict(self.X)

        np.testing.assert_array_almost_equal(expected_predictions, json_predictions)

    def test_catboost_classifier(self):
        if 'CatBoostClassifier' in __optionals__:
            self.check_model(CatBoostClassifier(allow_writing_files=False, verbose=False), 'catboost-cls.json')

    def test_catboost(self):
        # catboost.CatBoost is the library's generic base estimator (used
        # directly for custom loss/objective combos not covered by
        # CatBoostClassifier/CatBoostRegressor/CatBoostRanker). Exercised here
        # with a classification-style loss to mirror test_catboost_classifier.
        if 'CatBoost' in __optionals__:
            model = CatBoost(params={'loss_function': 'MultiClass', 'allow_writing_files': False, 'verbose': False})
            self.check_catboost_model(model, 'catboost.json')

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
        serialized_model = ml2json.to_dict(model)
        deserialized_model = ml2json.from_dict(serialized_model)

        # Then
        expected_predictions = model.transform(self.X).toarray()
        actual_predictions = deserialized_model.transform(self.X).toarray()

        np.testing.assert_array_equal(expected_predictions, actual_predictions)

        # When
        ml2json.to_json(model, model_name)
        deserialized_model = ml2json.from_json(model_name)
        os.remove(model_name)

        # JSON
        actual_predictions = deserialized_model.transform(self.X).toarray()

        np.testing.assert_array_equal(expected_predictions, actual_predictions)

    def test_random_trees_embedding(self):
        self.check_random_trees_embedding_model(RandomTreesEmbedding(n_estimators=100, random_state=1234), 'random-trees-embedding.json')

    def check_nearest_neighbour_model(self, model, model_name, multitask: bool = False):
        model.fit(self.X, self.y if not multitask else self.y_multitask)

        # When
        serialized_model = ml2json.to_dict(model)
        deserialized_model = ml2json.from_dict(serialized_model)

        # Then
        expected_predictions = model.predict(self.X)
        expected_neigh_dist, expected_neigh_ind  = model.kneighbors(self.X)
        actual_predictions = deserialized_model.predict(self.X)
        actual_neigh_dist, actual_neigh_ind = deserialized_model.kneighbors(self.X)

        np.testing.assert_array_equal(expected_predictions, actual_predictions)
        np.testing.assert_array_equal(expected_neigh_dist, actual_neigh_dist)
        np.testing.assert_array_equal(expected_neigh_ind, actual_neigh_ind)

        # When
        ml2json.to_json(model, model_name)
        deserialized_model = ml2json.from_json(model_name)
        os.remove(model_name)

        # JSON
        actual_predictions = deserialized_model.predict(self.X)
        actual_neigh_dist, actual_neigh_ind = deserialized_model.kneighbors(self.X)

        np.testing.assert_array_equal(expected_predictions, actual_predictions)
        np.testing.assert_array_equal(expected_neigh_dist, actual_neigh_dist)
        np.testing.assert_array_equal(expected_neigh_ind, actual_neigh_ind)

    def test_nearest_neighbour_classifier(self):
        self.check_nearest_neighbour_model(KNeighborsClassifier(), 'knn-classifier.json', multitask=False)
        self.check_nearest_neighbour_model(KNeighborsClassifier(), 'knn-classifier.json', multitask=True)

    def test_stacking_classifier(self):
        estimators = [
            ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
            ('nb', GaussianNB()),
            ('knn', KNeighborsClassifier())
        ]
        model = StackingClassifier(
            estimators=estimators, final_estimator=LogisticRegression()
        )
        self.check_model(model, 'stacking-classifier.json')

    def test_voting_classifier(self):
        estimators = [
            ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
            ('nb', GaussianNB()),
            ('knn', KNeighborsClassifier())
        ]
        model = VotingClassifier(estimators=estimators, voting='soft')
        self.check_model(model, 'voting-classifier.json')

    def test_categorical_nb(self):
        X_cat = np.random.randint(0, 3, size=self.X.shape)
        model = CategoricalNB()
        model.fit(X_cat, self.y)
        expected_predictions = model.predict(X_cat)

        serialized_model = ml2json.to_dict(model)
        deserialized_model = ml2json.from_dict(serialized_model)

        actual_predictions = deserialized_model.predict(X_cat)
        np.testing.assert_array_equal(expected_predictions, actual_predictions)

        model_name = 'categorical-nb.json'
        ml2json.to_json(model, model_name)
        deserialized_model = ml2json.from_json(model_name)
        os.remove(model_name)

        actual_predictions = deserialized_model.predict(X_cat)
        np.testing.assert_array_equal(expected_predictions, actual_predictions)

    def test_linear_svc(self):
        self.check_model(LinearSVC(random_state=0, max_iter=5000), 'linear-svc.json')
        self.check_sparse_model(LinearSVC(random_state=0, max_iter=5000), 'linear-svc.json')

    def test_nu_svc(self):
        self.check_model(NuSVC(random_state=0), 'nu-svc.json')
        self.check_sparse_model(NuSVC(random_state=0), 'nu-svc.json')

    def check_outlier_model(self, model, model_name):
        model.fit(self.X)
        expected_predictions = model.predict(self.X)

        serialized_model = ml2json.to_dict(model)
        deserialized_model = ml2json.from_dict(serialized_model)

        actual_predictions = deserialized_model.predict(self.X)
        np.testing.assert_array_equal(expected_predictions, actual_predictions)

        ml2json.to_json(model, model_name)
        deserialized_model = ml2json.from_json(model_name)
        os.remove(model_name)

        actual_predictions = deserialized_model.predict(self.X)
        np.testing.assert_array_equal(expected_predictions, actual_predictions)

    def test_one_class_svm(self):
        self.check_outlier_model(OneClassSVM(), 'one-class-svm.json')

    def test_sgd_one_class_svm(self):
        self.check_outlier_model(SGDOneClassSVM(random_state=0), 'sgd-one-class-svm.json')

    def test_passive_aggressive_classifier(self):
        self.check_model(PassiveAggressiveClassifier(random_state=0), 'passive-aggressive-classifier.json')
        self.check_sparse_model(PassiveAggressiveClassifier(random_state=0), 'passive-aggressive-classifier.json')

    def test_ridge_classifier(self):
        self.check_model(RidgeClassifier(), 'ridge-classifier.json')
        self.check_sparse_model(RidgeClassifier(), 'ridge-classifier.json')

    def test_ridge_classifier_cv(self):
        self.check_model(RidgeClassifierCV(), 'ridge-classifier-cv.json')
        self.check_sparse_model(RidgeClassifierCV(), 'ridge-classifier-cv.json')

    def test_sgd_classifier(self):
        self.check_model(SGDClassifier(random_state=0), 'sgd-classifier.json')
        self.check_sparse_model(SGDClassifier(random_state=0), 'sgd-classifier.json')

    def test_logistic_regression_cv(self):
        self.check_model(LogisticRegressionCV(cv=3), 'logistic-regression-cv.json')
        self.check_sparse_model(LogisticRegressionCV(cv=3), 'logistic-regression-cv.json')

    def test_radius_neighbors_classifier(self):
        # A large radius avoids "no neighbors found" ValueErrors that are
        # unrelated to (de)serialization - check_sparse_model fits on
        # differently-scaled hashed features but predicts on self.X.
        self.check_model(RadiusNeighborsClassifier(radius=1e6), 'radius-neighbors-classifier.json')
        self.check_sparse_model(RadiusNeighborsClassifier(radius=1e6), 'radius-neighbors-classifier.json')

    def test_nearest_centroid(self):
        self.check_model(NearestCentroid(), 'nearest-centroid.json')
        self.check_sparse_model(NearestCentroid(), 'nearest-centroid.json')

    def test_bernoulli_rbm(self):
        model = BernoulliRBM(n_components=5, random_state=0)
        model.fit(np.absolute(self.X))
        expected_t = model.transform(np.absolute(self.X))

        serialized_model = ml2json.to_dict(model)
        deserialized_model = ml2json.from_dict(serialized_model)

        actual_t = deserialized_model.transform(np.absolute(self.X))
        np.testing.assert_array_almost_equal(expected_t, actual_t)

        model_name = 'bernoulli-rbm.json'
        ml2json.to_json(model, model_name)
        deserialized_model = ml2json.from_json(model_name)
        os.remove(model_name)

        actual_t = deserialized_model.transform(np.absolute(self.X))
        np.testing.assert_array_almost_equal(expected_t, actual_t)
