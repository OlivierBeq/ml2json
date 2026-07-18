# -*- coding: utf-8 -*-

import os
import random
import unittest

import numpy as np
from sklearn.datasets import make_regression
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import (LinearRegression, Lasso, Ridge, ElasticNet, ARDRegression, BayesianRidge,
                                  ElasticNetCV, LassoCV, MultiTaskElasticNet, MultiTaskElasticNetCV, MultiTaskLasso,
                                  MultiTaskLassoCV, GammaRegressor, PoissonRegressor, TweedieRegressor,
                                  HuberRegressor, Lars, LarsCV, LassoLars, LassoLarsCV, LassoLarsIC,
                                  OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV,
                                  PassiveAggressiveRegressor, QuantileRegressor, RANSACRegressor, RidgeCV,
                                  SGDRegressor, TheilSenRegressor)
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import (AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor, RandomForestRegressor,
                              StackingRegressor, VotingRegressor, HistGradientBoostingRegressor)
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, LinearSVR, NuSVR
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor

# Allow testing of additional optional dependencies
__optionals__ = []
try:
    from xgboost import XGBRegressor, XGBRFRegressor, XGBRanker
    __optionals__.extend(['XGBRegressor', 'XGBRFRegressor', 'XGBRanker'])
except:
    pass
try:
    from lightgbm import LGBMRegressor, LGBMRanker
    __optionals__.extend(['LGBMRegressor', 'LGBMRanker'])
except:
    pass
try:
    from catboost import CatBoostRegressor, CatBoostRanker, Pool
    __optionals__.extend(['CatBoostRegressor', 'CatBoostRanker'])
except:
    pass

from src import ml2json


class TestAPI(unittest.TestCase):

    def setUp(self):
        self.X, self.y = make_regression(n_samples=50, n_features=3, n_informative=3, random_state=0, shuffle=False)
        self.y_rank = np.argsort(np.argsort(self.y)).tolist()

        feature_hasher = FeatureHasher(n_features=3)
        features = []
        for i in range(0, 100):
            features.append({'a': random.randint(0, 2), 'b': random.randint(3, 5), 'c': random.randint(6, 8)})
        self.y_sparse = [random.random() for i in range(0, 100)]
        self.X_sparse = feature_hasher.transform(features)

        self.y_pos = np.abs(self.y) + 0.1
        self.y_multitask = np.vstack((self.y, self.y[::-1])).T

    def check_model(self, model, model_name):
        # Given
        model.fit(self.X, self.y)

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

    def check_sparse_model(self, model, model_name):
        # Given
        model.fit(self.X_sparse, self.y_sparse)

        # When
        serialized_model = ml2json.to_dict(model)
        deserialized_model = ml2json.from_dict(serialized_model)

        # Then
        expected_predictions = model.predict(self.X_sparse)
        actual_predictions = deserialized_model.predict(self.X_sparse)

        np.testing.assert_array_equal(expected_predictions, actual_predictions)

        # JSON
        ml2json.to_json(model, model_name)
        deserialized_model = ml2json.from_json(model_name)
        os.remove(model_name)

        # Then
        expected_predictions = model.predict(self.X_sparse)
        actual_predictions = deserialized_model.predict(self.X_sparse)

        np.testing.assert_array_equal(expected_predictions, actual_predictions)

    def test_linear_regression(self):
        self.check_model(LinearRegression(), 'linear-regression.json')
        self.check_sparse_model(LinearRegression(), 'linear-regression.json')

    def test_lasso_regression(self):
        self.check_model(Lasso(alpha=0.1), 'lasso-regression.json')
        self.check_sparse_model(Lasso(alpha=0.1), 'lasso-regression.json')

    def test_elasticnet_regression(self):
        self.check_model(ElasticNet(alpha=0.1), 'elaticnet.json')
        self.check_sparse_model(ElasticNet(alpha=0.1), 'elasticnet.json')

    def test_ridge_regression(self):
        self.check_model(Ridge(alpha=0.5), 'ridge-regression.json')
        self.check_sparse_model(Ridge(alpha=0.5), 'ridge-regression.json')

    def test_svr(self):
        self.check_model(SVR(gamma='scale', C=1.0, epsilon=0.2), 'SVR.json')
        self.check_sparse_model(SVR(gamma='scale', C=1.0, epsilon=0.2), 'SVR.json')

    def test_decision_tree_regression(self):
        self.check_model(DecisionTreeRegressor(), 'decision-tree.json')
        self.check_sparse_model(DecisionTreeRegressor(), 'decision-tree.json')

    def test_extra_tree_regression(self):
        self.check_model(ExtraTreeRegressor(), 'extra-tree-reg.json')
        self.check_sparse_model(ExtraTreeRegressor(), 'extra-tree-reg.json')

    def test_gradient_boosting_regression(self):
        self.check_model(GradientBoostingRegressor(), 'gradientboosting-regressor.json')
        self.check_sparse_model(GradientBoostingRegressor(), 'gradientboosting-regressor.json')

    def test_random_forest_regression(self):
        self.check_model(RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100), 'rf-regressor.json')
        self.check_sparse_model(RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100), 'rf-regressor.json')

    def test_mlp_regression(self):
        self.check_model(MLPRegressor(max_iter=10000), 'MLP-regressor.json')
        self.check_sparse_model(MLPRegressor(max_iter=10000), 'MLP-regressor.json')

    def check_ranking_model(self, model, model_name):
        # Given
        model.fit(self.X, self.y_rank, group=[10, len(self.y) - 10])

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

    def test_xgboost_ranker(self):
        if 'XGBRanker' in __optionals__:
            self.check_ranking_model(XGBRanker(objective='rank:pairwise'), 'XGB-ranker.json')

    def test_xgboost_regressor(self):
        if 'XGBRegressor' in __optionals__:
            self.check_model(XGBRegressor(), 'XGB-regressor.json')

    def test_xgboost_rf_regressor(self):
        if 'XGBRFRegressor' in __optionals__:
            self.check_model(XGBRFRegressor(), 'XGB-RF-regressor.json')

    def test_lightgbm_regressor(self):
        if 'LGBMRegressor' in __optionals__:
            self.check_model(LGBMRegressor(), 'lightgbm-regressor.json')

    def test_lightgbm_ranker(self):
        if 'LGBMRanker' in __optionals__:
            self.check_ranking_model(LGBMRanker(label_gain=[i for i in range(self.X.shape[0] + 1)]), 'lightgbm-ranker.json')

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

        np.testing.assert_array_almost_equal(expected_predictions, actual_predictions)

        # JSON
        ml2json.to_json(model, model_name)
        deserialized_model = ml2json.from_json(model_name)
        os.remove(model_name)

        # Then
        actual_predictions = deserialized_model.predict(self.X)

        np.testing.assert_array_almost_equal(expected_predictions, actual_predictions)

    def check_catboost_ranking_model(self, model, model_name, abs=False):
        # Given
        if abs:
            model.fit(np.absolute(self.X), self.y)
        else:
            model.fit(self.X, self.y, group_id=[0] * 10 + [1] * (len(self.y) - 10))

        pool = Pool(data=self.X, label=self.y, feature_names=list(range(self.X.shape[1])))

        # When
        serialized_model = ml2json.to_dict(model, pool)
        deserialized_model = ml2json.from_dict(serialized_model)

        # Then
        expected_predictions = model.predict(self.X)
        actual_predictions = deserialized_model.predict(self.X)

        np.testing.assert_array_almost_equal(expected_predictions, actual_predictions)

        # JSON
        ml2json.to_json(model, model_name)
        deserialized_model = ml2json.from_json(model_name)
        os.remove(model_name)

        # Then
        actual_predictions = deserialized_model.predict(self.X)

        np.testing.assert_array_almost_equal(expected_predictions, actual_predictions)

    def test_catboost_regressor(self):
        if 'CatBoostRegressor' in __optionals__:
            self.check_catboost_model(CatBoostRegressor(allow_writing_files=False, verbose=False), 'catboost-regressor.json')

    def test_catboost_ranker(self):
        if 'CatBoostRanker' in __optionals__:
            self.check_catboost_ranking_model(CatBoostRanker(allow_writing_files=False, verbose=False), 'catboost-ranker.json')

    def test_adaboost_regressor(self):
        self.check_model(AdaBoostRegressor(n_estimators=25, learning_rate=1.0), 'adaboost-reg.json')
        self.check_sparse_model(AdaBoostRegressor(n_estimators=25, learning_rate=1.0), 'adaboost-reg.json')

    def test_bagging_regressor(self):
        self.check_model(BaggingRegressor(n_estimators=25), 'bagging-reg.json')
        self.check_sparse_model(BaggingRegressor(n_estimators=25), 'bagging-reg.json')

    def test_extratrees_regressor(self):
        self.check_model(ExtraTreesRegressor(n_estimators=25), 'extratrees-reg.json')
        self.check_sparse_model(ExtraTreesRegressor(n_estimators=25), 'extratrees-reg.json')
        self.check_model(ExtraTreesRegressor(n_estimators=25, oob_score=True, bootstrap=True), 'extratrees-reg.json')
        self.check_sparse_model(ExtraTreesRegressor(n_estimators=25, oob_score=True, bootstrap=True), 'extratrees-reg.json')

    def check_nearest_neighbour_model(self, model, model_name):
        model.fit(self.X, self.y)

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

    def test_nearest_neighbour_regressor(self):
        self.check_nearest_neighbour_model(KNeighborsRegressor(), 'knn-regressor.json')

    def test_stacking_regressor(self):
        estimators = [
            ('rf', RandomForestRegressor(n_estimators=10, random_state=42)),
            ('ridge', Ridge(random_state=42)),
            ('knn', KNeighborsRegressor()),
            ('svm', SVR())
        ]
        model = StackingRegressor(
            estimators=estimators, final_estimator=LinearRegression()
        )
        self.check_model(model, 'stacking-regressor.json')

    def test_voting_regressor(self):
        estimators = [
            ('rf', RandomForestRegressor(n_estimators=10, random_state=42)),
            ('ridge', Ridge(random_state=42)),
            ('knn', KNeighborsRegressor()),
            ('svm', SVR())
        ]
        model = VotingRegressor(estimators=estimators)
        self.check_model(model, 'stacking-regressor.json')

    def check_multitask_model(self, model, model_name):
        model.fit(self.X, self.y_multitask)
        expected_predictions = model.predict(self.X)

        serialized_model = ml2json.to_dict(model)
        deserialized_model = ml2json.from_dict(serialized_model)

        actual_predictions = deserialized_model.predict(self.X)
        np.testing.assert_array_almost_equal(expected_predictions, actual_predictions)

        ml2json.to_json(model, model_name)
        deserialized_model = ml2json.from_json(model_name)
        os.remove(model_name)

        actual_predictions = deserialized_model.predict(self.X)
        np.testing.assert_array_almost_equal(expected_predictions, actual_predictions)

    def check_positive_model(self, model, model_name):
        model.fit(self.X, self.y_pos)
        expected_predictions = model.predict(self.X)

        serialized_model = ml2json.to_dict(model)
        deserialized_model = ml2json.from_dict(serialized_model)

        actual_predictions = deserialized_model.predict(self.X)
        np.testing.assert_array_almost_equal(expected_predictions, actual_predictions)

        ml2json.to_json(model, model_name)
        deserialized_model = ml2json.from_json(model_name)
        os.remove(model_name)

        actual_predictions = deserialized_model.predict(self.X)
        np.testing.assert_array_almost_equal(expected_predictions, actual_predictions)

    def test_ard_regression(self):
        self.check_model(ARDRegression(), 'ard-regression.json')

    def test_bayesian_ridge(self):
        self.check_model(BayesianRidge(), 'bayesian-ridge.json')

    def test_elasticnet_cv(self):
        self.check_model(ElasticNetCV(cv=3), 'elasticnet-cv.json')

    def test_lasso_cv(self):
        self.check_model(LassoCV(cv=3), 'lasso-cv.json')

    def test_multitask_elasticnet(self):
        self.check_multitask_model(MultiTaskElasticNet(), 'multitask-elasticnet.json')

    def test_multitask_elasticnet_cv(self):
        self.check_multitask_model(MultiTaskElasticNetCV(cv=3), 'multitask-elasticnet-cv.json')

    def test_multitask_lasso(self):
        self.check_multitask_model(MultiTaskLasso(), 'multitask-lasso.json')

    def test_multitask_lasso_cv(self):
        self.check_multitask_model(MultiTaskLassoCV(cv=3), 'multitask-lasso-cv.json')

    def test_gamma_regressor(self):
        self.check_positive_model(GammaRegressor(), 'gamma-regressor.json')

    def test_poisson_regressor(self):
        self.check_positive_model(PoissonRegressor(), 'poisson-regressor.json')

    def test_tweedie_regressor(self):
        self.check_positive_model(TweedieRegressor(), 'tweedie-regressor.json')

    def test_huber_regressor(self):
        self.check_model(HuberRegressor(), 'huber-regressor.json')

    def test_lars(self):
        self.check_model(Lars(), 'lars.json')

    def test_lars_cv(self):
        self.check_model(LarsCV(cv=3), 'lars-cv.json')

    def test_lasso_lars(self):
        self.check_model(LassoLars(), 'lasso-lars.json')

    def test_lasso_lars_cv(self):
        self.check_model(LassoLarsCV(cv=3), 'lasso-lars-cv.json')

    def test_lasso_lars_ic(self):
        self.check_model(LassoLarsIC(), 'lasso-lars-ic.json')

    def test_orthogonal_matching_pursuit(self):
        self.check_model(OrthogonalMatchingPursuit(), 'orthogonal-matching-pursuit.json')

    def test_orthogonal_matching_pursuit_cv(self):
        self.check_model(OrthogonalMatchingPursuitCV(cv=3), 'orthogonal-matching-pursuit-cv.json')

    def test_passive_aggressive_regressor(self):
        self.check_model(PassiveAggressiveRegressor(random_state=0), 'passive-aggressive-regressor.json')

    def test_quantile_regressor(self):
        self.check_model(QuantileRegressor(), 'quantile-regressor.json')

    def test_ransac_regressor(self):
        self.check_model(RANSACRegressor(random_state=0), 'ransac-regressor.json')

    def test_ridge_cv(self):
        self.check_model(RidgeCV(), 'ridge-cv.json')

    def test_sgd_regressor(self):
        self.check_model(SGDRegressor(random_state=0), 'sgd-regressor.json')

    def test_theilsen_regressor(self):
        self.check_model(TheilSenRegressor(random_state=0), 'theilsen-regressor.json')

    def test_linear_svr(self):
        self.check_model(LinearSVR(random_state=0, max_iter=5000), 'linear-svr.json')
        self.check_sparse_model(LinearSVR(random_state=0, max_iter=5000), 'linear-svr.json')

    def test_nu_svr(self):
        self.check_model(NuSVR(), 'nu-svr.json')
        self.check_sparse_model(NuSVR(), 'nu-svr.json')

    def test_radius_neighbors_regressor(self):
        self.check_model(RadiusNeighborsRegressor(radius=1e6), 'radius-neighbors-regressor.json')
