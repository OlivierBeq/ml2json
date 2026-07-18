# -*- coding: utf-8 -*-

"""Tests for the native, non-sklearn-estimator boosting-library objects:
xgboost.Booster, lightgbm.Booster, lightgbm.Dataset and catboost.Pool.
"""

import os
import unittest

import numpy as np

# Allow testing of additional optional dependencies
__optionals__ = []
try:
    import xgboost as xgb
    __optionals__.append('xgboost')
except:
    pass
try:
    import lightgbm as lgb
    __optionals__.append('lightgbm')
except:
    pass
try:
    import catboost
    __optionals__.append('catboost')
except:
    pass

from src import ml2json


class TestBoosting(unittest.TestCase):

    def setUp(self):
        rng = np.random.RandomState(0)
        self.X = rng.rand(60, 4)
        self.y = rng.rand(60)
        self.w = rng.rand(60)

    def test_xgboost_booster(self):
        if 'xgboost' not in __optionals__:
            return
        dtrain = xgb.DMatrix(self.X, label=self.y)
        booster = xgb.train({'objective': 'reg:squarederror'}, dtrain, num_boost_round=10)
        expected = booster.predict(dtrain)

        model_dict = ml2json.to_dict(booster)
        self.assertEqual(model_dict['meta'], 'xgboost.booster')
        deserialized = ml2json.from_dict(model_dict)
        actual = deserialized.predict(xgb.DMatrix(self.X))
        np.testing.assert_allclose(expected, actual)

        model_json = 'xgboost-booster.json'
        ml2json.to_json(booster, model_json)
        deserialized = ml2json.from_json(model_json)
        os.remove(model_json)
        actual = deserialized.predict(xgb.DMatrix(self.X))
        np.testing.assert_allclose(expected, actual)

    def test_lightgbm_booster(self):
        if 'lightgbm' not in __optionals__:
            return
        train_set = lgb.Dataset(self.X, label=self.y)
        booster = lgb.train({'objective': 'regression', 'verbosity': -1}, train_set, num_boost_round=10)
        expected = booster.predict(self.X)

        model_dict = ml2json.to_dict(booster)
        self.assertEqual(model_dict['meta'], 'lightgbm.booster')
        deserialized = ml2json.from_dict(model_dict)
        actual = deserialized.predict(self.X)
        np.testing.assert_allclose(expected, actual)

        model_json = 'lightgbm-booster.json'
        ml2json.to_json(booster, model_json)
        deserialized = ml2json.from_json(model_json)
        os.remove(model_json)
        actual = deserialized.predict(self.X)
        np.testing.assert_allclose(expected, actual)

    def test_lightgbm_dataset(self):
        if 'lightgbm' not in __optionals__:
            return
        dataset = lgb.Dataset(self.X, label=self.y, weight=self.w, categorical_feature=[0],
                              feature_name=['a', 'b', 'c', 'd'], free_raw_data=False)

        model_dict = ml2json.to_dict(dataset)
        self.assertEqual(model_dict['meta'], 'lightgbm.dataset')
        deserialized = ml2json.from_dict(model_dict)
        np.testing.assert_allclose(dataset.data, deserialized.data)
        np.testing.assert_allclose(dataset.label, deserialized.label)
        np.testing.assert_allclose(dataset.weight, deserialized.weight)
        self.assertEqual(dataset.categorical_feature, deserialized.categorical_feature)
        self.assertEqual(dataset.feature_name, deserialized.feature_name)

        model_json = 'lightgbm-dataset.json'
        ml2json.to_json(dataset, model_json)
        deserialized = ml2json.from_json(model_json)
        os.remove(model_json)
        np.testing.assert_allclose(dataset.data, deserialized.data)
        np.testing.assert_allclose(dataset.label, deserialized.label)

        # A Booster trained on the reconstructed Dataset must predict the
        # same as one trained on the original - not just matching arrays.
        original_booster = lgb.train({'objective': 'regression', 'verbosity': -1}, dataset, num_boost_round=5)
        rebuilt_booster = lgb.train({'objective': 'regression', 'verbosity': -1},
                                    ml2json.from_dict(ml2json.to_dict(dataset)), num_boost_round=5)
        np.testing.assert_allclose(original_booster.predict(self.X), rebuilt_booster.predict(self.X))

    def test_lightgbm_dataset_freed_raw_data_raises(self):
        """free_raw_data defaults to True; once a Dataset has been constructed
        (e.g. via lgb.train()), lightgbm discards the raw arrays internally
        and there is no public API to recover them - this must fail loudly
        rather than silently produce a broken/empty Dataset."""
        if 'lightgbm' not in __optionals__:
            return
        dataset = lgb.Dataset(self.X, label=self.y)
        dataset.construct()
        with self.assertRaises(ValueError):
            ml2json.to_dict(dataset)

    def test_catboost_pool(self):
        if 'catboost' not in __optionals__:
            return
        pool = catboost.Pool(data=self.X, label=self.y, weight=self.w,
                             feature_names=['a', 'b', 'c', 'd'])

        model_dict = ml2json.to_dict(pool)
        self.assertEqual(model_dict['meta'], 'catboost.pool')
        deserialized = ml2json.from_dict(model_dict)
        np.testing.assert_allclose(pool.get_features(), deserialized.get_features())
        np.testing.assert_allclose(pool.get_label(), deserialized.get_label())
        np.testing.assert_allclose(pool.get_weight(), deserialized.get_weight())
        self.assertEqual(pool.get_feature_names(), deserialized.get_feature_names())

        model_json = 'catboost-pool.json'
        ml2json.to_json(pool, model_json)
        deserialized = ml2json.from_json(model_json)
        os.remove(model_json)
        np.testing.assert_allclose(pool.get_features(), deserialized.get_features())

        # A model trained on the reconstructed Pool must predict the same as
        # one trained on the original.
        original_model = catboost.CatBoostRegressor(iterations=5, allow_writing_files=False, verbose=False)
        original_model.fit(pool)
        rebuilt_model = catboost.CatBoostRegressor(iterations=5, allow_writing_files=False, verbose=False)
        rebuilt_model.fit(ml2json.from_dict(ml2json.to_dict(pool)))
        np.testing.assert_allclose(original_model.predict(self.X), rebuilt_model.predict(self.X))

    def test_catboost_pool_categorical_features_raises(self):
        """catboost.Pool.get_features() only supports pools with purely
        numeric feature columns; there is no public accessor to recover raw
        values for categorical/text/embedding columns from a constructed
        Pool - this must fail loudly rather than silently drop columns."""
        if 'catboost' not in __optionals__:
            return
        rng = np.random.RandomState(0)
        X_obj = np.empty((30, 4), dtype=object)
        X_obj[:, :] = rng.rand(30, 4)
        X_obj[:, 0] = rng.choice(['a', 'b', 'c'], size=30)
        pool = catboost.Pool(data=X_obj, label=self.y[:30], cat_features=[0])
        with self.assertRaises(ValueError):
            ml2json.to_dict(pool)


if __name__ == '__main__':
    unittest.main()
