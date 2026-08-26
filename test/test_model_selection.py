# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np
from scipy.stats import randint, uniform
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import (KFold, StratifiedKFold, GroupKFold, StratifiedGroupKFold, RepeatedKFold,
                                     RepeatedStratifiedKFold, LeaveOneOut, LeavePOut, LeaveOneGroupOut,
                                     LeavePGroupsOut, ShuffleSplit, StratifiedShuffleSplit, GroupShuffleSplit,
                                     TimeSeriesSplit, PredefinedSplit, ParameterGrid, ParameterSampler,
                                     GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV)

from src import ml2json


class TestSplitters(unittest.TestCase):

    def setUp(self):
        self.X, self.y = make_classification(n_samples=40, n_features=5, n_classes=2, random_state=0)
        self.groups = np.array([0, 1, 2, 3] * 10)

    def check_splits(self, splitter, model_name, groups=None):
        expected_splits = list(splitter.split(self.X, self.y, groups=groups))

        serialized_dict_model = ml2json.to_dict(splitter)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        ml2json.to_json(splitter, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_splits = list(deserialized_model.split(self.X, self.y, groups=groups))
            self.assertEqual(len(expected_splits), len(actual_splits))
            for (etr, ete), (atr, ate) in zip(expected_splits, actual_splits):
                np.testing.assert_array_equal(etr, atr)
                np.testing.assert_array_equal(ete, ate)

    def test_kfold(self):
        self.check_splits(KFold(n_splits=4, shuffle=True, random_state=42), 'kfold.json')

    def test_stratified_kfold(self):
        self.check_splits(StratifiedKFold(n_splits=4, shuffle=True, random_state=42), 'stratified-kfold.json')

    def test_group_kfold(self):
        self.check_splits(GroupKFold(n_splits=4), 'group-kfold.json', groups=self.groups)

    def test_stratified_group_kfold(self):
        self.check_splits(StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42),
                          'stratified-group-kfold.json', groups=self.groups)

    def test_repeated_kfold(self):
        self.check_splits(RepeatedKFold(n_splits=4, n_repeats=2, random_state=42), 'repeated-kfold.json')

    def test_repeated_stratified_kfold(self):
        self.check_splits(RepeatedStratifiedKFold(n_splits=4, n_repeats=2, random_state=42),
                          'repeated-stratified-kfold.json')

    def test_leave_one_out(self):
        self.check_splits(LeaveOneOut(), 'leave-one-out.json')

    def test_leave_p_out(self):
        self.check_splits(LeavePOut(p=3), 'leave-p-out.json')

    def test_leave_one_group_out(self):
        self.check_splits(LeaveOneGroupOut(), 'leave-one-group-out.json', groups=self.groups)

    def test_leave_p_groups_out(self):
        self.check_splits(LeavePGroupsOut(n_groups=2), 'leave-p-groups-out.json', groups=self.groups)

    def test_shuffle_split(self):
        self.check_splits(ShuffleSplit(n_splits=4, random_state=42), 'shuffle-split.json')

    def test_stratified_shuffle_split(self):
        self.check_splits(StratifiedShuffleSplit(n_splits=4, random_state=42), 'stratified-shuffle-split.json')

    def test_group_shuffle_split(self):
        self.check_splits(GroupShuffleSplit(n_splits=4, random_state=42), 'group-shuffle-split.json',
                          groups=self.groups)

    def test_time_series_split(self):
        self.check_splits(TimeSeriesSplit(n_splits=4), 'time-series-split.json')

    def test_predefined_split(self):
        self.check_splits(PredefinedSplit(test_fold=self.groups), 'predefined-split.json')


class TestParameterHelpers(unittest.TestCase):

    def test_parameter_grid(self):
        grid = ParameterGrid({'a': [1, 2, 3], 'b': ['x', 'y']})
        expected = list(grid)

        serialized_dict_model = ml2json.to_dict(grid)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        ml2json.to_json(grid, 'parameter-grid.json')
        deserialized_json_model = ml2json.from_json('parameter-grid.json')
        os.remove('parameter-grid.json')

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            self.assertEqual(expected, list(deserialized_model))

    def test_parameter_sampler(self):
        sampler = ParameterSampler({'a': randint(1, 10), 'b': uniform(0, 1)}, n_iter=5, random_state=42)
        expected = list(sampler)

        serialized_dict_model = ml2json.to_dict(sampler)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        ml2json.to_json(sampler, 'parameter-sampler.json')
        deserialized_json_model = ml2json.from_json('parameter-sampler.json')
        os.remove('parameter-sampler.json')

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            np.testing.assert_equal(expected, list(deserialized_model))


class TestSearchCV(unittest.TestCase):

    def setUp(self):
        self.X, self.y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=0)

    def check_model(self, model, model_name):
        model.fit(self.X, self.y)
        expected_predictions = model.predict(self.X)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_predictions = deserialized_model.predict(self.X)
            np.testing.assert_array_equal(expected_predictions, actual_predictions)
            self.assertEqual(model.best_params_, deserialized_model.best_params_)

    def test_grid_search_cv(self):
        self.check_model(GridSearchCV(RandomForestClassifier(random_state=42),
                                       {'n_estimators': [2, 3], 'max_depth': [2, 3]}, cv=3),
                          'grid-search-cv.json')

    def test_grid_search_cv_unfitted(self):
        model = GridSearchCV(RandomForestClassifier(random_state=42), {'n_estimators': [2, 3]}, cv=3)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        self.assertEqual(model.param_grid, deserialized_dict_model.param_grid)
        self.assertEqual(model.estimator.get_params(), deserialized_dict_model.estimator.get_params())

    def test_randomized_search_cv(self):
        self.check_model(RandomizedSearchCV(RandomForestClassifier(random_state=42),
                                            {'n_estimators': randint(2, 10), 'max_depth': [2, 3, None]},
                                            n_iter=3, cv=3, random_state=42),
                          'randomized-search-cv.json')

    def test_halving_grid_search_cv(self):
        self.check_model(HalvingGridSearchCV(RandomForestClassifier(random_state=42),
                                             {'n_estimators': [2, 3], 'max_depth': [2, 3]},
                                             cv=3, random_state=42),
                          'halving-grid-search-cv.json')

    def test_halving_randomized_search_cv(self):
        self.check_model(HalvingRandomSearchCV(RandomForestClassifier(random_state=42),
                                               {'n_estimators': randint(2, 10), 'max_depth': [2, 3, None]},
                                               n_candidates=4, cv=3, random_state=42),
                          'halving-randomized-search-cv.json')


if __name__ == '__main__':
    unittest.main()
