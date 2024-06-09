# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np
from sklearn.datasets import make_classification, fetch_openml, clear_data_home
from sklearn.preprocessing import scale, label_binarize

# Allow testing of additional optional dependencies
__optionals__ = []
try:
        from imblearn.under_sampling import (ClusterCentroids, CondensedNearestNeighbour, EditedNearestNeighbours,
                                             RepeatedEditedNearestNeighbours, AllKNN, InstanceHardnessThreshold,
                                             NearMiss, NeighbourhoodCleaningRule, OneSidedSelection,
                                             RandomUnderSampler, TomekLinks)
        from imblearn.over_sampling import (RandomOverSampler, SMOTE, SMOTEN, SMOTENC, ADASYN, BorderlineSMOTE,
                                            KMeansSMOTE, SVMSMOTE)
        from imblearn.combine import SMOTEENN, SMOTETomek
        from imblearn.ensemble import (EasyEnsembleClassifier, RUSBoostClassifier, BalancedBaggingClassifier,
                                       BalancedRandomForestClassifier)

        __optionals__.extend(['imblearn'])
except:
    pass

from src import ml2json


class TestAPI(unittest.TestCase):

    def setUp(self):
        self.X, self.y =  make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9],
                                              n_informative=3, n_redundant=1, flip_y=0,
                                              n_features=20, n_clusters_per_class=1,
                                              n_samples=1000, random_state=10)
        self.openml_X, self.openml_y = fetch_openml('diabetes', version=1, return_X_y=True)
        self.openml_X = scale(self.openml_X)
        self.openml_y = label_binarize(self.openml_y, classes=['tested_positive', 'tested_negative'])

    def check_fitresample_model(self, model, model_name, X, y, decimal=6, is_text=False):
        expected_ft = model.fit_resample(X, y)

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

            actual_ft = deserialized_model.fit_resample(X, y)

            if not is_text:
                np.testing.assert_array_almost_equal(expected_ft[0], actual_ft[0], decimal=decimal)
                np.testing.assert_array_almost_equal(expected_ft[1], actual_ft[1], decimal=decimal)
            else:
                np.testing.assert_array_equal(expected_ft[0], actual_ft[0])
                np.testing.assert_array_equal(expected_ft[1], actual_ft[1])


    def test_cluster_centroids(self):
        from sklearn.cluster import KMeans, BisectingKMeans, MiniBatchKMeans
        for estimator in [KMeans, BisectingKMeans, MiniBatchKMeans]:
            model = ClusterCentroids(estimator=estimator(random_state=1234), random_state=42)
            self.check_fitresample_model(model, 'cluster_centroids.json',
                                         self.X, self.y)

    def test_condensed_nearest_neighbours(self):
        model = CondensedNearestNeighbour(random_state=495)
        self.check_fitresample_model(model, 'condensed_nearest_neighbours.json',
                                     self.openml_X, self.openml_y)

    def test_edited_nearest_neighbours(self):
        model = EditedNearestNeighbours()
        self.check_fitresample_model(model, 'edited_nearest_neighbours.json',
                                     self.openml_X, self.openml_y)

    def test_repeated_edited_nearest_neighbours(self):
        model = RepeatedEditedNearestNeighbours(n_neighbors=4)
        self.check_fitresample_model(model, 'repeated_nearest_neighbours.json',
                                     self.openml_X, self.openml_y)

    def test_all_knn(self):
        model = AllKNN(n_neighbors=4)
        self.check_fitresample_model(model, 'all_knn.json',
                                     self.openml_X, self.openml_y)

    def test_instance_hardness_threshold(self):
        model = InstanceHardnessThreshold(cv=10, random_state=1234)
        self.check_fitresample_model(model, 'instance_hardness_threshold.json',
                                     self.openml_X, self.openml_y)

    def test_near_miss(self):
        model = NearMiss()
        self.check_fitresample_model(model, 'near_miss.json',
                                     self.openml_X, self.openml_y)

    def test_neighbourhood_cleaning_rule(self):
        model = NeighbourhoodCleaningRule()
        self.check_fitresample_model(model, 'neighbourhood_cleaning_rule.json',
                                     self.openml_X, self.openml_y)

    def test_one_sided_selection(self):
        model = OneSidedSelection(random_state=42)
        self.check_fitresample_model(model, 'one_sided_selection.json',
                                     self.openml_X, self.openml_y)

    def test_random_under_sampler(self):
        model = RandomUnderSampler(random_state=8468546)
        self.check_fitresample_model(model, 'random_under_sampler.json',
                                     self.openml_X, self.openml_y)

    def test_tomek_links(self):
        model = TomekLinks()
        self.check_fitresample_model(model, 'tomek_links.json',
                                     self.openml_X, self.openml_y)

    def test_random_over_sampler(self):
        model = RandomOverSampler(random_state=48961)
        self.check_fitresample_model(model, 'random_over_sampler.json',
                                     self.X, self.y)

    def test_smote(self):
        model = SMOTE(random_state=48961)
        self.check_fitresample_model(model, 'smote.json',
                                     self.X, self.y)

    def test_smotenc(self):
        model = SMOTENC(categorical_features=[18, 19], random_state=48961)
        X = np.copy(self.X)
        X[:, -2:] = np.random.default_rng(1234).integers(0, 4, size=(1000, 2))
        self.check_fitresample_model(model, 'smotenc.json',
                                     X, self.y)

    def test_smoten(self):
        model = SMOTEN(random_state=6387687)
        X = np.array(["A"] * 10 + ["B"] * 20 + ["C"] * 30, dtype=object).reshape(-1, 1)
        y = np.array([0] * 20 + [1] * 40, dtype=np.int32)
        self.check_fitresample_model(model, 'smoten.json', X, y, is_text=True)

    def test_adasyn(self):
        model = SMOTEN(random_state=486547865183 // 4856)
        self.check_fitresample_model(model, 'adasyn.json', self.X, self.y)

    def test_borderline_smote(self):
        model = BorderlineSMOTE(random_state=647189)
        self.check_fitresample_model(model, 'borderline_smote.json', self.X, self.y)

    def test_kmeans_smote(self):
        model = KMeansSMOTE(random_state=4148)
        self.check_fitresample_model(model, 'kmeans_smote.json', self.X, self.y)

    def test_svm_smote(self):
        model = SVMSMOTE(random_state=51655)
        self.check_fitresample_model(model, 'svm_smote.json', self.X, self.y)

    def test_smote_enn(self):
        model = SMOTEENN(random_state=52787)
        self.check_fitresample_model(model, 'smote_enn.json', self.X, self.y)

    def test_smote_tomek(self):
        model = SMOTETomek(random_state=987634)
        self.check_fitresample_model(model, 'smote_tomek.json', self.X, self.y)
