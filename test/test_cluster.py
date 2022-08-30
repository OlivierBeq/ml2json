# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import (AffinityPropagation, AgglomerativeClustering,
                             Birch, DBSCAN, FeatureAgglomeration, KMeans,
                             MiniBatchKMeans, MeanShift, OPTICS, SpectralClustering,
                             SpectralBiclustering, SpectralCoclustering)

from src import sklearn_json as skljson


class TestAPI(unittest.TestCase):

    def setUp(self):
        centers = [[1, 1], [-1, -1], [1, -1]]
        self.n_centers = len(centers)
        self.X, self.labels = make_blobs(n_samples=3000,
                                         centers=centers,
                                         cluster_std=0.7,
                                         random_state=1234)
        self.simple_X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])

    def check_transform_model(self, model, data):
        model.fit(data)
        expected_t = model.transform(data)

        serialized_dict_model = skljson.to_dict(model)
        deserialized_dict_model = skljson.from_dict(serialized_dict_model)

        skljson.to_json(model, 'model.json')
        deserialized_json_model = skljson.from_json('model.json')
        os.remove('model.json')

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:

            for key in sorted(model.__dict__.keys()):
                if isinstance(model.__dict__[key], np.ndarray):
                    print(key, (model.__dict__[key] == deserialized_dict_model.__dict__[key]).all())
                else:
                    print(key, model.__dict__[key] == deserialized_dict_model.__dict__[key])

            actual_t = deserialized_model.transform(data)

            np.testing.assert_array_almost_equal(expected_t, actual_t)

    def check_fittransform_model(self, model, data):
        expected_ft = model.fit_transform(data)

        serialized_dict_model = skljson.to_dict(model)
        deserialized_dict_model = skljson.from_dict(serialized_dict_model)

        skljson.to_json(model, 'model.json')
        deserialized_json_model = skljson.from_json('model.json')
        os.remove('model.json')

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:

            for key in sorted(model.__dict__.keys()):
                if isinstance(model.__dict__[key], np.ndarray):
                    print(key, (model.__dict__[key] == deserialized_dict_model.__dict__[key]).all())
                else:
                    print(key, model.__dict__[key] == deserialized_dict_model.__dict__[key])

            actual_ft = deserialized_model.fit_transform(data)

            np.testing.assert_array_almost_equal(expected_ft, actual_ft)

    def check_predict_model(self, model, data):
        model.fit(data)
        expected_p = model.predict(data)

        serialized_dict_model = skljson.to_dict(model)
        deserialized_dict_model = skljson.from_dict(serialized_dict_model)

        skljson.to_json(model, 'model.json')
        deserialized_json_model = skljson.from_json('model.json')
        os.remove('model.json')

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:

            for key in sorted(model.__dict__.keys()):
                if isinstance(model.__dict__[key], np.ndarray):
                    print(key, (model.__dict__[key] == deserialized_dict_model.__dict__[key]).all())
                else:
                    print(key, model.__dict__[key] == deserialized_dict_model.__dict__[key])

            actual_fp = deserialized_model.fit_predict(data)
            actual_p = deserialized_model.predict(data)

            np.testing.assert_array_equal(expected_p, actual_p)

    def check_fitpredict_model(self, model, data):
        expected_fp = model.fit_predict(data)

        serialized_dict_model = skljson.to_dict(model)
        deserialized_dict_model = skljson.from_dict(serialized_dict_model)

        skljson.to_json(model, 'model.json')
        deserialized_json_model = skljson.from_json('model.json')
        os.remove('model.json')

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:

            for key in sorted(model.__dict__.keys()):
                if isinstance(model.__dict__[key], np.ndarray):
                    print(key, (model.__dict__[key] == deserialized_dict_model.__dict__[key]).all())
                else:
                    print(key, model.__dict__[key] == deserialized_dict_model.__dict__[key])

            actual_fp = deserialized_model.fit_predict(data)

            np.testing.assert_array_equal(expected_fp, actual_fp)

    def test_kmeans(self):
        for model in [KMeans(n_clusters=self.n_centers, init='k-means++',
                             random_state=1234, n_init=100, max_iter=10000,
                             verbose=0, tol=1e-999),
                      KMeans(n_clusters=self.n_centers, init='random',
                             random_state=1234, n_init=100, max_iter=10000,
                             verbose=0, tol=1e-999)]:
            self.check_transform_model(model, self.X)
            self.check_fittransform_model(model, self.X)
            self.check_predict_model(model, self.X)
            self.check_fitpredict_model(model, self.X)

    def test_minibatch_kmeans(self):
        for model in [MiniBatchKMeans(n_clusters=self.n_centers, init='k-means++',
                             random_state=1234, n_init=100, max_iter=10000,
                             verbose=0, tol=1e-999),
                      MiniBatchKMeans(n_clusters=self.n_centers, init='random',
                             random_state=1234, n_init=100, max_iter=10000,
                             verbose=0, tol=1e-999)]:
            self.check_transform_model(model, self.X)
            self.check_fittransform_model(model, self.X)
            self.check_predict_model(model, self.X)
            self.check_fitpredict_model(model, self.X)

    def test_affinity_propagation(self):
        self.check_predict_model(AffinityPropagation(), self.simple_X)
        self.check_fitpredict_model(AffinityPropagation(), self.simple_X)

    def test_agglomerative_clustering(self):
        self.check_fitpredict_model(AgglomerativeClustering(), self.simple_X)

    def test_dbscan(self):
        self.check_fitpredict_model(DBSCAN(), self.X)

    def test_optics(self):
        self.check_fitpredict_model(OPTICS(), self.simple_X)

    def test_spectral_clustering(self):
        self.check_fitpredict_model(SpectralClustering(random_state=1234, n_clusters=2), self.simple_X)

    def test_feature_agglomeration(self):
        self.check_transform_model(FeatureAgglomeration(pooling_func=np.mean), self.X)
        self.check_fittransform_model(FeatureAgglomeration(pooling_func=np.mean), self.X)
