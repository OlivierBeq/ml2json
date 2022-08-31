# -*- coding: utf-8 -*-

import os
import unittest
import itertools

import numpy as np
from sklearn.datasets import make_blobs, make_checkerboard
from sklearn.cluster import (AffinityPropagation, AgglomerativeClustering,
                             Birch, DBSCAN, FeatureAgglomeration, KMeans,
                             MiniBatchKMeans, MeanShift, OPTICS, SpectralClustering,
                             SpectralBiclustering, SpectralCoclustering)
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes

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

    def check_transform_model(self, model, model_name, data):
        model.fit(data)
        expected_t = model.transform(data)

        serialized_dict_model = skljson.to_dict(model)
        deserialized_dict_model = skljson.from_dict(serialized_dict_model)

        skljson.to_json(model, model_name)
        deserialized_json_model = skljson.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:

            for key in sorted(model.__dict__.keys()):
                if isinstance(model.__dict__[key], np.ndarray):
                    print(key, (model.__dict__[key] == deserialized_dict_model.__dict__[key]).all())
                else:
                    print(key, model.__dict__[key] == deserialized_dict_model.__dict__[key])

            actual_t = deserialized_model.transform(data)

            np.testing.assert_array_almost_equal(expected_t, actual_t)

    def check_fittransform_model(self, model, model_name, data):
        expected_ft = model.fit_transform(data)

        serialized_dict_model = skljson.to_dict(model)
        deserialized_dict_model = skljson.from_dict(serialized_dict_model)

        skljson.to_json(model, model_name)
        deserialized_json_model = skljson.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:

            for key in sorted(model.__dict__.keys()):
                if isinstance(model.__dict__[key], np.ndarray):
                    print(key, (model.__dict__[key] == deserialized_dict_model.__dict__[key]).all())
                else:
                    print(key, model.__dict__[key] == deserialized_dict_model.__dict__[key])

            actual_ft = deserialized_model.fit_transform(data)

            np.testing.assert_array_almost_equal(expected_ft, actual_ft)

    def check_predict_model(self, model, model_name, data):
        model.fit(data)
        expected_p = model.predict(data)

        serialized_dict_model = skljson.to_dict(model)
        deserialized_dict_model = skljson.from_dict(serialized_dict_model)

        skljson.to_json(model, model_name)
        deserialized_json_model = skljson.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:

            for key in sorted(model.__dict__.keys()):
                if isinstance(model.__dict__[key], np.ndarray):
                    print(key, (model.__dict__[key] == deserialized_dict_model.__dict__[key]).all())
                else:
                    print(key, model.__dict__[key] == deserialized_dict_model.__dict__[key])

            deserialized_model.fit(data)
            actual_p = deserialized_model.predict(data)

            np.testing.assert_array_equal(expected_p, actual_p)

    def check_fitpredict_model(self, model, model_name, data):
        expected_fp = model.fit_predict(data)

        serialized_dict_model = skljson.to_dict(model)
        deserialized_dict_model = skljson.from_dict(serialized_dict_model)

        skljson.to_json(model, model_name)
        deserialized_json_model = skljson.from_json(model_name)
        os.remove(model_name)

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
            self.check_transform_model(model, 'kmeans.json', self.X)
            self.check_fittransform_model(model, 'kmeans.json', self.X)
            self.check_predict_model(model, 'kmeans.json', self.X)
            self.check_fitpredict_model(model, 'kmeans.json', self.X)

    def test_minibatch_kmeans(self):
        for model in [MiniBatchKMeans(n_clusters=self.n_centers, init='k-means++',
                             random_state=1234, n_init=100, max_iter=10000,
                             verbose=0, tol=1e-999),
                      MiniBatchKMeans(n_clusters=self.n_centers, init='random',
                             random_state=1234, n_init=100, max_iter=10000,
                             verbose=0, tol=1e-999)]:
            self.check_transform_model(model, 'minibatch-kmeans.json', self.X)
            self.check_fittransform_model(model, 'minibatch-kmeans.json', self.X)
            self.check_predict_model(model, 'minibatch-kmeans.json', self.X)
            self.check_fitpredict_model(model, 'minibatch-kmeans.json', self.X)

    def test_affinity_propagation(self):
        self.check_predict_model(AffinityPropagation(), 'affinity-propagation.json', self.simple_X)
        self.check_fitpredict_model(AffinityPropagation(), 'affinity-propagation.json', self.simple_X)

    def test_agglomerative_clustering(self):
        self.check_fitpredict_model(AgglomerativeClustering(), 'agglomerative-clustering.json', self.simple_X)

    def test_dbscan(self):
        self.check_fitpredict_model(DBSCAN(), 'dbscan.json', self.X)

    def test_optics(self):
        self.check_fitpredict_model(OPTICS(), 'optics.json', self.simple_X)

    def test_spectral_clustering(self):
        self.check_fitpredict_model(SpectralClustering(random_state=1234, n_clusters=2), 'spectral.json', self.simple_X)

    def test_feature_agglomeration(self):
        self.check_transform_model(FeatureAgglomeration(pooling_func=np.mean), 'feature-agg.json', self.X)
        self.check_fittransform_model(FeatureAgglomeration(pooling_func=np.mean), 'feature-agg.json', self.X)

    def test_meanshift(self):
        self.check_predict_model(MeanShift(), 'meanshift.json', self.simple_X)
        self.check_fitpredict_model(MeanShift(), 'meanshift.json', self.simple_X)

    def check_spectral_model(self, model, model_name, n_clusters):
        data, rows, columns = make_checkerboard(shape=(300, 300), n_clusters=n_clusters,
                                                noise=10, shuffle=False, random_state=1234)
        rng = np.random.RandomState(1234)
        row_idx = rng.permutation(data.shape[0])
        col_idx = rng.permutation(data.shape[1])
        data = data[row_idx][:, col_idx]

        # Create model
        model.fit(data)

        # Compare internal data to serialized
        expected_indices = [model.get_indices(i) for i in range(len(model.biclusters_))]
        expected_shapes = [model.get_shape(i) for i in range(len(model.biclusters_))]
        expected_matrices = [model.get_submatrix(i, data) for i in range(len(model.biclusters_))]

        serialized_dict_model = skljson.to_dict(model)
        deserialized_dict_model = skljson.from_dict(serialized_dict_model)

        skljson.to_json(model, model_name)
        deserialized_json_model = skljson.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_indices = [deserialized_model.get_indices(i) for i in range(len(deserialized_model.biclusters_))]
            actual_shapes = [deserialized_model.get_shape(i) for i in range(len(deserialized_model.biclusters_))]
            actual_matrices = [deserialized_model.get_submatrix(i, data) for i in range(len(deserialized_model.biclusters_))]

            self.assertEqual(len(expected_indices), len(actual_indices))
            for (w, x), (y,z) in zip(expected_indices, actual_indices):
                np.testing.assert_array_equal(w, y)
                np.testing.assert_array_equal(x, z)

            self.assertEqual(len(expected_shapes), len(actual_shapes))
            for x, y in zip(expected_shapes, actual_shapes):
                self.assertEqual(x, y)

            self.assertEqual(len(expected_matrices), len(actual_matrices))
            for x, y in zip(expected_matrices, actual_matrices):
                np.testing.assert_array_equal(x, y)

    def test_spectral_biclustering(self):
        n_clusters = (4, 3)
        self.check_spectral_model(SpectralBiclustering(n_clusters=n_clusters, method="log", random_state=1234),
                                  'spectral-biclus.json', n_clusters)

    def test_spectral_coclustering(self):
        n_clusters = 5
        self.check_spectral_model(SpectralCoclustering(n_clusters=n_clusters, svd_method="arpack", random_state=1234),
                                  'spectral-coclus.json', n_clusters)

    def test_kmodes(self):
        self.check_fitpredict_model(KModes(), 'kmodes.json', self.X)

    def check_kprototype_model(self, model, model_name, data):

        rng = np.random.default_rng(1234)
        cat_data = rng.permuted(data.astype(int))
        all_data = np.concatenate((data, cat_data), axis=1)

        cat_indices = [i + data.shape[1] for i in range(cat_data.shape[1])]

        model.fit(all_data, categorical=cat_indices)
        expected_t = model.predict(all_data, categorical=cat_indices)

        serialized_dict_model = skljson.to_dict(model)
        deserialized_dict_model = skljson.from_dict(serialized_dict_model)

        skljson.to_json(model, model_name)
        deserialized_json_model = skljson.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:

            for key in sorted(model.__dict__.keys()):
                if isinstance(model.__dict__[key], np.ndarray):
                    print(key, (model.__dict__[key] == deserialized_dict_model.__dict__[key]).all())
                else:
                    print(key, model.__dict__[key] == deserialized_dict_model.__dict__[key])

            actual_t = deserialized_model.predict(all_data, categorical=cat_indices)

            np.testing.assert_array_almost_equal(expected_t, actual_t)

    def test_kprototypes(self):
        self.check_kprototype_model(KPrototypes(n_clusters=2, random_state=1234), 'kproto.json', self.X)
