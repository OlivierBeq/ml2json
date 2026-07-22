# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np
from sklearn.datasets import make_blobs, make_checkerboard
from sklearn.cluster import (AffinityPropagation, AgglomerativeClustering,
                             Birch, DBSCAN, FeatureAgglomeration, KMeans,
                             BisectingKMeans, MiniBatchKMeans, MeanShift, OPTICS,
                             SpectralClustering, SpectralBiclustering, SpectralCoclustering,
                             HDBSCAN as SklearnHDBSCAN)

# Allow testing of additional optional dependencies
__optionals__ = []
try:
    from kmodes.kmodes import KModes
    from kmodes.kprototypes import KPrototypes
    __optionals__.extend(['KModes', 'KPrototypes'])
except:
    pass
try:
    from hdbscan import HDBSCAN, RobustSingleLinkage
    __optionals__.extend(['HDBSCAN', 'RobustSingleLinkage'])
except:
    pass
try:
    from sklearn_extra.cluster import KMedoids, CommonNNClustering
    __optionals__.extend(['KMedoids', 'CommonNNClustering'])
except:
    pass

from src import ml2json


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

            actual_t = deserialized_model.transform(data)

            np.testing.assert_array_almost_equal(expected_t, actual_t)

    def check_fittransform_model(self, model, model_name, data):
        expected_ft = model.fit_transform(data)

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

            actual_ft = deserialized_model.transform(data)

            np.testing.assert_array_almost_equal(expected_ft, actual_ft)

    def check_predict_model(self, model, model_name, data):
        model.fit(data)
        expected_p = model.predict(data)

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

            actual_p = deserialized_model.predict(data)

            np.testing.assert_array_equal(expected_p, actual_p)

    def check_fitpredict_model(self, model, model_name, data):
        expected_fp = model.fit_predict(data)

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

            actual_fp = deserialized_model.fit_predict(data)

            np.testing.assert_array_equal(expected_fp, actual_fp)

    def check_fitpredict_and_predict_model(self, model, model_name, data):
        expected_fp = model.fit_predict(data)

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

            actual_fp = deserialized_model.predict(data)

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
            self.check_fitpredict_and_predict_model(model, 'kmeans.json', self.X)

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
            self.check_fitpredict_and_predict_model(model, 'minibatch-kmeans.json', self.X)

    def test_affinity_propagation(self):
        self.check_predict_model(AffinityPropagation(), 'affinity-propagation.json', self.simple_X)
        self.check_fitpredict_and_predict_model(AffinityPropagation(), 'affinity-propagation.json', self.simple_X)

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
        self.check_fitpredict_and_predict_model(MeanShift(), 'meanshift.json', self.simple_X)

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

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
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
        if 'KModes' in __optionals__:
            self.check_fitpredict_and_predict_model(KModes(random_state=1234), 'kmodes.json', self.X)

    def check_kprototype_model(self, model, model_name, data):

        rng = np.random.default_rng(1234)
        cat_data = rng.permuted(data.astype(int))
        all_data = np.concatenate((data, cat_data), axis=1)

        cat_indices = [i + data.shape[1] for i in range(cat_data.shape[1])]

        model.fit(all_data, categorical=cat_indices)
        expected_t = model.predict(all_data, categorical=cat_indices)

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

            actual_t = deserialized_model.predict(all_data, categorical=cat_indices)

            np.testing.assert_array_almost_equal(expected_t, actual_t)

    def test_kprototypes(self):
        if 'KPrototypes' in __optionals__:
            self.check_kprototype_model(KPrototypes(n_clusters=2, random_state=1234), 'kproto.json', self.X)

    def test_birch(self):
        self.check_fitpredict_and_predict_model(Birch(), 'birch.json', self.X)
        self.check_predict_model(Birch(), 'birch.json', self.X)
        self.check_fittransform_model(Birch(), 'birch.json', self.X)
        self.check_transform_model(Birch(), 'birch.json', self.X)

    def test_bisecting_kmeans(self):
        self.check_fitpredict_and_predict_model(
            BisectingKMeans(n_clusters=2, tol=1e-999, random_state=1234, n_init=100, max_iter=10000),
            'bisecting-kmeans.json', self.X)
        self.check_predict_model(
            BisectingKMeans(n_clusters=2, tol=1e-999, random_state=1234, n_init=100, max_iter=10000),
            'bisecting-kmeans.json', self.X)
        self.check_fittransform_model(
            BisectingKMeans(n_clusters=2, tol=1e-999, random_state=1234, n_init=100, max_iter=10000),
            'bisecting-kmeans.json', self.X)
        self.check_transform_model(
            BisectingKMeans(n_clusters=2, tol=1e-999, random_state=1234, n_init=100, max_iter=10000),
            'bisecting-kmeans.json', self.X)

    def test_hdbscan(self):
        if 'HDBSCAN' in __optionals__:
            self.check_fitpredict_model(HDBSCAN(), 'hdbscan.json', self.X)
            self.check_fitpredict_model(HDBSCAN(gen_min_span_tree=True), 'hdbscan.json', self.X)

    def test_sklearn_hdbscan(self):
        self.check_fitpredict_model(SklearnHDBSCAN(), 'sklearn-hdbscan.json', self.X)
        self.check_fitpredict_model(SklearnHDBSCAN(store_centers='both'), 'sklearn-hdbscan.json', self.X)

    def test_robust_single_linkage(self):
        if 'RobustSingleLinkage' in __optionals__:
            self.check_fitpredict_model(RobustSingleLinkage(), 'robust-single-linkage.json', self.X)

    def test_agglomerative_clustering_linkage_metric(self):
        for linkage, metric in [('ward', 'euclidean'), ('complete', 'euclidean'), ('complete', 'manhattan'),
                                 ('complete', 'cosine'), ('average', 'euclidean'), ('average', 'manhattan'),
                                 ('single', 'euclidean')]:
            self.check_fitpredict_model(
                AgglomerativeClustering(n_clusters=3, linkage=linkage, metric=metric),
                'agglomerative-clustering.json', self.simple_X)

    def test_agglomerative_clustering_distance_threshold(self):
        self.check_fitpredict_model(
            AgglomerativeClustering(n_clusters=None, distance_threshold=3.0),
            'agglomerative-clustering.json', self.simple_X)

    def test_agglomerative_clustering_connectivity(self):
        from sklearn.neighbors import kneighbors_graph
        connectivity = kneighbors_graph(self.simple_X, n_neighbors=2, include_self=False)
        self.check_fitpredict_model(
            AgglomerativeClustering(n_clusters=2, linkage='ward', connectivity=connectivity),
            'agglomerative-clustering.json', self.simple_X)

    def test_dbscan_algorithm(self):
        for algorithm in ['auto', 'ball_tree', 'kd_tree', 'brute']:
            self.check_fitpredict_model(DBSCAN(algorithm=algorithm), 'dbscan.json', self.simple_X)

    def test_dbscan_manhattan(self):
        self.check_fitpredict_model(DBSCAN(metric='manhattan'), 'dbscan.json', self.simple_X)

    def test_dbscan_precomputed(self):
        from sklearn.metrics import pairwise_distances
        D = pairwise_distances(self.simple_X, metric='euclidean')
        np.testing.assert_array_almost_equal(D, D.T)
        self.check_fitpredict_model(DBSCAN(metric='precomputed', eps=3), 'dbscan.json', D)

    def test_optics_algorithm(self):
        for algorithm in ['auto', 'ball_tree', 'kd_tree', 'brute']:
            self.check_fitpredict_model(OPTICS(algorithm=algorithm, min_samples=2), 'optics.json', self.simple_X)

    def test_optics_min_samples_eps(self):
        self.check_fitpredict_model(OPTICS(min_samples=3, eps=2.0, cluster_method='dbscan'), 'optics.json', self.simple_X)

    def test_spectral_clustering_affinity(self):
        for affinity in ['rbf', 'nearest_neighbors']:
            self.check_fitpredict_model(
                SpectralClustering(random_state=1234, n_clusters=2, affinity=affinity, n_neighbors=3),
                'spectral.json', self.simple_X)

    def test_spectral_clustering_precomputed(self):
        from sklearn.metrics.pairwise import rbf_kernel
        affinity = rbf_kernel(self.simple_X)
        np.testing.assert_array_almost_equal(affinity, affinity.T)
        self.check_fitpredict_model(
            SpectralClustering(random_state=1234, n_clusters=2, affinity='precomputed'),
            'spectral.json', affinity)

    def test_spectral_clustering_assign_labels(self):
        for assign_labels in ['kmeans', 'discretize', 'cluster_qr']:
            self.check_fitpredict_model(
                SpectralClustering(random_state=1234, n_clusters=2, assign_labels=assign_labels),
                'spectral.json', self.simple_X)

    def test_affinity_propagation_precomputed(self):
        from sklearn.metrics.pairwise import euclidean_distances
        S = -euclidean_distances(self.simple_X, squared=True)
        np.testing.assert_array_almost_equal(S, S.T)
        self.check_fitpredict_model(
            AffinityPropagation(affinity='precomputed', random_state=1234), 'affinity-propagation.json', S)

    def test_affinity_propagation_damping(self):
        self.check_predict_model(AffinityPropagation(damping=0.9, random_state=1234), 'affinity-propagation.json', self.simple_X)

    def test_meanshift_bandwidth(self):
        from sklearn.cluster import estimate_bandwidth
        bw = estimate_bandwidth(self.X, random_state=1234)
        self.check_predict_model(MeanShift(bandwidth=bw), 'meanshift.json', self.X)

    def test_meanshift_bin_seeding(self):
        self.check_predict_model(MeanShift(bin_seeding=True), 'meanshift.json', self.simple_X)

    def test_meanshift_cluster_all_false(self):
        self.check_fitpredict_model(MeanShift(cluster_all=False), 'meanshift.json', self.simple_X)

    def test_birch_threshold_branching(self):
        self.check_fitpredict_and_predict_model(Birch(threshold=0.3, branching_factor=20), 'birch.json', self.X)

    def test_birch_n_clusters_none(self):
        self.check_fittransform_model(Birch(n_clusters=None), 'birch.json', self.X)

    def test_birch_n_clusters_estimator(self):
        self.check_fitpredict_and_predict_model(
            Birch(n_clusters=AgglomerativeClustering(n_clusters=3)), 'birch.json', self.X)

    def test_sklearn_hdbscan_cluster_selection_method(self):
        for method in ['eom', 'leaf']:
            self.check_fitpredict_model(SklearnHDBSCAN(cluster_selection_method=method), 'sklearn-hdbscan.json', self.X)

    def test_sklearn_hdbscan_metric(self):
        for metric in ['manhattan', 'chebyshev']:
            self.check_fitpredict_model(SklearnHDBSCAN(metric=metric), 'sklearn-hdbscan.json', self.X)

    def test_sklearn_hdbscan_algorithm(self):
        for algorithm in ['auto', 'ball_tree', 'kd_tree', 'brute']:
            self.check_fitpredict_model(SklearnHDBSCAN(algorithm=algorithm), 'sklearn-hdbscan.json', self.X)

    def test_sklearn_hdbscan_store_centers(self):
        for store_centers in ['centroid', 'medoid']:
            self.check_fitpredict_model(SklearnHDBSCAN(store_centers=store_centers), 'sklearn-hdbscan.json', self.X)

    def test_sklearn_hdbscan_allow_single_cluster(self):
        self.check_fitpredict_model(SklearnHDBSCAN(allow_single_cluster=True), 'sklearn-hdbscan.json', self.X)

    def test_hdbscan_cluster_selection_method(self):
        if 'HDBSCAN' in __optionals__:
            for method in ['eom', 'leaf']:
                self.check_fitpredict_model(HDBSCAN(cluster_selection_method=method), 'hdbscan.json', self.X)

    def test_kmedoids(self):
        if 'KMedoids' in __optionals__:
            self.check_predict_model(KMedoids(n_clusters=self.n_centers, random_state=1234), 'kmedoids.json', self.X)

    def test_kmedoids_method_init(self):
        if 'KMedoids' in __optionals__:
            for method, init in [('alternate', 'random'), ('pam', 'heuristic'), ('pam', 'k-medoids++')]:
                self.check_predict_model(
                    KMedoids(n_clusters=self.n_centers, method=method, init=init, random_state=1234),
                    'kmedoids.json', self.X)

    def test_kmedoids_metric(self):
        if 'KMedoids' in __optionals__:
            self.check_predict_model(
                KMedoids(n_clusters=self.n_centers, metric='manhattan', random_state=1234),
                'kmedoids.json', self.X)

    def test_common_nn_clustering(self):
        if 'CommonNNClustering' in __optionals__:
            self.check_fitpredict_model(CommonNNClustering(eps=2.0, min_samples=3), 'common-nn-clustering.json', self.simple_X)

    def test_hdbscan_algorithm(self):
        if 'HDBSCAN' in __optionals__:
            for algorithm in ['best', 'generic', 'prims_kdtree', 'prims_balltree', 'boruvka_kdtree', 'boruvka_balltree']:
                self.check_fitpredict_model(HDBSCAN(algorithm=algorithm), 'hdbscan.json', self.X)

    def test_hdbscan_metric(self):
        if 'HDBSCAN' in __optionals__:
            for metric in ['manhattan', 'chebyshev']:
                self.check_fitpredict_model(HDBSCAN(metric=metric, algorithm='generic'), 'hdbscan.json', self.X)

    def test_spectral_biclustering_method_svd(self):
        n_clusters = (3, 2)
        for method in ['bistochastic', 'scale', 'log']:
            self.check_spectral_model(
                SpectralBiclustering(n_clusters=n_clusters, method=method, random_state=1234),
                'spectral-biclus.json', n_clusters)
        for svd_method in ['randomized', 'arpack']:
            self.check_spectral_model(
                SpectralBiclustering(n_clusters=n_clusters, method='log', svd_method=svd_method, random_state=1234),
                'spectral-biclus.json', n_clusters)

    def test_spectral_coclustering_svd_method(self):
        n_clusters = 4
        for svd_method in ['randomized', 'arpack']:
            self.check_spectral_model(
                SpectralCoclustering(n_clusters=n_clusters, svd_method=svd_method, random_state=1234),
                'spectral-coclus.json', n_clusters)

    def test_float32_input(self):
        X32 = self.simple_X.astype(np.float32)
        self.check_fitpredict_model(KMeans(n_clusters=self.n_centers, random_state=1234, n_init=10), 'kmeans.json', self.X.astype(np.float32))
        self.check_fitpredict_model(DBSCAN(), 'dbscan.json', X32)
        self.check_fitpredict_model(AgglomerativeClustering(n_clusters=2), 'agglomerative-clustering.json', X32)
