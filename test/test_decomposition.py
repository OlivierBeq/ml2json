# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.datasets import load_iris, make_sparse_coded_signal, make_friedman1, fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.decomposition import (PCA, KernelPCA, DictionaryLearning, FactorAnalysis, FastICA, IncrementalPCA,
                                   LatentDirichletAllocation, MiniBatchDictionaryLearning, MiniBatchSparsePCA, NMF,
                                   MiniBatchNMF, SparsePCA, SparseCoder, TruncatedSVD)

from src import sklearn_json as skljson


class TestAPI(unittest.TestCase):

    def setUp(self):
        self.X, self.y = load_iris(return_X_y=True)
        self.dict_X, self.dict_dict, self.dict_code = make_sparse_coded_signal(n_samples=20, n_components=15,
                                                                               n_features=10, n_nonzero_coefs=10,
                                                                               random_state=1234, data_transposed=False)
        self.friedman, _ = make_friedman1(n_samples=200, n_features=30, random_state=1234)
        self.news, _ = fetch_20newsgroups(shuffle=True, random_state=1234, return_X_y=True,
                                          remove=("headers", "footers", "quotes"))
        # Convert to term frequencies
        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=50, stop_words="english")
        tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=50, stop_words="english")
        self.tf_news = tf_vectorizer.fit_transform(self.news[:200])
        self.tfidf_news = tfidf_vectorizer.fit_transform(self.news[:200])


    def check_model(self, model, model_name):
        expected_ft = model.fit_transform(self.X)
        expected_t = model.transform(self.X)
        expected_it = model.inverse_transform(expected_t)

        serialized_dict_model = skljson.to_dict(model)
        deserialized_dict_model = skljson.from_dict(serialized_dict_model)

        skljson.to_json(model, model_name)
        deserialized_json_model = skljson.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_t = deserialized_model.transform(self.X)
            actual_ft = deserialized_model.fit_transform(self.X)
            actual_it = deserialized_model.inverse_transform(actual_ft)

            np.testing.assert_array_almost_equal(expected_t, actual_t)
            np.testing.assert_array_almost_equal(expected_ft, actual_ft)
            np.testing.assert_array_almost_equal(expected_it, actual_it)

    def test_pca(self):
        self.check_model(PCA(n_components=2, svd_solver='full'), 'pca.json')
        self.check_model(PCA(svd_solver='arpack'), 'pca.json')

    def test_kernel_pca(self):
        self.check_model(KernelPCA(fit_inverse_transform=True), 'kernel-pca.json')

    def test_incremental_pca(self):
        self.check_model(IncrementalPCA(), 'incremental-pca.json')

    def test_sparse_pca(self):
        self.check_fit_transform_model(SparsePCA(random_state=1234), 'sparse-pca.json', self.friedman)

    def test_minibatch_sparse_pca(self):
        self.check_fit_transform_model(MiniBatchSparsePCA(random_state=1234), 'minibatch-sparse-pca.json', self.friedman)

    def check_fit_transform_model(self, model, model_name, data):
        expected_ft = model.fit_transform(data)
        expected_t = model.transform(data)

        serialized_dict_model = skljson.to_dict(model)
        deserialized_dict_model = skljson.from_dict(serialized_dict_model)

        skljson.to_json(model, model_name)
        deserialized_json_model = skljson.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_t = deserialized_model.transform(data)
            actual_ft = deserialized_model.fit_transform(data)

            np.testing.assert_array_almost_equal(expected_t, actual_t)
            np.testing.assert_array_almost_equal(expected_ft, actual_ft)

    def test_dictionary_learning(self):
        self.check_fit_transform_model(DictionaryLearning(n_components=10, transform_algorithm='lars',
                                                          transform_alpha=0.1, random_state=1234),
                                       'dictionary-learning.json',
                                       self.dict_X)
        self.check_fit_transform_model(DictionaryLearning(n_components=10, transform_algorithm='lasso_lars',
                                                          transform_alpha=0.1, random_state=1234),
                                       'dictionary-learning.json',
                                       self.dict_X)
        self.check_fit_transform_model(DictionaryLearning(n_components=10, transform_algorithm='lasso_cd',
                                                          transform_alpha=0.1, random_state=1234),
                                       'dictionary-learning.json',
                                       self.dict_X)
        self.check_fit_transform_model(DictionaryLearning(n_components=10, transform_algorithm='omp',
                                                          transform_alpha=0.1, random_state=1234),
                                       'dictionary-learning.json',
                                       self.dict_X)
        self.check_fit_transform_model(DictionaryLearning(n_components=10, transform_algorithm='threshold',
                                                          transform_alpha=0.1, random_state=1234),
                                       'dictionary-learning.json',
                                       self.dict_X)

    def test_minibatch_dictionary_learning(self):
        self.check_fit_transform_model(MiniBatchDictionaryLearning(n_components=10, transform_algorithm='lars',
                                                                   transform_alpha=0.1, random_state=1234,
                                                                   batch_size=256, max_iter=100),
                                       'dictionary-learning.json',
                                       self.dict_X)
        self.check_fit_transform_model(MiniBatchDictionaryLearning(n_components=10, transform_algorithm='omp',
                                                                   transform_alpha=0.1, random_state=1234,
                                                                   batch_size=256, max_iter=100),
                                       'dictionary-learning.json',
                                       self.dict_X)
        self.check_fit_transform_model(MiniBatchDictionaryLearning(n_components=10, transform_algorithm='threshold',
                                                                   transform_alpha=0.1, random_state=1234,
                                                                   batch_size=256, max_iter=100),
                                       'dictionary-learning.json',
                                       self.dict_X)
        self.check_fit_transform_model(MiniBatchDictionaryLearning(n_components=10, transform_algorithm='lasso_lars',
                                                                   transform_alpha=0.1, random_state=1234,
                                                                   batch_size=256, transform_max_iter=100,
                                                                   max_iter=100),
                                       'dictionary-learning.json',
                                       self.dict_X)
        self.check_fit_transform_model(MiniBatchDictionaryLearning(n_components=10, transform_algorithm='lasso_cd',
                                                                   transform_alpha=0.1, random_state=1234,
                                                                   batch_size=256, transform_max_iter=100,
                                                                   max_iter=100),
                                       'dictionary-learning.json',
                                       self.dict_X)

    def test_factor_analysis(self):
        self.check_fit_transform_model(FactorAnalysis(random_state=1234),
                                       'factor-analysis.json',
                                       self.X)
        self.check_fit_transform_model(FactorAnalysis(svd_method='randomized', random_state=1234),
                                       'factor-analysis.json',
                                       self.X)
        self.check_fit_transform_model(FactorAnalysis(rotation='varimax', random_state=1234),
                                       'factor-analysis.json',
                                       self.X)
        self.check_fit_transform_model(FactorAnalysis(rotation='quartimax', random_state=1234),
                                       'factor-analysis.json',
                                       self.X)

    def test_fast_ica(self):
        self.check_fit_transform_model(FastICA(algorithm='parallel', whiten='arbitrary-variance', random_state=1234),
                                       'fast-ica.json',
                                       self.X)
        self.check_fit_transform_model(FastICA(algorithm='parallel', whiten='unit-variance', random_state=1234),
                                       'fast-ica.json',
                                       self.X)
        self.check_fit_transform_model(FastICA(algorithm='deflation', whiten='arbitrary-variance', random_state=1234),
                                       'fast-ica.json',
                                       self.X)
        self.check_fit_transform_model(FastICA(algorithm='deflation', whiten='unit-variance', random_state=1234),
                                       'fast-ica.json',
                                       self.X)

    def test_latent_dirichlet_allocation(self):
        self.check_fit_transform_model(LatentDirichletAllocation(random_state=1234),
                                       'latent-dirichlet-allocation.json', self.tf_news)

    def test_nmf(self):
        self.check_fit_transform_model(NMF(random_state=1234),
                                       'nmf.json', self.tfidf_news)

    def test_minibatch_nmf(self):
        self.check_fit_transform_model(MiniBatchNMF(random_state=1234, max_iter=2000),
                                       'minibatch-nmf.json', self.tfidf_news)

    def test_sparse_coder(self):
        def ricker_function(resolution, center, width):
            """Discrete sub-sampled Ricker (Mexican hat) wavelet"""
            x = np.linspace(0, resolution - 1, resolution)
            x = (
                    (2 / (np.sqrt(3 * width) * np.pi ** 0.25))
                    * (1 - (x - center) ** 2 / width ** 2)
                    * np.exp(-((x - center) ** 2) / (2 * width ** 2))
            )
            return x

        def ricker_matrix(width, resolution, n_components):
            """Dictionary of Ricker (Mexican hat) wavelets"""
            centers = np.linspace(0, resolution - 1, n_components)
            D = np.empty((n_components, resolution))
            for i, center in enumerate(centers):
                D[i] = ricker_function(resolution, center, width)
            D /= np.sqrt(np.sum(D ** 2, axis=1))[:, np.newaxis]
            return D

        resolution = 1024
        subsampling = 3  # subsampling factor
        width = 100
        n_components = resolution // subsampling

        # Compute a wavelet dictionary
        D_fixed = ricker_matrix(width=width, resolution=resolution, n_components=n_components)
        D_multi = np.r_[tuple(ricker_matrix(width=w, resolution=resolution, n_components=n_components // 5)
                              for w in (10, 50, 100, 500, 1000))]

        # Generate a signal
        y = np.linspace(0, resolution - 1, resolution)
        first_quarter = y < resolution / 4
        y[first_quarter] = 3.0
        y[np.logical_not(first_quarter)] = -1.0

        # List the different sparse coding methods in the following format:
        # (title, transform_algorithm, transform_alpha,
        #  transform_n_nozero_coefs)
        estimators = [("OMP", "omp", None, 15),
                      ("Lasso", "lasso_lars", 2, None)]

        model_name = 'sparse-coder.json'

        for D in (D_fixed, D_multi):
            # Do a wavelet approximation
            for title, algo, alpha, n_nonzero in estimators:
                coder = SparseCoder(
                    dictionary=D,
                    transform_n_nonzero_coefs=n_nonzero,
                    transform_alpha=alpha,
                    transform_algorithm=algo,
                )
                expected_t = coder.transform(y.reshape(1, -1))

                serialized_dict_model = skljson.to_dict(coder)
                deserialized_dict_model = skljson.from_dict(serialized_dict_model)

                skljson.to_json(coder, model_name)
                deserialized_json_model = skljson.from_json(model_name)
                os.remove(model_name)

                for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
                    actual_t = deserialized_model.transform(y.reshape(1, -1))

                    np.testing.assert_array_almost_equal(expected_t, actual_t)

    def test_truncated_svd(self):
        rng = np.random.RandomState(1234)
        X_dense = rng.rand(100, 100)
        X_dense[:, 2 * np.arange(50)] = 0
        X = csr_matrix(X_dense)
        model = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
        model.fit(X)

        expected_t = model.transform(X)

        serialized_dict_model = skljson.to_dict(model)
        deserialized_dict_model = skljson.from_dict(serialized_dict_model)

        model_name = 'truncated-svd.json'
        skljson.to_json(model, model_name)
        deserialized_json_model = skljson.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_t = deserialized_model.transform(X)

            np.testing.assert_array_almost_equal(expected_t, actual_t)
