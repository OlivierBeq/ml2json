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

from src import ml2json


def _cube_fun(x):
    return x ** 3, (3 * x ** 2).mean(axis=-1)


class TestAPI(unittest.TestCase):

    def setUp(self):
        self.X, self.y = load_iris(return_X_y=True)
        self.dict_X, self.dict_dict, self.dict_code = make_sparse_coded_signal(n_samples=20, n_components=15,
                                                                               n_features=10, n_nonzero_coefs=10,
                                                                               random_state=1234)
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

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
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

    def test_pca_solvers(self):
        for solver in ['auto', 'randomized', 'covariance_eigh']:
            self.check_model(PCA(n_components=2, svd_solver=solver, random_state=1234), 'pca-solver.json')

    def test_pca_whiten(self):
        self.check_model(PCA(n_components=2, svd_solver='full', whiten=True), 'pca-whiten.json')

    def test_pca_n_components_variants(self):
        self.check_model(PCA(n_components=0.95, svd_solver='full'), 'pca-ncomp-float.json')
        self.check_model(PCA(n_components='mle', svd_solver='full'), 'pca-mle.json')

    def test_pca_rank_deficient(self):
        rng = np.random.RandomState(1234)
        base = rng.rand(20, 3)
        # Duplicate/collinear columns so n_features > n_samples-ish rank and
        # the covariance matrix used internally is singular.
        X_wide = np.hstack([base, base, base, base * 2, base + 1])
        self.check_model(PCA(n_components=3, svd_solver='full'), 'pca-rank-deficient.json')
        model = PCA(n_components=3, svd_solver='full')
        model.fit(X_wide)
        expected_t = model.transform(X_wide)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)
        actual_t = deserialized_dict_model.transform(X_wide)
        np.testing.assert_array_almost_equal(expected_t, actual_t)

    def test_kernel_pca(self):
        self.check_model(KernelPCA(fit_inverse_transform=True), 'kernel-pca.json')

    def test_kernel_pca_kernels(self):
        # decimal=4 rather than check_model's default 6: the "fit_transform on
        # the deserialized model" leg re-fits from scratch (a fresh eigen-
        # decomposition, not the round-tripped one), and poly/sigmoid's
        # kernel-ridge-regression pre-image reconstruction amplifies that
        # ordinary re-fit floating-point noise well past 1e-6.
        for kernel in ['linear', 'poly', 'rbf', 'sigmoid', 'cosine']:
            model = KernelPCA(kernel=kernel, n_components=2, fit_inverse_transform=True)
            expected_t = model.fit_transform(self.X)
            expected_it = model.inverse_transform(expected_t)

            serialized_dict_model = ml2json.to_dict(model)
            deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

            actual_t = deserialized_dict_model.transform(self.X)
            actual_ft = deserialized_dict_model.fit_transform(self.X)
            actual_it = deserialized_dict_model.inverse_transform(actual_ft)

            np.testing.assert_array_almost_equal(expected_t, actual_t)
            np.testing.assert_array_almost_equal(expected_t, actual_ft, decimal=4)
            np.testing.assert_array_almost_equal(expected_it, actual_it, decimal=4)

    def test_kernel_pca_precomputed(self):
        gram = self.X.dot(self.X.T)
        np.testing.assert_array_almost_equal(gram, gram.T)
        model = KernelPCA(kernel='precomputed', n_components=2)
        model.fit(gram)
        expected_t = model.transform(gram)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        model_name = 'kernel-pca-precomputed.json'
        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_t = deserialized_model.transform(gram)
            np.testing.assert_array_almost_equal(expected_t, actual_t)

    def test_incremental_pca(self):
        self.check_model(IncrementalPCA(), 'incremental-pca.json')

    def test_pca_float32_input(self):
        self.check_model(PCA(n_components=2, svd_solver='full'), 'pca-float32.json')
        model = PCA(n_components=2, svd_solver='full')
        X32 = self.X.astype(np.float32)
        model.fit(X32)
        expected_t = model.transform(X32)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)
        actual_t = deserialized_dict_model.transform(X32)
        np.testing.assert_array_almost_equal(expected_t, actual_t)

    def test_sparse_pca(self):
        self.check_fit_transform_model(SparsePCA(random_state=1234), 'sparse-pca.json', self.friedman)

    def test_minibatch_sparse_pca(self):
        self.check_fit_transform_model(MiniBatchSparsePCA(random_state=1234), 'minibatch-sparse-pca.json', self.friedman)

    def check_fit_transform_model(self, model, model_name, data):
        expected_ft = model.fit_transform(data)
        expected_t = model.transform(data)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_t = deserialized_model.transform(data)
            actual_ft = deserialized_model.fit_transform(data)

            np.testing.assert_array_almost_equal(expected_t, actual_t)
            np.testing.assert_array_almost_equal(expected_ft, actual_ft)

    def test_dictionary_learning(self):
        self.check_fit_transform_model(DictionaryLearning(n_components=10, transform_algorithm='lars',
                                                          transform_alpha=0.1, random_state=1234),
                                       'dictionary-learning1.json',
                                       self.dict_X)
        self.check_fit_transform_model(DictionaryLearning(n_components=10, transform_algorithm='lasso_lars',
                                                          transform_alpha=0.1, random_state=1234),
                                       'dictionary-learning2.json',
                                       self.dict_X)
        self.check_fit_transform_model(DictionaryLearning(n_components=10, transform_algorithm='lasso_cd',
                                                          transform_alpha=0.1, random_state=1234),
                                       'dictionary-learning3.json',
                                       self.dict_X)
        self.check_fit_transform_model(DictionaryLearning(n_components=10, transform_algorithm='omp',
                                                          transform_alpha=0.1, random_state=1234),
                                       'dictionary-learning4.json',
                                       self.dict_X)
        self.check_fit_transform_model(DictionaryLearning(n_components=10, transform_algorithm='threshold',
                                                          transform_alpha=0.1, random_state=1234),
                                       'dictionary-learning5.json',
                                       self.dict_X)

    def test_minibatch_dictionary_learning(self):
        self.check_fit_transform_model(MiniBatchDictionaryLearning(n_components=10, transform_algorithm='lars',
                                                                   transform_alpha=0.1, random_state=1234,
                                                                   batch_size=256, max_iter=100),
                                       'dictionary-learning6.json',
                                       self.dict_X)
        self.check_fit_transform_model(MiniBatchDictionaryLearning(n_components=10, transform_algorithm='omp',
                                                                   transform_alpha=0.1, random_state=1234,
                                                                   batch_size=256, max_iter=100),
                                       'dictionary-learning7.json',
                                       self.dict_X)
        self.check_fit_transform_model(MiniBatchDictionaryLearning(n_components=10, transform_algorithm='threshold',
                                                                   transform_alpha=0.1, random_state=1234,
                                                                   batch_size=256, max_iter=100),
                                       'dictionary-learning8.json',
                                       self.dict_X)
        self.check_fit_transform_model(MiniBatchDictionaryLearning(n_components=10, transform_algorithm='lasso_lars',
                                                                   transform_alpha=0.1, random_state=1234,
                                                                   batch_size=256, transform_max_iter=100,
                                                                   max_iter=100),
                                       'dictionary-learning9.json',
                                       self.dict_X)
        self.check_fit_transform_model(MiniBatchDictionaryLearning(n_components=10, transform_algorithm='lasso_cd',
                                                                   transform_alpha=0.1, random_state=1234,
                                                                   batch_size=256, transform_max_iter=100,
                                                                   max_iter=100),
                                       'dictionary-learning10.json',
                                       self.dict_X)

    def test_factor_analysis(self):
        self.check_fit_transform_model(FactorAnalysis(random_state=1234),
                                       'factor-analysis.json',
                                       self.X)
        self.check_fit_transform_model(FactorAnalysis(svd_method='randomized', random_state=1234),
                                       'factor-analysis.json',
                                       self.X)
        self.check_fit_transform_model(FactorAnalysis(svd_method='lapack', random_state=1234),
                                       'factor-analysis.json',
                                       self.X)
        self.check_fit_transform_model(FactorAnalysis(rotation='varimax', random_state=1234),
                                       'factor-analysis.json',
                                       self.X)
        self.check_fit_transform_model(FactorAnalysis(rotation='quartimax', random_state=1234),
                                       'factor-analysis.json',
                                       self.X)
        self.check_fit_transform_model(FactorAnalysis(noise_variance_init=np.ones(self.X.shape[1]), random_state=1234),
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

    def test_fast_ica_fun_variants(self):
        for fun in ['logcosh', 'exp', 'cube']:
            self.check_fit_transform_model(FastICA(fun=fun, random_state=1234), 'fast-ica-fun.json', self.X)
        self.check_fit_transform_model(FastICA(fun=_cube_fun, random_state=1234),
                                       'fast-ica-fun-callable.json', self.X)

    def test_fast_ica_whiten_solver_and_float32(self):
        for whiten_solver in ['eigh', 'svd']:
            self.check_fit_transform_model(FastICA(whiten_solver=whiten_solver, random_state=1234),
                                           'fast-ica-whiten-solver.json', self.X)
        self.check_fit_transform_model(FastICA(random_state=1234), 'fast-ica-float32.json', self.X.astype(np.float32))

    def test_fast_ica_whiten_attribute_roundtrip(self):
        # Regression test: deserialize_fast_ica must restore the `whiten`
        # constructor attribute under its real name, not a typo'd one.
        model = FastICA(algorithm='parallel', whiten='unit-variance', random_state=1234)
        model.fit(self.X)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        self.assertEqual(model.whiten, deserialized_dict_model.whiten)
        self.assertFalse(hasattr(deserialized_dict_model, 'whithen'))

    def test_latent_dirichlet_allocation(self):
        self.check_fit_transform_model(LatentDirichletAllocation(random_state=1234),
                                       'latent-dirichlet-allocation.json', self.tf_news)

    def test_nmf(self):
        self.check_fit_transform_model(NMF(random_state=1234),
                                       'nmf.json', self.tfidf_news)

    def test_nmf_init_solver_beta_loss(self):
        # Dense, well-conditioned positive data - unlike the sparse, low-rank
        # tfidf_news fixture, nndsvd-family inits here don't hand back an
        # all-zero component row that would make model.transform() itself
        # raise regardless of serialization.
        dense_positive = np.abs(self.friedman)
        for init in ['random', 'nndsvd', 'nndsvda', 'nndsvdar']:
            self.check_fit_transform_model(NMF(n_components=5, init=init, random_state=1234),
                                           'nmf-init.json', dense_positive)
        self.check_fit_transform_model(NMF(n_components=5, solver='mu', beta_loss='kullback-leibler', random_state=1234),
                                       'nmf-mu-kl.json', dense_positive)
        strictly_positive_news = self.tfidf_news.toarray() + 0.1
        self.check_fit_transform_model(NMF(n_components=5, solver='mu', beta_loss='itakura-saito', random_state=1234),
                                       'nmf-mu-is.json', strictly_positive_news)
        self.check_fit_transform_model(NMF(n_components=5, solver='mu', beta_loss=1.5, random_state=1234),
                                       'nmf-mu-float-beta.json', dense_positive)
        self.check_fit_transform_model(NMF(n_components=5, alpha_W=0.1, alpha_H=0.1, l1_ratio=0.5, random_state=1234),
                                       'nmf-regularized.json', dense_positive)

    def test_minibatch_nmf(self):
        self.check_fit_transform_model(MiniBatchNMF(random_state=1234, max_iter=2000),
                                       'minibatch-nmf.json', self.tfidf_news)

    def test_minibatch_nmf_init_beta_loss(self):
        dense_positive = np.abs(self.friedman)
        for init in ['random', 'nndsvd', 'nndsvda', 'nndsvdar']:
            self.check_fit_transform_model(MiniBatchNMF(n_components=5, init=init, random_state=1234, max_iter=2000),
                                           'minibatch-nmf-init.json', dense_positive)
        self.check_fit_transform_model(MiniBatchNMF(n_components=5, beta_loss='kullback-leibler', random_state=1234, max_iter=2000),
                                       'minibatch-nmf-kl.json', dense_positive)

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

                serialized_dict_model = ml2json.to_dict(coder)
                deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

                ml2json.to_json(coder, model_name)
                deserialized_json_model = ml2json.from_json(model_name)
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

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        model_name = 'truncated-svd.json'
        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_t = deserialized_model.transform(X)

            np.testing.assert_array_almost_equal(expected_t, actual_t)

    def test_truncated_svd_float32_sparse(self):
        rng = np.random.RandomState(1234)
        X_dense = rng.rand(100, 100).astype(np.float32)
        X_dense[:, 2 * np.arange(50)] = 0
        X = csr_matrix(X_dense)
        model = TruncatedSVD(n_components=5, algorithm='arpack', random_state=42)
        model.fit(X)
        expected_t = model.transform(X)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)
        actual_t = deserialized_dict_model.transform(X)
        np.testing.assert_array_almost_equal(expected_t, actual_t)
