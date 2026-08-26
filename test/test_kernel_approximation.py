# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np
from sklearn.datasets import load_iris
from sklearn.kernel_approximation import (AdditiveChi2Sampler, Nystroem, PolynomialCountSketch, RBFSampler,
                                          SkewedChi2Sampler)

# Allow testing of additional optional dependencies
__optionals__ = []
try:
    from sklearn_extra.kernel_approximation import Fastfood
    __optionals__.append('Fastfood')
except:
    pass

from src import ml2json


class TestAPI(unittest.TestCase):

    def setUp(self):
        X, _ = load_iris(return_X_y=True)
        self.X = np.abs(X)

    def check_model(self, model, model_name):
        expected_t = model.fit_transform(self.X)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_t = deserialized_model.transform(self.X)

            if hasattr(expected_t, 'toarray'):
                expected_t = expected_t.toarray()
            if hasattr(actual_t, 'toarray'):
                actual_t = actual_t.toarray()

            np.testing.assert_array_almost_equal(expected_t, actual_t)

    def test_additive_chi2_sampler(self):
        self.check_model(AdditiveChi2Sampler(), 'additive-chi2-sampler.json')

    def test_additive_chi2_sampler_sample_steps(self):
        for sample_steps in [1, 2, 3]:
            self.check_model(AdditiveChi2Sampler(sample_steps=sample_steps), 'additive-chi2-sampler-steps.json')

    def test_additive_chi2_sampler_sample_interval(self):
        self.check_model(AdditiveChi2Sampler(sample_steps=2, sample_interval=0.5),
                         'additive-chi2-sampler-interval.json')

    def test_nystroem(self):
        self.check_model(Nystroem(n_components=10, random_state=1234), 'nystroem.json')

    def test_nystroem_kernels(self):
        for kernel in ['linear', 'poly', 'sigmoid', 'laplacian', 'chi2']:
            self.check_model(Nystroem(kernel=kernel, n_components=10, random_state=1234), 'nystroem-kernel.json')

    def test_nystroem_kernel_params(self):
        self.check_model(Nystroem(kernel='poly', degree=3, coef0=1, gamma=0.5, n_components=10, random_state=1234),
                         'nystroem-kernel-params.json')

    def test_nystroem_precomputed(self):
        from sklearn.metrics.pairwise import pairwise_kernels
        K = pairwise_kernels(self.X, metric='linear')
        # kernel='precomputed' requires n_components == n_samples: Nystroem's
        # basis-sampling step re-slices the kernel to a square submatrix, which
        # only stays square (and thus valid as a precomputed kernel) when every
        # sample is selected as a basis point.
        model = Nystroem(kernel='precomputed', n_components=K.shape[0], random_state=1234)
        expected_t = model.fit_transform(K)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        model_name = 'nystroem-precomputed.json'
        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_t = deserialized_model.transform(K)
            np.testing.assert_array_almost_equal(expected_t, actual_t)

    def test_polynomial_count_sketch(self):
        self.check_model(PolynomialCountSketch(n_components=10, random_state=1234), 'polynomial-count-sketch.json')

    def test_polynomial_count_sketch_params(self):
        for degree, coef0 in [(2, 0), (4, 1.0)]:
            self.check_model(PolynomialCountSketch(degree=degree, coef0=coef0, n_components=10, random_state=1234),
                             'polynomial-count-sketch-params.json')

    def test_rbf_sampler(self):
        self.check_model(RBFSampler(n_components=10, random_state=1234), 'rbf-sampler.json')

    def test_rbf_sampler_gamma(self):
        for gamma in [0.1, 1.0, 10.0]:
            self.check_model(RBFSampler(gamma=gamma, n_components=10, random_state=1234), 'rbf-sampler-gamma.json')

    def test_skewed_chi2_sampler(self):
        self.check_model(SkewedChi2Sampler(n_components=10, random_state=1234), 'skewed-chi2-sampler.json')

    def test_skewed_chi2_sampler_params(self):
        for skewedness in [0.5, 2.0]:
            self.check_model(SkewedChi2Sampler(skewedness=skewedness, n_components=10, random_state=1234),
                             'skewed-chi2-sampler-params.json')

    def test_fastfood(self):
        if 'Fastfood' in __optionals__:
            self.check_model(Fastfood(n_components=10, random_state=1234), 'fastfood.json')

    def test_fastfood_sigma(self):
        if 'Fastfood' in __optionals__:
            for sigma in [0.5, 1.0, 5.0]:
                self.check_model(Fastfood(sigma=sigma, n_components=10, random_state=1234), 'fastfood-sigma.json')

    def test_dtypes(self):
        for dtype in [np.float32, np.float64]:
            X = self.X.astype(dtype)
            model = RBFSampler(n_components=10, random_state=1234)
            expected_t = model.fit_transform(X)

            serialized_dict_model = ml2json.to_dict(model)
            deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

            model_name = 'rbf-sampler-dtype.json'
            ml2json.to_json(model, model_name)
            deserialized_json_model = ml2json.from_json(model_name)
            os.remove(model_name)

            for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
                actual_t = deserialized_model.transform(X)
                np.testing.assert_array_almost_equal(expected_t, actual_t)
