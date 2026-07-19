# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import (LabelEncoder, LabelBinarizer, MultiLabelBinarizer,
                                   MinMaxScaler, StandardScaler, KernelCenterer,
                                   OneHotEncoder, RobustScaler, MaxAbsScaler,
                                   OrdinalEncoder, Normalizer, Binarizer, PowerTransformer,
                                   QuantileTransformer, KBinsDiscretizer, PolynomialFeatures,
                                   SplineTransformer, TargetEncoder)
from sklearn.metrics.pairwise import pairwise_kernels

from src import ml2json


class TestAPI(unittest.TestCase):

    def setUp(self):
        self.data = [
            {'action', 'drama', 'fantasy'},
            {'comedy', 'horror'},
            {'comedy', 'romance'},
            {'horror'},
            {'mystery', 'thriller'},
            {'sci-fi', 'thriller'},
        ]
        self.labels = np.array([
            [1, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 1],
        ])

        self.simple_fit_data = np.array([[0, 1, 1], [1, 0, 0]])
        self.simple_test_data = [0, 1, 2, 1]
        self.simple_test_labels = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]])

        self.X = fetch_california_housing(as_frame=True)['data']
        self.kernel_X = pairwise_kernels(self.X[:100], metric="linear", filter_params=True, degree=3, coef0=1)

    def check_model(self, model, model_name, data, labels):
        expected_ft = model.fit_transform(data)
        expected_t = model.transform(data)
        expected_it = model.inverse_transform(labels)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_t = deserialized_model.transform(data)
            actual_ft = deserialized_model.fit_transform(data)
            actual_it = deserialized_model.inverse_transform(labels)

            if hasattr(model, 'sparse_output') and model.sparse_output:
                np.testing.assert_array_equal(expected_t.indptr, actual_t.indptr)
                np.testing.assert_array_equal(expected_t.indices, actual_t.indices)
                np.testing.assert_array_equal(expected_t.data, actual_t.data)
                np.testing.assert_array_equal(expected_ft.indptr, actual_ft.indptr)
                np.testing.assert_array_equal(expected_ft.indices, actual_ft.indices)
                np.testing.assert_array_equal(expected_ft.data, actual_ft.data)
                if isinstance(actual_it, np.ndarray):
                    np.testing.assert_array_equal(expected_it, actual_it)
                else:
                    self.assertEqual(expected_it, actual_it)
            else:
                np.testing.assert_array_equal(expected_t, actual_t)
                np.testing.assert_array_equal(expected_ft, actual_ft)
                if isinstance(actual_it, np.ndarray):
                    np.testing.assert_array_equal(expected_it, actual_it)
                else:
                    self.assertEqual(expected_it, actual_it)

    def test_label_encoder(self):
        self.check_model(LabelEncoder(), 'label-encoder.json', ["paris", "paris", "tokyo", "amsterdam"], [0, 0, 1, 2])

    def test_label_binarizer(self):
        self.check_model(LabelBinarizer(), 'label-binarizer.json', self.simple_test_data, self.simple_test_labels)
        self.check_model(LabelBinarizer(sparse_output=True), 'label-binarizer.json', self.simple_test_data, self.simple_test_labels)

    def test_multilabel_binarizer(self):
        self.check_model(MultiLabelBinarizer(), 'multilabel-binarizer.json', self.data, self.labels)
        self.check_model(MultiLabelBinarizer(sparse_output=True), 'multilabel-binarizer.json', self.data, self.labels)

    def check_scaler(self, scaler, model_name):
        expected_ft = scaler.fit_transform(self.X)
        expected_t = scaler.transform(self.X)
        expected_it = scaler.inverse_transform(expected_t)

        serialized_dict_model = ml2json.to_dict(scaler)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        ml2json.to_json(scaler, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_t = deserialized_model.transform(self.X)
            actual_ft = deserialized_model.fit_transform(self.X)
            actual_it = deserialized_model.inverse_transform(actual_t)

            np.testing.assert_array_equal(expected_t, actual_t)
            np.testing.assert_array_equal(expected_ft, actual_ft)
            np.testing.assert_array_equal(expected_it, actual_it)

    def test_minmax_scaler(self):
        self.check_scaler(MinMaxScaler(), 'minmax-scaler.json')
        self.check_scaler(MinMaxScaler(feature_range=(10, 20)), 'minmax-scaler.json')
        self.check_scaler(MinMaxScaler(clip=True), 'minmax-scaler.json')

    def test_standard_scaler(self):
        self.check_scaler(StandardScaler(), 'standard-scaler.json')
        self.check_scaler(StandardScaler(with_mean=False), 'standard-scaler.json')
        self.check_scaler(StandardScaler(with_std=False), 'standard-scaler.json')
        self.check_scaler(StandardScaler(with_mean=False, with_std=False), 'standard-scaler.json')

    def test_robust_scaler(self):
        self.check_scaler(RobustScaler(), 'robust-scaler.json')
        self.check_scaler(RobustScaler(with_centering=False), 'robust-scaler.json')
        self.check_scaler(RobustScaler(with_scaling=False), 'robust-scaler.json')
        self.check_scaler(RobustScaler(with_centering=False, with_scaling=False), 'robust-scaler.json')
        
    def test_maxabs_scaler(self):
        self.check_scaler(MaxAbsScaler(), 'maxabs-scaler.json')

    def check_centerer(self, centerer, model_name):
        expected_ft = centerer.fit_transform(self.kernel_X)
        expected_t = centerer.transform(self.kernel_X)

        serialized_dict_model = ml2json.to_dict(centerer)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        ml2json.to_json(centerer, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_t = deserialized_model.transform(self.kernel_X)
            actual_ft = deserialized_model.fit_transform(self.kernel_X)

            np.testing.assert_array_equal(expected_t, actual_t)
            np.testing.assert_array_equal(expected_ft, actual_ft)

    def test_kernel_centerer(self):
        self.check_centerer(KernelCenterer(), 'kernel-centerer.json')

    def test_onehot_encoder(self):
        model  = OneHotEncoder(handle_unknown='ignore')
        model.fit([['Male', 1], ['Female', 3], ['Female', 2]])
        expected_t = model.transform([['Female', 1], ['Male', 4]]).toarray()
        expected_it = model.inverse_transform(expected_t)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        model_name = 'onehot-encoder.json'
        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_t = deserialized_model.transform([['Female', 1], ['Male', 4]]).toarray()
            actual_it = deserialized_model.inverse_transform(actual_t)

            np.testing.assert_array_equal(expected_t, actual_t)
            np.testing.assert_array_equal(expected_it, actual_it)

    def test_onehot_encoder_variants(self):
        X_train = [['Male', 1], ['Female', 3], ['Female', 2]]
        for model in [OneHotEncoder(handle_unknown='ignore', drop='first'),
                      OneHotEncoder(handle_unknown='ignore', drop='if_binary'),
                      OneHotEncoder(handle_unknown='ignore', sparse_output=False),
                      OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=2),
                      OneHotEncoder(handle_unknown='infrequent_if_exist', max_categories=1)]:
            model.fit(X_train)
            expected_t = model.transform([['Female', 1], ['Male', 4]])
            if hasattr(expected_t, 'toarray'):
                expected_t = expected_t.toarray()

            serialized_dict_model = ml2json.to_dict(model)
            deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

            model_name = 'onehot-encoder-variant.json'
            ml2json.to_json(model, model_name)
            deserialized_json_model = ml2json.from_json(model_name)
            os.remove(model_name)

            for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
                actual_t = deserialized_model.transform([['Female', 1], ['Male', 4]])
                if hasattr(actual_t, 'toarray'):
                    actual_t = actual_t.toarray()
                np.testing.assert_array_equal(expected_t, actual_t)

    def test_onehot_encoder_float32(self):
        X_train = np.array([[0, 1], [1, 0], [0, 0]], dtype=np.float32)
        model = OneHotEncoder(handle_unknown='ignore')
        model.fit(X_train)
        expected_t = model.transform(X_train).toarray()

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        for deserialized_model in [deserialized_dict_model]:
            actual_t = deserialized_model.transform(X_train).toarray()
            np.testing.assert_array_equal(expected_t, actual_t)

    def test_ordinal_encoder(self):
        X_train = np.array([["a"] * 5 + ["b"] * 20 + ["c"] * 10 + ["d"] * 3 + [np.nan]],dtype=object).T
        model = OrdinalEncoder()
        model.fit(X_train)
        X_test = np.array([["a"], ["b"], ["c"], ["d"]], dtype=object)

        expected_t = model.transform(X_test)
        expected_it = model.inverse_transform(expected_t)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        model_name = 'onehot-encoder.json'
        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_t = deserialized_model.transform(X_test)
            actual_it = deserialized_model.inverse_transform(actual_t)

            np.testing.assert_array_equal(expected_t, actual_t)
            np.testing.assert_array_equal(expected_it, actual_it)

        model = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=3,
                               max_categories=3, encoded_missing_value=4)
        model.fit(X_train)
        X_test = np.array([["a"], ["b"], ["c"], ["d"], ["e"]], dtype=object)

        expected_t = model.transform(X_test)
        expected_it = model.inverse_transform(expected_t)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        model_name = 'onehot-encoder.json'
        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_t = deserialized_model.transform(X_test)
            np.testing.assert_array_equal(expected_t, actual_t)

    def test_ordinal_encoder_min_frequency(self):
        X_train = np.array([["a"] * 5 + ["b"] * 20 + ["c"] * 10 + ["d"] * 3], dtype=object).T
        model = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan, min_frequency=4)
        model.fit(X_train)
        X_test = np.array([["a"], ["b"], ["c"], ["d"]], dtype=object)

        expected_t = model.transform(X_test)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        model_name = 'ordinal-encoder-min-freq.json'
        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_t = deserialized_model.transform(X_test)
            np.testing.assert_array_equal(expected_t, actual_t)

    def test_normalizer(self):
        scaler = Normalizer()

        expected_ft = scaler.fit_transform(self.X)
        expected_t = scaler.transform(self.X)

        serialized_dict_model = ml2json.to_dict(scaler)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        ml2json.to_json(scaler, 'normalizer.json')
        deserialized_json_model = ml2json.from_json('normalizer.json')
        os.remove('normalizer.json')

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_t = deserialized_model.transform(self.X)
            actual_ft = deserialized_model.fit_transform(self.X)

            np.testing.assert_array_equal(expected_t, actual_t)
            np.testing.assert_array_equal(expected_ft, actual_ft)

    def check_transformer(self, transformer, model_name, data=None):
        data = self.X if data is None else data
        expected_ft = transformer.fit_transform(data)
        expected_t = transformer.transform(data)

        serialized_dict_model = ml2json.to_dict(transformer)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        ml2json.to_json(transformer, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_t = deserialized_model.transform(data)

            if hasattr(expected_t, 'toarray'):
                expected_t = expected_t.toarray()
            if hasattr(actual_t, 'toarray'):
                actual_t = actual_t.toarray()

            np.testing.assert_array_almost_equal(expected_t, actual_t)

    def test_binarizer(self):
        self.check_transformer(Binarizer(), 'binarizer.json')

    def test_power_transformer(self):
        self.check_transformer(PowerTransformer(), 'power-transformer.json', data=self.X + 1.0)

    def test_power_transformer_boxcox(self):
        positive_X = np.abs(self.X) + 1.0
        self.check_transformer(PowerTransformer(method='box-cox'), 'power-transformer-boxcox.json', data=positive_X)
        self.check_transformer(PowerTransformer(method='box-cox', standardize=False), 'power-transformer-boxcox.json',
                               data=positive_X)
        self.check_transformer(PowerTransformer(method='yeo-johnson', standardize=False), 'power-transformer-yj.json',
                               data=self.X + 1.0)

    def test_quantile_transformer(self):
        self.check_transformer(QuantileTransformer(n_quantiles=100), 'quantile-transformer.json')

    def test_quantile_transformer_variants(self):
        self.check_transformer(QuantileTransformer(n_quantiles=100, output_distribution='normal'),
                               'quantile-transformer-normal.json')
        # n_quantiles larger than n_samples (but within the subsample cap): sklearn clips it to n_samples internally.
        self.check_transformer(QuantileTransformer(n_quantiles=1000), 'quantile-transformer-many-q.json',
                               data=self.X[:50])

    def test_kbins_discretizer(self):
        self.check_transformer(KBinsDiscretizer(n_bins=3, encode='ordinal'), 'kbins-discretizer.json')
        self.check_transformer(KBinsDiscretizer(n_bins=3, encode='onehot-dense'), 'kbins-discretizer.json')

    def test_kbins_discretizer_variants(self):
        self.check_transformer(KBinsDiscretizer(n_bins=3, encode='onehot', strategy='uniform'),
                               'kbins-discretizer-uniform.json')
        self.check_transformer(KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='kmeans'),
                               'kbins-discretizer-kmeans.json')

    def test_polynomial_features(self):
        self.check_transformer(PolynomialFeatures(degree=2), 'polynomial-features.json')

    def test_polynomial_features_variants(self):
        self.check_transformer(PolynomialFeatures(degree=3, interaction_only=True), 'polynomial-features-io.json')
        self.check_transformer(PolynomialFeatures(degree=2, include_bias=False), 'polynomial-features-nobias.json')

    def test_spline_transformer(self):
        self.check_transformer(SplineTransformer(), 'spline-transformer.json')

    def test_spline_transformer_variants(self):
        self.check_transformer(SplineTransformer(knots='quantile'), 'spline-transformer-quantile.json')
        self.check_transformer(SplineTransformer(extrapolation='linear'), 'spline-transformer-linear.json')
        self.check_transformer(SplineTransformer(extrapolation='periodic'), 'spline-transformer-periodic.json')
        self.check_transformer(SplineTransformer(include_bias=False), 'spline-transformer-nobias.json')

    def test_robust_scaler_quantile_range(self):
        self.check_scaler(RobustScaler(quantile_range=(10.0, 90.0)), 'robust-scaler-qrange.json')
        self.check_scaler(RobustScaler(unit_variance=True), 'robust-scaler-unitvar.json')

    def test_target_encoder(self):
        X_train = np.array([["a"] * 5 + ["b"] * 20 + ["c"] * 10 + ["d"] * 3], dtype=object).T
        y_train = np.array([0, 1] * 19)
        model = TargetEncoder(target_type='binary')
        expected_ft = model.fit_transform(X_train, y_train)
        expected_t = model.transform(X_train)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        model_name = 'target-encoder.json'
        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_t = deserialized_model.transform(X_train)
            np.testing.assert_array_almost_equal(expected_t, actual_t)

    def test_target_encoder_variants(self):
        X_train = np.array([["a"] * 5 + ["b"] * 20 + ["c"] * 10 + ["d"] * 3], dtype=object).T
        y_train_cont = np.linspace(0, 1, 38)
        for model in [TargetEncoder(target_type='continuous', smooth=0.0),
                      TargetEncoder(target_type='continuous', smooth=5.0),
                      TargetEncoder(target_type='binary', cv=3)]:
            y_train = y_train_cont if model.target_type == 'continuous' else np.array([0, 1] * 19)
            model.fit(X_train, y_train)
            expected_t = model.transform(X_train)

            serialized_dict_model = ml2json.to_dict(model)
            deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

            model_name = 'target-encoder-variant.json'
            ml2json.to_json(model, model_name)
            deserialized_json_model = ml2json.from_json(model_name)
            os.remove(model_name)

            for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
                actual_t = deserialized_model.transform(X_train)
                np.testing.assert_array_almost_equal(expected_t, actual_t)
