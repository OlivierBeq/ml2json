# -*- coding: utf-8 -*-

import os
import unittest

import shutil
import tempfile
import numpy as np
from joblib import Memory
from sklearn.datasets import make_classification, make_regression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.multioutput import MultiOutputRegressor

from src import ml2json


class TestAPI(unittest.TestCase):

    def setUp(self):
        self.X, self.y = make_classification(n_samples=10_000, random_state=12340)

    def test_pipeline(self):
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC

        X_train, X_test, y_train, _ = train_test_split(self.X, self.y, random_state=1234)

        pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(C=10))])
        pipe.fit(X_train, y_train)
        expected = pipe.predict(X_test)

        serialized_dict_model = ml2json.to_dict(pipe)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        ml2json.to_json(pipe, 'pipe.json')
        deserialized_json_model = ml2json.from_json('pipe.json')
        os.remove('pipe.json')

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual = deserialized_model.predict(X_test)
            np.testing.assert_array_equal(expected, actual)

    def test_feature_union(self):
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        union = FeatureUnion([('pca', PCA(n_components=2, random_state=1234)), ('scaler', StandardScaler())])
        expected = union.fit_transform(self.X)

        serialized_dict_model = ml2json.to_dict(union)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        ml2json.to_json(union, 'feature-union.json')
        deserialized_json_model = ml2json.from_json('feature-union.json')
        os.remove('feature-union.json')

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual = deserialized_model.transform(self.X)
            np.testing.assert_array_almost_equal(expected, actual)

    def test_feature_union_transformer_weights(self):
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        union = FeatureUnion([('pca', PCA(n_components=2, random_state=1234)), ('scaler', StandardScaler())],
                             transformer_weights={'pca': 0.3, 'scaler': 1.7})
        expected = union.fit_transform(self.X)

        serialized_dict_model = ml2json.to_dict(union)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        ml2json.to_json(union, 'feature-union-weights.json')
        deserialized_json_model = ml2json.from_json('feature-union-weights.json')
        os.remove('feature-union-weights.json')

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual = deserialized_model.transform(self.X)
            np.testing.assert_array_almost_equal(expected, actual)

    def test_pipeline_memory(self):
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC

        cache_dir = tempfile.mkdtemp()
        try:
            memory = Memory(location=cache_dir, verbose=0)
            pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(C=10))], memory=memory)
            pipe.fit(self.X, self.y)
            expected = pipe.predict(self.X)

            serialized_dict_model = ml2json.to_dict(pipe)
            deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

            ml2json.to_json(pipe, 'pipe-memory.json')
            deserialized_json_model = ml2json.from_json('pipe-memory.json')
            os.remove('pipe-memory.json')

            for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
                actual = deserialized_model.predict(self.X)
                np.testing.assert_array_equal(expected, actual)
                self.assertIsInstance(deserialized_model.memory, Memory)
        finally:
            shutil.rmtree(cache_dir, ignore_errors=True)

    def test_pipeline_passthrough_step(self):
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC

        pipe = Pipeline([('scaler', StandardScaler()), ('passthrough', 'passthrough'), ('svc', SVC(C=10))])
        pipe.fit(self.X, self.y)
        expected = pipe.predict(self.X)

        serialized_dict_model = ml2json.to_dict(pipe)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        ml2json.to_json(pipe, 'pipe-passthrough.json')
        deserialized_json_model = ml2json.from_json('pipe-passthrough.json')
        os.remove('pipe-passthrough.json')

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual = deserialized_model.predict(self.X)
            np.testing.assert_array_equal(expected, actual)

    def test_pipeline_with_nested_column_transformer(self):
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        from sklearn.svm import SVC

        ct = ColumnTransformer([('scale', StandardScaler(), [0, 1]),
                                ('minmax', MinMaxScaler(), [2, 3])],
                               remainder='passthrough')
        pipe = Pipeline([('ct', ct), ('svc', SVC(C=10))])
        pipe.fit(self.X, self.y)
        expected = pipe.predict(self.X)

        serialized_dict_model = ml2json.to_dict(pipe)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        ml2json.to_json(pipe, 'pipe-nested-ct.json')
        deserialized_json_model = ml2json.from_json('pipe-nested-ct.json')
        os.remove('pipe-nested-ct.json')

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual = deserialized_model.predict(self.X)
            np.testing.assert_array_equal(expected, actual)

    def test_pipeline_with_multioutput_regressor(self):
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        X_reg, y_reg = make_regression(n_samples=200, n_features=5, n_targets=2, random_state=0)
        pipe = Pipeline([('scaler', StandardScaler()), ('multi', MultiOutputRegressor(Ridge()))])
        pipe.fit(X_reg, y_reg)
        expected = pipe.predict(X_reg)

        serialized_dict_model = ml2json.to_dict(pipe)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        ml2json.to_json(pipe, 'pipe-multioutput.json')
        deserialized_json_model = ml2json.from_json('pipe-multioutput.json')
        os.remove('pipe-multioutput.json')

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual = deserialized_model.predict(X_reg)
            np.testing.assert_array_almost_equal(expected, actual)
