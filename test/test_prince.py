# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np
import pandas as pd

# Allow testing of additional optional dependencies
__optionals__ = []
try:
    from prince import PCA, CA, MCA, MFA, FAMD, GPA, PGA
    __optionals__.append('Prince')
except ImportError:
    pass

from src import ml2json


class TestAPI(unittest.TestCase):

    def setUp(self):
        rng = np.random.RandomState(1234)

        self.numerical_data = pd.DataFrame(rng.rand(30, 5), columns=list('abcde'))
        self.contingency_data = pd.DataFrame(rng.randint(1, 20, size=(15, 6)), columns=list('abcdef'))
        self.categorical_data = pd.DataFrame({
            'cat1': rng.choice(list('xyz'), 30),
            'cat2': rng.choice(list('pq'), 30),
            'cat3': rng.choice(list('lmno'), 30),
        })
        self.mixed_data = pd.DataFrame({
            'num1': rng.rand(30),
            'num2': rng.rand(30),
            'cat1': rng.choice(list('xyz'), 30),
            'cat2': rng.choice(list('pq'), 30),
        })
        self.grouped_data = pd.DataFrame(rng.rand(30, 4), columns=['num1', 'num2', 'num3', 'num4'])
        self.shapes = rng.rand(4, 10, 2)
        quaternions = rng.rand(20, 4)
        quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
        self.quaternion_data = pd.DataFrame(quaternions, columns=['qw', 'qx', 'qy', 'qz'])

    def check_model(self, model, model_name, X, method='transform', fit_kwargs=None, call_args=None):
        fit_kwargs = fit_kwargs or {}
        call_args = call_args if call_args is not None else (X,)

        model.fit(X, **fit_kwargs)
        expected = getattr(model, method)(*call_args)

        serialized_dict_model = ml2json.to_dict(model)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        ml2json.to_json(model, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual = getattr(deserialized_model, method)(*call_args)
            if isinstance(expected, pd.DataFrame):
                pd.testing.assert_frame_equal(expected, actual)
            elif isinstance(expected, np.ndarray):
                np.testing.assert_array_almost_equal(expected, actual)
            else:
                self.assertEqual(expected, actual)

    @unittest.skipIf(len(__optionals__) == 0, 'Optional dependencies not installed.')
    def test_pca(self):
        self.check_model(PCA(n_components=3), 'prince_pca.json', self.numerical_data)

    @unittest.skipIf(len(__optionals__) == 0, 'Optional dependencies not installed.')
    def test_ca(self):
        self.check_model(CA(n_components=2), 'prince_ca.json', self.contingency_data, method='row_coordinates')

    @unittest.skipIf(len(__optionals__) == 0, 'Optional dependencies not installed.')
    def test_mca(self):
        self.check_model(MCA(n_components=2), 'prince_mca.json', self.categorical_data, method='row_coordinates')

    @unittest.skipIf(len(__optionals__) == 0, 'Optional dependencies not installed.')
    def test_famd(self):
        self.check_model(FAMD(n_components=2), 'prince_famd.json', self.mixed_data)

    @unittest.skipIf(len(__optionals__) == 0, 'Optional dependencies not installed.')
    def test_mfa(self):
        groups = {'g1': ['num1', 'num2'], 'g2': ['num3', 'num4']}
        self.check_model(MFA(n_components=2), 'prince_mfa.json', self.grouped_data,
                         fit_kwargs={'groups': groups}, call_args=(self.grouped_data,))

    @unittest.skipIf(len(__optionals__) == 0, 'Optional dependencies not installed.')
    def test_gpa(self):
        self.check_model(GPA(), 'prince_gpa.json', self.shapes, call_args=(self.shapes,))

    @unittest.skipIf(len(__optionals__) == 0, 'Optional dependencies not installed.')
    def test_pga(self):
        self.check_model(PGA(), 'prince_pga.json', self.quaternion_data)


if __name__ == '__main__':
    unittest.main()
