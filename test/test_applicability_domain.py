import os
import unittest

import numpy as np
from sklearn.datasets import fetch_california_housing
from mlchemad.applicability_domains import (BoundingBoxApplicabilityDomain,
                                            ConvexHullApplicabilityDomain,
                                            PCABoundingBoxApplicabilityDomain,
                                            TopKatApplicabilityDomain,
                                            LeverageApplicabilityDomain,
                                            HotellingT2ApplicabilityDomain,
                                            KernelDensityApplicabilityDomain,
                                            IsolationForestApplicabilityDomain,
                                            CentroidDistanceApplicabilityDomain,
                                            KNNApplicabilityDomain,
                                            StandardizationApproachApplicabilityDomain)

from src import ml2json

class TestAPI(unittest.TestCase):

    def setUp(self):
        self.X = fetch_california_housing()['data']
        
    def check_applicability_domain(self, applicability_domain, model_name):
        applicability_domain.fit(self.X)
        expected_c = applicability_domain.contains(self.X)

        serialized_dict_model = ml2json.to_dict(applicability_domain)
        deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

        ml2json.to_json(applicability_domain, model_name)
        deserialized_json_model = ml2json.from_json(model_name)
        os.remove(model_name)

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_c = deserialized_model.contains(self.X)

            np.testing.assert_array_equal(expected_c, actual_c)
            
    def test_bounding_box_applicability_domain(self):
        model = BoundingBoxApplicabilityDomain()
        self.check_applicability_domain(model, 'bounding-box-ad.json')