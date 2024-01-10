import os
import unittest

import numpy as np
from sklearn.datasets import fetch_california_housing
from src import ml2json

# Allow additional dependencies to be optional
__optionals__ = []
try:
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
    __optionals__.extend(['BoundingBoxApplicabilityDomain',
                            'ConvexHullApplicabilityDomain',
                            'PCABoundingBoxApplicabilityDomain',
                            'TopKatApplicabilityDomain',
                            'LeverageApplicabilityDomain',
                            'HotellingT2ApplicabilityDomain',
                            'KernelDensityApplicabilityDomain',
                            'IsolationForestApplicabilityDomain',
                            'CentroidDistanceApplicabilityDomain',
                            'KNNApplicabilityDomain',
                            'StandardizationApproachApplicabilityDomain'])
except ImportError:
    pass

class TestAPI(unittest.TestCase):

    def setUp(self):
        # take only the first 1000 samples
        self.X = fetch_california_housing()['data'][:1000]
        
    def check_applicability_domain(self, applicability_domain, model_name):
        for fit in [True, False]:
            if fit:
                applicability_domain.fit(self.X)
            expected_c = applicability_domain.contains(self.X)

            serialized_dict_model = ml2json.to_dict(applicability_domain)
            deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

            ml2json.to_json(applicability_domain, model_name)
            deserialized_json_model = ml2json.from_json(model_name)
            os.remove(model_name)

            if fit:
                for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
                    actual_c = deserialized_model.contains(self.X)

                    np.testing.assert_array_equal(expected_c, actual_c)
    
    @unittest.skipIf(len(__optionals__) == 0, 'Optional dependencies not installed.')
    def test_bounding_box_applicability_domain(self):
        model = BoundingBoxApplicabilityDomain()
        self.check_applicability_domain(model, 'bounding-box-ad.json')
        
        # check with extremum values per feature
        model = BoundingBoxApplicabilityDomain(
            range_=(list(self.X.min(axis=0)),
                    list(self.X.max(axis=0))))
        self.check_applicability_domain(model, 'bounding-box-ad.json')
        
        # check with percentiles is None
        model = BoundingBoxApplicabilityDomain(percentiles=None)
        self.check_applicability_domain(model, 'bounding-box-ad.json')
    
    @unittest.skipIf(len(__optionals__) == 0, 'Optional dependencies not installed.')
    def test_convex_hull_applicability_domain(self):
        model = ConvexHullApplicabilityDomain()
        self.check_applicability_domain(model, 'convex-hull-ad.json')
    
    @unittest.skipIf(len(__optionals__) == 0, 'Optional dependencies not installed.')   
    def test_pca_bounding_box_applicability_domain(self):
        model = PCABoundingBoxApplicabilityDomain()
        self.check_applicability_domain(model, 'pca-bounding-box-ad.json')
    
    @unittest.skipIf(len(__optionals__) == 0, 'Optional dependencies not installed.')   
    def test_topkat_applicability_domain(self):
        model = TopKatApplicabilityDomain()
        self.check_applicability_domain(model, 'topkat-ad.json')
    
    @unittest.skipIf(len(__optionals__) == 0, 'Optional dependencies not installed.')   
    def test_leverage_applicability_domain(self):
        model = LeverageApplicabilityDomain()
        self.check_applicability_domain(model, 'leverage-ad.json')
    
    @unittest.skipIf(len(__optionals__) == 0, 'Optional dependencies not installed.')
    def test_hotelling_t2_applicability_domain(self):
        model = HotellingT2ApplicabilityDomain()
        self.check_applicability_domain(model, 'hotelling-t2-ad.json')

    @unittest.skipIf(len(__optionals__) == 0, 'Optional dependencies not installed.')
    def test_kernel_density_applicability_domain(self):
        model = KernelDensityApplicabilityDomain()
        self.check_applicability_domain(model, 'kernel-density-ad.json')
    
    @unittest.skipIf(len(__optionals__) == 0, 'Optional dependencies not installed.')
    def test_isolation_forest_applicability_domain(self):
        model = IsolationForestApplicabilityDomain()
        self.check_applicability_domain(model, 'isolation-forest-ad.json')
    
    @unittest.skipIf(len(__optionals__) == 0, 'Optional dependencies not installed.') 
    def test_centroid_distance_applicability_domain(self):
        model = CentroidDistanceApplicabilityDomain()
        self.check_applicability_domain(model, 'centroid-distance-ad.json')
    
    @unittest.skipIf(len(__optionals__) == 0, 'Optional dependencies not installed.') 
    def test_knn_applicability_domain(self):
        model = KNNApplicabilityDomain()
        self.check_applicability_domain(model, 'knn-ad.json')
        
        model = KNNApplicabilityDomain(scaling=None)
        self.check_applicability_domain(model, 'knn-ad.json')
    
    @unittest.skipIf(len(__optionals__) == 0, 'Optional dependencies not installed.')
    def test_standardization_approach_applicability_domain(self):
        model = StandardizationApproachApplicabilityDomain()
        self.check_applicability_domain(model, 'standardization-approach-ad.json')