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

try:
    from mlchemad.applicability_domains import LocalOutlierFactorApplicabilityDomain
    __optionals__.append('LocalOutlierFactorApplicabilityDomain')
except ImportError:
    pass

class TestAPI(unittest.TestCase):

    def setUp(self):
        # take only the first 1000 samples
        self.X = fetch_california_housing()['data'][:1000]

    def check_applicability_domain(self, applicability_domain, model_name, X=None, dtype=None):
        X = self.X if X is None else X
        if dtype is not None:
            X = X.astype(dtype)
        for fit in [True, False]:
            if fit:
                applicability_domain.fit(X)
                expected_c = applicability_domain.contains(X)

            serialized_dict_model = ml2json.to_dict(applicability_domain)
            deserialized_dict_model = ml2json.from_dict(serialized_dict_model)

            ml2json.to_json(applicability_domain, model_name)
            deserialized_json_model = ml2json.from_json(model_name)
            os.remove(model_name)

            if fit:
                for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
                    actual_c = deserialized_model.contains(X)

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

        # check with a scalar (int/float) range_ applied to every feature
        model = BoundingBoxApplicabilityDomain(percentiles=None, range_=(0, 20))
        self.check_applicability_domain(model, 'bounding-box-ad.json')

        # check with a non-default percentiles tuple
        model = BoundingBoxApplicabilityDomain(percentiles=(0.05, 0.95))
        self.check_applicability_domain(model, 'bounding-box-ad.json')

        # check with float32 input
        model = BoundingBoxApplicabilityDomain()
        self.check_applicability_domain(model, 'bounding-box-ad.json', dtype=np.float32)

    @unittest.skipIf(len(__optionals__) == 0, 'Optional dependencies not installed.')
    def test_convex_hull_applicability_domain(self):
        model = ConvexHullApplicabilityDomain()
        self.check_applicability_domain(model, 'convex-hull-ad.json')

    @unittest.skipIf(len(__optionals__) == 0, 'Optional dependencies not installed.')
    def test_pca_bounding_box_applicability_domain(self):
        model = PCABoundingBoxApplicabilityDomain()
        self.check_applicability_domain(model, 'pca-bounding-box-ad.json')

        # check with minmax scaling
        model = PCABoundingBoxApplicabilityDomain(scaling='minmax')
        self.check_applicability_domain(model, 'pca-bounding-box-ad.json')

        # check with maxabs scaling
        model = PCABoundingBoxApplicabilityDomain(scaling='maxabs')
        self.check_applicability_domain(model, 'pca-bounding-box-ad.json')

        # check with standard scaling
        model = PCABoundingBoxApplicabilityDomain(scaling='standard')
        self.check_applicability_domain(model, 'pca-bounding-box-ad.json')

        # check with no scaling at all
        model = PCABoundingBoxApplicabilityDomain(scaling=None)
        self.check_applicability_domain(model, 'pca-bounding-box-ad.json')

        # check with a lower explained-variance cutoff, keeping fewer components
        # (exercises a non-square, rank-reduced components_ matrix)
        model = PCABoundingBoxApplicabilityDomain(explained_var=0.5)
        self.check_applicability_domain(model, 'pca-bounding-box-ad.json')

        # check with extra kwargs forwarded to the scaler and the PCA
        model = PCABoundingBoxApplicabilityDomain(scaling='robust',
                                                   scaler_kwargs={'quantile_range': (10.0, 90.0)},
                                                   pca_kwargs={'svd_solver': 'randomized'})
        self.check_applicability_domain(model, 'pca-bounding-box-ad.json')

    @unittest.skipIf(len(__optionals__) == 0, 'Optional dependencies not installed.')
    def test_topkat_applicability_domain(self):
        model = TopKatApplicabilityDomain()
        self.check_applicability_domain(model, 'topkat-ad.json')

        # a constant (zero-variance) feature exercises TopKat's divide-by-zero guard,
        # which leaves the eigen-decomposition of a still-symmetric but degenerate S^T.S matrix
        model = TopKatApplicabilityDomain()
        X_constant_col = np.c_[self.X, np.full(self.X.shape[0], 5.0)]
        self.check_applicability_domain(model, 'topkat-ad.json', X=X_constant_col)

        # float32 input
        model = TopKatApplicabilityDomain()
        self.check_applicability_domain(model, 'topkat-ad.json', dtype=np.float32)

    @unittest.skipIf(len(__optionals__) == 0, 'Optional dependencies not installed.')
    def test_leverage_applicability_domain(self):
        model = LeverageApplicabilityDomain()
        self.check_applicability_domain(model, 'leverage-ad.json')

        # a small, well-conditioned sample exercises a differently-shaped (but still
        # symmetric) var_covar = inv(X^T X) matrix
        model = LeverageApplicabilityDomain()
        self.check_applicability_domain(model, 'leverage-ad.json', X=self.X[:50, :3])

    @unittest.skipIf(len(__optionals__) == 0, 'Optional dependencies not installed.')
    def test_hotelling_t2_applicability_domain(self):
        model = HotellingT2ApplicabilityDomain()
        self.check_applicability_domain(model, 'hotelling-t2-ad.json')

        model = HotellingT2ApplicabilityDomain(significance=0.01)
        self.check_applicability_domain(model, 'hotelling-t2-ad.json')

        model = HotellingT2ApplicabilityDomain(significance=0.20)
        self.check_applicability_domain(model, 'hotelling-t2-ad.json')

    @unittest.skipIf(len(__optionals__) == 0, 'Optional dependencies not installed.')
    def test_kernel_density_applicability_domain(self):
        model = KernelDensityApplicabilityDomain()
        self.check_applicability_domain(model, 'kernel-density-ad.json')

        for kernel in ['tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']:
            model = KernelDensityApplicabilityDomain(kernel=kernel)
            self.check_applicability_domain(model, 'kernel-density-ad.json')

        model = KernelDensityApplicabilityDomain(bandwidth='silverman')
        self.check_applicability_domain(model, 'kernel-density-ad.json')

        model = KernelDensityApplicabilityDomain(bandwidth=0.5)
        self.check_applicability_domain(model, 'kernel-density-ad.json')

        model = KernelDensityApplicabilityDomain(metric='manhattan')
        self.check_applicability_domain(model, 'kernel-density-ad.json')

        model = KernelDensityApplicabilityDomain(threshold=0.0)
        self.check_applicability_domain(model, 'kernel-density-ad.json')

        model = KernelDensityApplicabilityDomain(threshold=1.0)
        self.check_applicability_domain(model, 'kernel-density-ad.json')

    @unittest.skipIf(len(__optionals__) == 0, 'Optional dependencies not installed.')
    def test_isolation_forest_applicability_domain(self):
        model = IsolationForestApplicabilityDomain()
        self.check_applicability_domain(model, 'isolation-forest-ad.json')

        # kwargs forwarded verbatim to sklearn's IsolationForest
        model = IsolationForestApplicabilityDomain(n_estimators=50, contamination=0.05, max_samples=0.5)
        self.check_applicability_domain(model, 'isolation-forest-ad.json')

    @unittest.skipIf(len(__optionals__) == 0, 'Optional dependencies not installed.')
    def test_centroid_distance_applicability_domain(self):
        model = CentroidDistanceApplicabilityDomain()
        self.check_applicability_domain(model, 'centroid-distance-ad.json')

        # explicit percentile threshold instead of the Tukey-fence default
        model = CentroidDistanceApplicabilityDomain(threshold=75)
        self.check_applicability_domain(model, 'centroid-distance-ad.json')

        for dist in ['cityblock', 'chebyshev', 'cosine', 'minkowski', 'sqeuclidean']:
            model = CentroidDistanceApplicabilityDomain(dist=dist)
            self.check_applicability_domain(model, 'centroid-distance-ad.json')

    @unittest.skipIf(len(__optionals__) == 0, 'Optional dependencies not installed.')
    def test_knn_applicability_domain(self):
        model = KNNApplicabilityDomain()
        self.check_applicability_domain(model, 'knn-ad.json')

        model = KNNApplicabilityDomain(scaling=None)
        self.check_applicability_domain(model, 'knn-ad.json')

        for scaling in ['minmax', 'maxabs', 'standard']:
            model = KNNApplicabilityDomain(scaling=scaling)
            self.check_applicability_domain(model, 'knn-ad.json')

        for dist in ['cityblock', 'chebyshev', 'cosine', 'minkowski', 'sqeuclidean']:
            model = KNNApplicabilityDomain(dist=dist)
            self.check_applicability_domain(model, 'knn-ad.json')

        model = KNNApplicabilityDomain(hard_threshold=5.0)
        self.check_applicability_domain(model, 'knn-ad.json')

        model = KNNApplicabilityDomain(alpha=0.8, k=10, njobs=2)
        self.check_applicability_domain(model, 'knn-ad.json')

        model = KNNApplicabilityDomain()
        self.check_applicability_domain(model, 'knn-ad.json', dtype=np.float32)

    @unittest.skipIf(len(__optionals__) == 0, 'Optional dependencies not installed.')
    def test_standardization_approach_applicability_domain(self):
        model = StandardizationApproachApplicabilityDomain()
        self.check_applicability_domain(model, 'standardization-approach-ad.json')

        model = StandardizationApproachApplicabilityDomain()
        self.check_applicability_domain(model, 'standardization-approach-ad.json', dtype=np.float32)

    @unittest.skipIf('LocalOutlierFactorApplicabilityDomain' not in __optionals__,
                     'Optional dependencies not installed.')
    def test_local_outlier_factor_applicability_domain(self):
        # Note: scaling=None isn't exercised here - LocalOutlierFactorApplicabilityDomain._fit
        # unconditionally calls self.scaler.fit_transform(X), unlike its KNN sibling, so it
        # raises AttributeError on a None scaler regardless of serialization; a mlchemad bug,
        # not an ml2json one.
        model = LocalOutlierFactorApplicabilityDomain()
        self.check_applicability_domain(model, 'lof-ad.json')

        model = LocalOutlierFactorApplicabilityDomain(scaling='standard', k=3)
        self.check_applicability_domain(model, 'lof-ad.json')

        for scaling in ['robust', 'maxabs']:
            model = LocalOutlierFactorApplicabilityDomain(scaling=scaling)
            self.check_applicability_domain(model, 'lof-ad.json')

        for dist in ['cityblock', 'chebyshev', 'minkowski']:
            model = LocalOutlierFactorApplicabilityDomain(dist=dist)
            self.check_applicability_domain(model, 'lof-ad.json')

        model = LocalOutlierFactorApplicabilityDomain(contamination=0.2, threshold=0.5)
        self.check_applicability_domain(model, 'lof-ad.json')
