# -*- coding: utf-8 -*-

import re
import sys
import json
import inspect
import importlib
import importlib.util
import warnings
from typing import Dict

from sklearn import svm, discriminant_analysis, dummy
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor
from sklearn.ensemble import (AdaBoostClassifier, AdaBoostRegressor, BaggingClassifier, BaggingRegressor,
                              ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier,
                              GradientBoostingRegressor, IsolationForest, RandomForestClassifier,
                              RandomForestRegressor, StackingClassifier, StackingRegressor, VotingClassifier,
                              VotingRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomTreesEmbedding)
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB, ComplementNB
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import (LabelEncoder, LabelBinarizer, MultiLabelBinarizer,
                                   MinMaxScaler, StandardScaler, KernelCenterer,
                                   OneHotEncoder, RobustScaler, MaxAbsScaler,
                                   OrdinalEncoder, Normalizer)
from sklearn.svm import SVR
from sklearn.cluster import (AffinityPropagation, AgglomerativeClustering,
                             Birch, DBSCAN, FeatureAgglomeration, KMeans,
                             BisectingKMeans, MiniBatchKMeans, MeanShift, OPTICS,
                             SpectralClustering, SpectralBiclustering, SpectralCoclustering)
from sklearn.cross_decomposition import (CCA, PLSCanonical,
                                         PLSRegression, PLSSVD)
from sklearn.decomposition import (PCA, KernelPCA, DictionaryLearning, FactorAnalysis, FastICA, IncrementalPCA,
                                   LatentDirichletAllocation, MiniBatchDictionaryLearning, MiniBatchSparsePCA, NMF,
                                   MiniBatchNMF, SparsePCA, SparseCoder, TruncatedSVD)
from sklearn.manifold import (Isomap, LocallyLinearEmbedding,
                              MDS, SpectralEmbedding, TSNE)
from sklearn.neighbors import NearestNeighbors, KDTree, KNeighborsClassifier, KNeighborsRegressor, KernelDensity
from sklearn.pipeline import FeatureUnion, Pipeline

from . import classification as clf
from . import regression as reg
from . import feature_extraction as ext
from . import preprocessing as pre
from . import cluster as clus
from . import decomposition as dec
from . import manifold as man
from . import neighbors as nei
from . import cross_decomposition as crdec
from . import applicability_domain as ad
from . import over_undersampling as ous
from . import pipeline as ppl
from numpy.random import RandomState

from . import _base
from .utils import is_model_fitted, recursive_inspection
from .utils.random_state import serialize_random_state, deserialize_random_state

# Make additional dependencies optional
if 'XGBRegressor' in reg.__optionals__:
    from xgboost import XGBRegressor, XGBRFRegressor, XGBRanker, XGBClassifier, XGBRFClassifier
if 'LGBMRegressor' in reg.__optionals__:
    from lightgbm import LGBMRegressor, LGBMRanker, LGBMClassifier
if 'CatBoostRegressor' in reg.__optionals__:
    from catboost import CatBoostRegressor, CatBoostRanker, Pool, CatBoostClassifier
else:
    from typing import TypeVar
    Pool = TypeVar('Pool')
if 'KModes' in clus.__optionals__:
    from kmodes.kmodes import KModes
    from kmodes.kprototypes import KPrototypes
if 'HDBSCAN' in clus.__optionals__:
    from hdbscan import HDBSCAN
if 'NNDescent' in nei.__optionals__:
    from pynndescent import NNDescent
if 'UMAP' in man.__optionals__:
    from umap import UMAP
if 'OpenTSNE' in man.__optionals__:
    from openTSNE import (TSNE as OpenTSNE, TSNEEmbedding as OpenTSNEEmbedding,
                          PartialTSNEEmbedding as OpenPartialTSNEEmbedding)
    from openTSNE.sklearn import TSNE as OpenTSNEsklearn
if 'BoundingBoxApplicabilityDomain' in ad.__optionals__:
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
if 'imblearn' in ous.__optionals__:
    from imblearn.under_sampling import (ClusterCentroids, CondensedNearestNeighbour, EditedNearestNeighbours,
                                         RepeatedEditedNearestNeighbours, AllKNN, InstanceHardnessThreshold,
                                         NearMiss, NeighbourhoodCleaningRule, OneSidedSelection,
                                         RandomUnderSampler, TomekLinks)
    from imblearn.over_sampling import (RandomOverSampler, SMOTE, SMOTEN, SMOTENC, ADASYN, BorderlineSMOTE,
                                        KMeansSMOTE, SVMSMOTE)
    from imblearn.combine import SMOTEENN, SMOTETomek
    from imblearn.ensemble import (EasyEnsembleClassifier, RUSBoostClassifier, BalancedBaggingClassifier,
                                   BalancedRandomForestClassifier)
if 'imblearn' in ppl.__optionals__:
    from imblearn.pipeline import Pipeline as ImblearnPipeline


# ---------------------------------------------------------------------------
# Registry-driven dispatch
#
# Every supported class is registered once as (class, serialize_fn,
# deserialize_fn). The 'meta' tag used to identify it on the wire is derived
# automatically from the class itself - its source library (top-level import
# package) plus its kebab-cased class name - rather than hand-picked, so two
# classes that happen to share a name across different libraries (e.g. a
# future addition clashing with sklearn's TSNE) can never collide: the
# library prefix keeps them apart.
# ---------------------------------------------------------------------------

def _kebab(name: str) -> str:
    """CamelCase/acronym class name -> kebab-case (UMAP -> umap, XGBClassifier -> xgb-classifier)."""
    s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1-\2', name)
    s2 = re.sub(r'([a-z0-9])([A-Z])', r'\1-\2', s1)
    return s2.lower()


def _meta_for(cls: type) -> str:
    library = inspect.getmodule(cls).__name__.partition('.')[0]
    return f'{library}.{_kebab(cls.__name__)}'


_REGISTRY = [
    # Classification
    (LogisticRegression, clf.serialize_logistic_regression, clf.deserialize_logistic_regression),
    (BernoulliNB, clf.serialize_bernoulli_nb, clf.deserialize_bernoulli_nb),
    (GaussianNB, clf.serialize_gaussian_nb, clf.deserialize_gaussian_nb),
    (MultinomialNB, clf.serialize_multinomial_nb, clf.deserialize_multinomial_nb),
    (ComplementNB, clf.serialize_complement_nb, clf.deserialize_complement_nb),
    (discriminant_analysis.LinearDiscriminantAnalysis, clf.serialize_lda, clf.deserialize_lda),
    (discriminant_analysis.QuadraticDiscriminantAnalysis, clf.serialize_qda, clf.deserialize_qda),
    (svm.SVC, clf.serialize_svm, clf.deserialize_svm),
    (Perceptron, clf.serialize_perceptron, clf.deserialize_perceptron),
    (DecisionTreeClassifier, clf.serialize_decision_tree, clf.deserialize_decision_tree),
    (GradientBoostingClassifier, clf.serialize_gradient_boosting, clf.deserialize_gradient_boosting),
    (RandomForestClassifier, clf.serialize_random_forest, clf.deserialize_random_forest),
    (MLPClassifier, clf.serialize_mlp, clf.deserialize_mlp),
    (AdaBoostClassifier, clf.serialize_adaboost_classifier, clf.deserialize_adaboost_classifier),
    (BaggingClassifier, clf.serialize_bagging_classifier, clf.deserialize_bagging_classifier),
    (ExtraTreeClassifier, clf.serialize_extra_tree_classifier, clf.deserialize_extra_tree_classifier),
    (ExtraTreesClassifier, clf.serialize_extratrees_classifier, clf.deserialize_extratrees_classifier),
    (IsolationForest, clf.serialize_isolation_forest, clf.deserialize_isolation_forest),
    (RandomTreesEmbedding, clf.serialize_random_trees_embedding, clf.deserialize_random_trees_embedding),
    (KNeighborsClassifier, clf.serialize_nearest_neighbour_classifier, clf.deserialize_nearest_neighbour_classifier),
    (StackingClassifier, clf.serialize_stacking_classifier, clf.deserialize_stacking_classifier),
    (VotingClassifier, clf.serialize_voting_classifier, clf.deserialize_voting_classifier),

    # Regression
    (LinearRegression, reg.serialize_linear_regressor, reg.deserialize_linear_regressor),
    (Lasso, reg.serialize_lasso_regressor, reg.deserialize_lasso_regressor),
    (ElasticNet, reg.serialize_elastic_regressor, reg.deserialize_elastic_regressor),
    (Ridge, reg.serialize_ridge_regressor, reg.deserialize_ridge_regressor),
    (SVR, reg.serialize_svr, reg.deserialize_svr),
    (ExtraTreeRegressor, reg.serialize_extra_tree_regressor, reg.deserialize_extra_tree_regressor),
    (DecisionTreeRegressor, reg.serialize_decision_tree_regressor, reg.deserialize_decision_tree_regressor),
    (GradientBoostingRegressor, reg.serialize_gradient_boosting_regressor, reg.deserialize_gradient_boosting_regressor),
    (RandomForestRegressor, reg.serialize_random_forest_regressor, reg.deserialize_random_forest_regressor),
    (ExtraTreesRegressor, reg.serialize_extratrees_regressor, reg.deserialize_extratrees_regressor),
    (MLPRegressor, reg.serialize_mlp_regressor, reg.deserialize_mlp_regressor),
    (AdaBoostRegressor, reg.serialize_adaboost_regressor, reg.deserialize_adaboost_regressor),
    (BaggingRegressor, reg.serialize_bagging_regressor, reg.deserialize_bagging_regressor),
    (KNeighborsRegressor, reg.serialize_nearest_neighbour_regressor, reg.deserialize_nearest_neighbour_regressor),
    (StackingRegressor, reg.serialize_stacking_regressor, reg.deserialize_stacking_regressor),
    (VotingRegressor, reg.serialize_voting_regressor, reg.deserialize_voting_regressor),

    # Clustering
    (FeatureAgglomeration, clus.serialize_feature_agglomeration, clus.deserialize_feature_agglomeration),
    (AffinityPropagation, clus.serialize_affinity_propagation, clus.deserialize_affinity_propagation),
    (AgglomerativeClustering, clus.serialize_agglomerative_clustering, clus.deserialize_agglomerative_clustering),
    (DBSCAN, clus.serialize_dbscan, clus.deserialize_dbscan),
    (MeanShift, clus.serialize_meanshift, clus.deserialize_meanshift),
    (BisectingKMeans, clus.serialize_bisecting_kmeans, clus.deserialize_bisecting_kmeans),
    (MiniBatchKMeans, clus.serialize_minibatch_kmeans, clus.deserialize_minibatch_kmeans),
    (KMeans, clus.serialize_kmeans, clus.deserialize_kmeans),
    (OPTICS, clus.serialize_optics, clus.deserialize_optics),
    (SpectralClustering, clus.serialize_spectral_clustering, clus.deserialize_spectral_clustering),
    (SpectralBiclustering, clus.serialize_spectral_biclustering, clus.deserialize_spectral_biclustering),
    (SpectralCoclustering, clus.serialize_spectral_coclustering, clus.deserialize_spectral_coclustering),
    (Birch, clus.serialize_birch, clus.deserialize_birch),

    # Cross-decomposition
    (CCA, crdec.serialize_cca, crdec.deserialize_cca),
    (PLSCanonical, crdec.serialize_pls_canonical, crdec.deserialize_pls_canonical),
    (PLSRegression, crdec.serialize_pls_regression, crdec.deserialize_pls_regression),
    (PLSSVD, crdec.serialize_pls_svd, crdec.deserialize_pls_svd),

    # Decomposition
    (PCA, dec.serialize_pca, dec.deserialize_pca),
    (KernelPCA, dec.serialize_kernel_pca, dec.deserialize_kernel_pca),
    (IncrementalPCA, dec.serialize_incremental_pca, dec.deserialize_incremental_pca),
    (MiniBatchSparsePCA, dec.serialize_minibatch_sparse_pca, dec.deserialize_minibatch_sparse_pca),
    (SparsePCA, dec.serialize_sparse_pca, dec.deserialize_sparse_pca),
    (MiniBatchDictionaryLearning, dec.serialize_minibatch_dictionary_learning, dec.deserialize_minibatch_dictionary_learning),
    (DictionaryLearning, dec.serialize_dictionary_learning, dec.deserialize_dictionary_learning),
    (FactorAnalysis, dec.serialize_factor_analysis, dec.deserialize_factor_analysis),
    (FastICA, dec.serialize_fast_ica, dec.deserialize_fast_ica),
    (LatentDirichletAllocation, dec.serialize_latent_dirichlet_allocation, dec.deserialize_latent_dirichlet_allocation),
    (MiniBatchNMF, dec.serialize_minibatch_nmf, dec.deserialize_minibatch_nmf),
    (NMF, dec.serialize_nmf, dec.deserialize_nmf),
    (SparseCoder, dec.serialize_sparse_coder, dec.deserialize_sparse_coder),
    (TruncatedSVD, dec.serialize_truncated_svd, dec.deserialize_truncated_svd),

    # Manifold
    (TSNE, man.serialize_tsne, man.deserialize_tsne),
    (MDS, man.serialize_mds, man.deserialize_mds),
    (Isomap, man.serialize_isomap, man.deserialize_isomap),
    (LocallyLinearEmbedding, man.serialize_locally_linear_embedding, man.deserialize_locally_linear_embedding),
    (SpectralEmbedding, man.serialize_spectral_embedding, man.deserialize_spectral_embedding),

    # Neighbors
    (NearestNeighbors, nei.serialize_nearest_neighbors, nei.deserialize_nearest_neighbors),
    (KDTree, nei.serialize_kdtree, nei.deserialize_kdtree),
    (KernelDensity, nei.serialize_kernel_density, nei.deserialize_kernel_density),

    # Feature Extraction
    (DictVectorizer, ext.serialize_dict_vectorizer, ext.deserialize_dict_vectorizer),

    # Preprocess
    (LabelEncoder, pre.serialize_label_encoder, pre.deserialize_label_encoder),
    (LabelBinarizer, pre.serialize_label_binarizer, pre.deserialize_label_binarizer),
    (MultiLabelBinarizer, pre.serialize_multilabel_binarizer, pre.deserialize_multilabel_binarizer),
    (MinMaxScaler, pre.serialize_minmax_scaler, pre.deserialize_minmax_scaler),
    (StandardScaler, pre.serialize_standard_scaler, pre.deserialize_standard_scaler),
    (RobustScaler, pre.serialize_robust_scaler, pre.deserialize_robust_scaler),
    (MaxAbsScaler, pre.serialize_maxabs_scaler, pre.deserialize_maxabs_scaler),
    (KernelCenterer, pre.serialize_kernel_centerer, pre.deserialize_kernel_centerer),
    (OneHotEncoder, pre.serialize_onehot_encoder, pre.deserialize_onehot_encoder),
    (OrdinalEncoder, pre.serialize_ordinal_encoder, pre.deserialize_ordinal_encoder),
    (Normalizer, pre.serialize_normalizer, pre.deserialize_normalizer),

    # Pipeline
    (Pipeline, ppl.serialize_pipeline, ppl.deserialize_pipeline),
]

# Optional dependencies: registered the same way, guarded by the same
# '<Name>' in <module>.__optionals__ checks the imports above already use.
if 'XGBClassifier' in clf.__optionals__:
    _REGISTRY.append((XGBClassifier, clf.serialize_xgboost_classifier, clf.deserialize_xgboost_classifier))
if 'XGBRFClassifier' in clf.__optionals__:
    _REGISTRY.append((XGBRFClassifier, clf.serialize_xgboost_rf_classifier, clf.deserialize_xgboost_rf_classifier))
if 'LGBMClassifier' in clf.__optionals__:
    _REGISTRY.append((LGBMClassifier, clf.serialize_lightgbm_classifier, clf.deserialize_lightgbm_classifier))
if 'CatBoostClassifier' in clf.__optionals__:
    _REGISTRY.append((CatBoostClassifier, clf.serialize_catboost_classifier, clf.deserialize_catboost_classifier))

if 'XGBRanker' in reg.__optionals__:
    _REGISTRY.append((XGBRanker, reg.serialize_xgboost_ranker, reg.deserialize_xgboost_ranker))
if 'XGBRegressor' in reg.__optionals__:
    _REGISTRY.append((XGBRegressor, reg.serialize_xgboost_regressor, reg.deserialize_xgboost_regressor))
if 'XGBRFRegressor' in reg.__optionals__:
    _REGISTRY.append((XGBRFRegressor, reg.serialize_xgboost_rf_regressor, reg.deserialize_xgboost_rf_regressor))
if 'LGBMRegressor' in reg.__optionals__:
    _REGISTRY.append((LGBMRegressor, reg.serialize_lightgbm_regressor, reg.deserialize_lightgbm_regressor))
if 'LGBMRanker' in reg.__optionals__:
    _REGISTRY.append((LGBMRanker, reg.serialize_lightgbm_ranker, reg.deserialize_lightgbm_ranker))
if 'CatBoostRegressor' in reg.__optionals__:
    _REGISTRY.append((CatBoostRegressor, reg.serialize_catboost_regressor, reg.deserialize_catboost_regressor))
if 'CatBoostRanker' in reg.__optionals__:
    _REGISTRY.append((CatBoostRanker, reg.serialize_catboost_ranker, reg.deserialize_catboost_ranker))

if 'KPrototypes' in clus.__optionals__:
    _REGISTRY.append((KPrototypes, clus.serialize_kprototypes, clus.deserialize_kprototypes))
if 'KModes' in clus.__optionals__:
    _REGISTRY.append((KModes, clus.serialize_kmodes, clus.deserialize_kmodes))
if 'HDBSCAN' in clus.__optionals__:
    _REGISTRY.append((HDBSCAN, clus.serialize_hdbscan, clus.deserialize_hdbscan))

if 'UMAP' in man.__optionals__:
    _REGISTRY.append((UMAP, man.serialize_umap, man.deserialize_umap))
if 'OpenTSNE' in man.__optionals__:
    _REGISTRY.append((OpenTSNE, man.serialize_opentsne, man.deserialize_opentsne))
    _REGISTRY.append((OpenTSNEsklearn, man.serialize_opentsne, man.deserialize_opentsne))
    _REGISTRY.append((OpenTSNEEmbedding, man.serialize_opentsne_embedding, man.deserialize_opentsne_embedding))
    _REGISTRY.append((OpenPartialTSNEEmbedding, man.serialize_opentsne_partial_embedding, man.deserialize_opentsne_partial_embedding))

if 'NNDescent' in nei.__optionals__:
    _REGISTRY.append((NNDescent, nei.serialize_nndescent, nei.deserialize_nndescent))

if 'BoundingBoxApplicabilityDomain' in ad.__optionals__:
    _REGISTRY.extend([
        (BoundingBoxApplicabilityDomain, ad.serialize_bounding_box_applicability_domain, ad.deserialize_bounding_box_applicability_domain),
        (ConvexHullApplicabilityDomain, ad.serialize_convex_hull_applicability_domain, ad.deserialize_convex_hull_applicability_domain),
        (PCABoundingBoxApplicabilityDomain, ad.serialize_pca_bounding_box_applicability_domain, ad.deserialize_pca_bounding_box_applicability_domain),
        (TopKatApplicabilityDomain, ad.serialize_topkat_applicability_domain, ad.deserialize_topkat_applicability_domain),
        (LeverageApplicabilityDomain, ad.serialize_leverage_applicability_domain, ad.deserialize_leverage_applicability_domain),
        (HotellingT2ApplicabilityDomain, ad.serialize_hotelling_t2_applicability_domain, ad.deserialize_hotelling_t2_applicability_domain),
        (KernelDensityApplicabilityDomain, ad.serialize_kernel_density_applicability_domain, ad.deserialize_kernel_density_applicability_domain),
        (IsolationForestApplicabilityDomain, ad.serialize_isolation_forest_applicability_domain, ad.deserialize_isolation_forest_applicability_domain),
        (CentroidDistanceApplicabilityDomain, ad.serialize_centroid_distance_applicability_domain, ad.deserialize_centroid_distance_applicability_domain),
        (KNNApplicabilityDomain, ad.serialize_knn_applicability_domain, ad.deserialize_knn_applicability_domain),
        (StandardizationApproachApplicabilityDomain, ad.serialize_standardization_approach_applicability_domain, ad.deserialize_standardization_approach_applicability_domain),
    ])

if 'imblearn' in ous.__optionals__:
    _REGISTRY.extend([
        (ClusterCentroids, ous.serialize_cluster_centroids, ous.deserialize_cluster_centroids),
        (CondensedNearestNeighbour, ous.serialize_condensed_nearest_neighbours, ous.deserialize_condensed_nearest_neighbours),
        (EditedNearestNeighbours, ous.serialize_edited_nearest_neighbours, ous.deserialize_edited_nearest_neighbours),
        (RepeatedEditedNearestNeighbours, ous.serialize_repeated_edited_nearest_neighbours, ous.deserialize_repeated_edited_nearest_neighbours),
        (AllKNN, ous.serialize_all_knn, ous.deserialize_all_knn),
        (InstanceHardnessThreshold, ous.serialize_instance_hardness_threshold, ous.deserialize_instance_hardness_threshold),
        (NearMiss, ous.serialize_near_miss, ous.deserialize_near_miss),
        (NeighbourhoodCleaningRule, ous.serialize_neighbourhood_cleaning_rule, ous.deserialize_neighbourhood_cleaning_rule),
        (OneSidedSelection, ous.serialize_one_sided_selection, ous.deserialize_one_sided_selection),
        (RandomUnderSampler, ous.serialize_random_under_sampler, ous.deserialize_random_under_sampler),
        (TomekLinks, ous.serialize_tomek_links, ous.deserialize_tomek_links),
        (RandomOverSampler, ous.serialize_random_over_sampler, ous.deserialize_random_over_sampler),
        (SMOTENC, ous.serialize_smotenc, ous.deserialize_smotenc),
        (SMOTEN, ous.serialize_smoten, ous.deserialize_smoten),
        (SMOTE, ous.serialize_smote, ous.deserialize_smote),
        (ADASYN, ous.serialize_adasyn, ous.deserialize_adasyn),
        (BorderlineSMOTE, ous.serialize_borderline_smote, ous.deserialize_borderline_smote),
        (KMeansSMOTE, ous.serialize_kmeans_smote, ous.deserialize_kmeans_smote),
        (SVMSMOTE, ous.serialize_svm_smote, ous.deserialize_svm_smote),
        (SMOTEENN, ous.serialize_smote_enn, ous.deserialize_smote_enn),
        (SMOTETomek, ous.serialize_smote_tomek, ous.deserialize_smote_tomek),
    ])

if 'imblearn' in ppl.__optionals__:
    _REGISTRY.append((ImblearnPipeline, ppl.serialize_imblearn_pipeline, ppl.deserialize_imblearn_pipeline))

# CatBoost serializers need an extra `catboost_data` argument that no other
# serializer takes, so they're routed separately in serialize_model rather
# than forcing every serialize_X function to accept an unused parameter.
_CATBOOST_SERIALIZE_FNS = {}
if 'CatBoostClassifier' in clf.__optionals__:
    _CATBOOST_SERIALIZE_FNS[CatBoostClassifier] = clf.serialize_catboost_classifier
if 'CatBoostRegressor' in reg.__optionals__:
    _CATBOOST_SERIALIZE_FNS[CatBoostRegressor] = reg.serialize_catboost_regressor
if 'CatBoostRanker' in reg.__optionals__:
    _CATBOOST_SERIALIZE_FNS[CatBoostRanker] = reg.serialize_catboost_ranker

_META_BY_TYPE = {}
_DESERIALIZE_BY_META = {}
for _cls, _ser_fn, _deser_fn in _REGISTRY:
    _meta = _meta_for(_cls)
    if _meta in _DESERIALIZE_BY_META:
        raise RuntimeError(f'Duplicate meta {_meta!r}: {_cls} collides with an existing registry entry')
    _DESERIALIZE_BY_META[_meta] = _deser_fn
    _META_BY_TYPE[_cls] = _meta
_SERIALIZE_BY_TYPE = {cls: ser_fn for cls, ser_fn, _ in _REGISTRY if cls not in _CATBOOST_SERIALIZE_FNS}
del _cls, _ser_fn, _deser_fn, _meta

# Legacy meta strings this library used before dispatch was table-driven
# (e.g. 'lr', 'isomap', 'umap'). Kept so previously-serialized JSON still
# deserializes; deserialize_model emits a DeprecationWarning when it falls
# back to this table.
_LEGACY_META_ALIASES = {
    # Classification
    'lr': clf.deserialize_logistic_regression,
    'bernoulli-nb': clf.deserialize_bernoulli_nb,
    'gaussian-nb': clf.deserialize_gaussian_nb,
    'multinomial-nb': clf.deserialize_multinomial_nb,
    'complement-nb': clf.deserialize_complement_nb,
    'lda': clf.deserialize_lda,
    'qda': clf.deserialize_qda,
    'svm': clf.deserialize_svm,
    'perceptron': clf.deserialize_perceptron,
    'decision-tree': clf.deserialize_decision_tree,
    'gb': clf.deserialize_gradient_boosting,
    'rf': clf.deserialize_random_forest,
    'mlp': clf.deserialize_mlp,
    'adaboost-classifier': clf.deserialize_adaboost_classifier,
    'bagging-classifier': clf.deserialize_bagging_classifier,
    'extra-tree-cls': clf.deserialize_extra_tree_classifier,
    'extratrees-classifier': clf.deserialize_extratrees_classifier,
    'isolation-forest': clf.deserialize_isolation_forest,
    'random-trees-embedding': clf.deserialize_random_trees_embedding,
    'nearest-neighbour-classifier': clf.deserialize_nearest_neighbour_classifier,
    'stacking-classifier': clf.deserialize_stacking_classifier,
    'voting-classifier': clf.deserialize_voting_classifier,

    # Regression
    'linear-regression': reg.deserialize_linear_regressor,
    'lasso-regression': reg.deserialize_lasso_regressor,
    'elasticnet-regression': reg.deserialize_elastic_regressor,
    'ridge-regression': reg.deserialize_ridge_regressor,
    'svr': reg.deserialize_svr,
    'decision-tree-regression': reg.deserialize_decision_tree_regressor,
    'gb-regression': reg.deserialize_gradient_boosting_regressor,
    'rf-regression': reg.deserialize_random_forest_regressor,
    'mlp-regression': reg.deserialize_mlp_regressor,
    'adaboost-regressor': reg.deserialize_adaboost_regressor,
    'bagging-regression': reg.deserialize_bagging_regressor,
    'extra-tree-reg': reg.deserialize_extra_tree_regressor,
    'extratrees-regressor': reg.deserialize_extratrees_regressor,
    'nearest-neighbour-regressor': reg.deserialize_nearest_neighbour_regressor,
    'stacking-regressor': reg.deserialize_stacking_regressor,
    'voting-regressor': reg.deserialize_voting_regressor,

    # Clustering
    'affinity-propagation': clus.deserialize_affinity_propagation,
    'agglomerative-clustering': clus.deserialize_agglomerative_clustering,
    'feature-agglomeration': clus.deserialize_feature_agglomeration,
    'dbscan': clus.deserialize_dbscan,
    'meanshift': clus.deserialize_meanshift,
    'kmeans': clus.deserialize_kmeans,
    'minibatch-kmeans': clus.deserialize_minibatch_kmeans,
    'optics': clus.deserialize_optics,
    'spectral-clustering': clus.deserialize_spectral_clustering,
    'spectral-biclustering': clus.deserialize_spectral_biclustering,
    'spectral-coclustering': clus.deserialize_spectral_coclustering,
    'birch': clus.deserialize_birch,
    'bisecting-kmeans': clus.deserialize_bisecting_kmeans,

    # Cross-decomposition
    'cca': crdec.deserialize_cca,
    'pls-canonical': crdec.deserialize_pls_canonical,
    'pls-regression': crdec.deserialize_pls_regression,
    'pls-svd': crdec.deserialize_pls_svd,

    # Decomposition
    'pca': dec.deserialize_pca,
    'kernel-pca': dec.deserialize_kernel_pca,
    'incremental-pca': dec.deserialize_incremental_pca,
    'sparse-pca': dec.deserialize_sparse_pca,
    'minibatch-sparse-pca': dec.deserialize_minibatch_sparse_pca,
    'dictionary-learning': dec.deserialize_dictionary_learning,
    'minibatch-dictionary-learning': dec.deserialize_minibatch_dictionary_learning,
    'factor-analysis': dec.deserialize_factor_analysis,
    'fast-ica': dec.deserialize_fast_ica,
    'latent-dirichlet-allocation': dec.deserialize_latent_dirichlet_allocation,
    'nmf': dec.deserialize_nmf,
    'minibatch-nmf': dec.deserialize_minibatch_nmf,
    'sparse-coder': dec.deserialize_sparse_coder,
    'truncated-svd': dec.deserialize_truncated_svd,

    # Manifold
    'tsne': man.deserialize_tsne,
    'mds': man.deserialize_mds,
    'isomap': man.deserialize_isomap,
    'locally-linear-embedding': man.deserialize_locally_linear_embedding,
    'spectral-embedding': man.deserialize_spectral_embedding,

    # Neighbors
    'nearest-neighbors': nei.deserialize_nearest_neighbors,
    'kdtree': nei.deserialize_kdtree,
    'kernel-density': nei.deserialize_kernel_density,

    # Feature Extraction
    'dict-vectorizer': ext.deserialize_dict_vectorizer,

    # Preprocess
    'label-encoder': pre.deserialize_label_encoder,
    'label-binarizer': pre.deserialize_label_binarizer,
    'multilabel-binarizer': pre.deserialize_multilabel_binarizer,
    'minmax-scaler': pre.deserialize_minmax_scaler,
    'standard-scaler': pre.deserialize_standard_scaler,
    'robust-scaler': pre.deserialize_robust_scaler,
    'maxabs-scaler': pre.deserialize_maxabs_scaler,
    'kernel-centerer': pre.deserialize_kernel_centerer,
    'onehot-encoder': pre.deserialize_onehot_encoder,
    'ordinal-encoder': pre.deserialize_ordinal_encoder,
    'normalizer': pre.deserialize_normalizer,

    # Pipeline
    'pipeline': ppl.deserialize_pipeline,
}

if 'XGBClassifier' in clf.__optionals__:
    _LEGACY_META_ALIASES['xgboost-classifier'] = clf.deserialize_xgboost_classifier
if 'XGBRFClassifier' in clf.__optionals__:
    _LEGACY_META_ALIASES['xgboost-rf-classifier'] = clf.deserialize_xgboost_rf_classifier
if 'LGBMClassifier' in clf.__optionals__:
    _LEGACY_META_ALIASES['lightgbm-classifier'] = clf.deserialize_lightgbm_classifier
if 'CatBoostClassifier' in clf.__optionals__:
    _LEGACY_META_ALIASES['catboost-classifier'] = clf.deserialize_catboost_classifier
if 'XGBRanker' in reg.__optionals__:
    _LEGACY_META_ALIASES['xgboost-ranker'] = reg.deserialize_xgboost_ranker
if 'XGBRegressor' in reg.__optionals__:
    _LEGACY_META_ALIASES['xgboost-regressor'] = reg.deserialize_xgboost_regressor
if 'XGBRFRegressor' in reg.__optionals__:
    _LEGACY_META_ALIASES['xgboost-rf-regressor'] = reg.deserialize_xgboost_rf_regressor
if 'LGBMRegressor' in reg.__optionals__:
    _LEGACY_META_ALIASES['lightgbm-regressor'] = reg.deserialize_lightgbm_regressor
if 'LGBMRanker' in reg.__optionals__:
    _LEGACY_META_ALIASES['lightgbm-ranker'] = reg.deserialize_lightgbm_ranker
if 'CatBoostRegressor' in reg.__optionals__:
    _LEGACY_META_ALIASES['catboost-regressor'] = reg.deserialize_catboost_regressor
if 'CatBoostRanker' in reg.__optionals__:
    _LEGACY_META_ALIASES['catboost-ranker'] = reg.deserialize_catboost_ranker
if 'KPrototypes' in clus.__optionals__:
    _LEGACY_META_ALIASES['kprototypes'] = clus.deserialize_kprototypes
if 'KModes' in clus.__optionals__:
    _LEGACY_META_ALIASES['kmodes'] = clus.deserialize_kmodes
if 'HDBSCAN' in clus.__optionals__:
    _LEGACY_META_ALIASES['hdbscan'] = clus.deserialize_hdbscan
if 'UMAP' in man.__optionals__:
    _LEGACY_META_ALIASES['umap'] = man.deserialize_umap
if 'OpenTSNE' in man.__optionals__:
    _LEGACY_META_ALIASES['openTSNE'] = man.deserialize_opentsne
    _LEGACY_META_ALIASES['openTSNEEmbedding'] = man.deserialize_opentsne_embedding
    _LEGACY_META_ALIASES['openTSNEPartialEmbedding'] = man.deserialize_opentsne_partial_embedding
if 'NNDescent' in nei.__optionals__:
    _LEGACY_META_ALIASES['nn-descent'] = nei.deserialize_nndescent
if 'BoundingBoxApplicabilityDomain' in ad.__optionals__:
    _LEGACY_META_ALIASES.update({
        'bounding-box-ad': ad.deserialize_bounding_box_applicability_domain,
        'convex-hull-ad': ad.deserialize_convex_hull_applicability_domain,
        'pca-bounding-box-ad': ad.deserialize_pca_bounding_box_applicability_domain,
        'topkat-ad': ad.deserialize_topkat_applicability_domain,
        'leverage-ad': ad.deserialize_leverage_applicability_domain,
        'hotelling-t2-ad': ad.deserialize_hotelling_t2_applicability_domain,
        'kernel-density-ad': ad.deserialize_kernel_density_applicability_domain,
        'isolation-forest-ad': ad.deserialize_isolation_forest_applicability_domain,
        'centroid-distance-ad': ad.deserialize_centroid_distance_applicability_domain,
        'knn-ad': ad.deserialize_knn_applicability_domain,
        'standardization-approach-ad': ad.deserialize_standardization_approach_applicability_domain,
    })
if 'imblearn' in ous.__optionals__:
    _LEGACY_META_ALIASES.update({
        'cluster-centroids': ous.deserialize_cluster_centroids,
        'condensed-nearest-neighbours': ous.deserialize_condensed_nearest_neighbours,
        'edited-nearest-neighbours': ous.deserialize_edited_nearest_neighbours,
        'repeated-edited-nearest-neighbours': ous.deserialize_repeated_edited_nearest_neighbours,
        'all-knn': ous.deserialize_all_knn,
        'instance-hardness-threshold': ous.deserialize_instance_hardness_threshold,
        'near-miss': ous.deserialize_near_miss,
        'neighbourhood-cleaning-rule': ous.deserialize_neighbourhood_cleaning_rule,
        'one-sided-selection': ous.deserialize_one_sided_selection,
        'random-under-sampler': ous.deserialize_random_under_sampler,
        'tomek-links': ous.deserialize_tomek_links,
        'random-over-sampler': ous.deserialize_random_over_sampler,
        'smotenc': ous.deserialize_smotenc,
        'smoten': ous.deserialize_smoten,
        'smote': ous.deserialize_smote,
        'adasyn': ous.deserialize_adasyn,
        'borderline-smote': ous.deserialize_borderline_smote,
        'kmeans-smote': ous.deserialize_kmeans_smote,
        'svm-smote': ous.deserialize_svm_smote,
        'smote-enn': ous.deserialize_smote_enn,
        'smote-tomek': ous.deserialize_smote_tomek,
    })


def serialize_model(model, catboost_data: Pool = None) -> Dict:
    """Serialize a model into a dictionary.

    :param model: machine learning model to be serialized
    :param catboost_data: if `model` is a CatBoost model, the data `Pool` used to train it
    """
    # Verify model is fit
    if not is_model_fitted(model):
        return serialize_unfitted_model(model)

    cls = type(model)
    if cls in _CATBOOST_SERIALIZE_FNS:
        model_dict = _CATBOOST_SERIALIZE_FNS[cls](model, catboost_data)
    elif cls in _SERIALIZE_BY_TYPE:
        model_dict = _SERIALIZE_BY_TYPE[cls](model)
    else:
        # Otherwise: fall back to generically walking the model's __dict__
        try:
            model_dict = _base.serialize_model_generic(model)
        except _base.ModelNotSupported:
            raise ModelNotSupported('This model type is not currently supported. Email support@mlrequest.com to request a feature or report a bug.')
        return serialize_version(model, model_dict)

    model_dict['meta'] = _META_BY_TYPE[cls]
    return serialize_version(model, model_dict)


def deserialize_model(model_dict: Dict):
    """Instantiate a machine learning model from a previously serialized model.

    :param model_dict: dictionary of the previously serialized model
    """
    # Verify model is fitted
    if 'unfitted' in model_dict.keys() and model_dict['unfitted']:
        check_version(model_dict)
        return deserialize_unfitted_model(model_dict)

    meta = model_dict['meta']

    if meta in _DESERIALIZE_BY_META:
        check_version(model_dict)
        return _DESERIALIZE_BY_META[meta](model_dict)
    elif meta in _LEGACY_META_ALIASES:
        warnings.warn(f'meta tag {meta!r} uses the pre-registry ml2json format; '
                      f're-serialize this model to upgrade it', DeprecationWarning)
        check_version(model_dict)
        return _LEGACY_META_ALIASES[meta](model_dict)
    # Otherwise: fall back to the generic engine for anything it serialized
    elif isinstance(meta, str) and meta.startswith('generic_object:'):
        check_version(model_dict)
        return _base.deserialize_model_generic(model_dict)
    else:
        raise ModelNotSupported('Model type not supported or corrupt JSON file.')


def serialize_unfitted_model(model):
    """Serialize an unfitted model.

    :param model: unfitted model
    """
    serialized_model = {
        'unfitted': True,
        'meta': (inspect.getmodule(model).__name__,
                 type(model).__name__),
        'params': model.get_params()
    }
    serialize_version(model, serialized_model)
    return serialized_model


def deserialize_unfitted_model(model_dict: Dict):
    """Deserialize an unfitter model.

    :param model_dict: previously serialized unfitted model
    """
    check_version(model_dict)
    model = getattr(importlib.import_module(model_dict['meta'][0]), model_dict['meta'][1])(**model_dict['params'])
    return model


def to_dict(model, catboost_data: Pool = None):
    """Equivalent to `serialize_model`"""
    return serialize_model(model, catboost_data)


def from_dict(model_dict):
    """Equivalent to `deserialize_model`"""
    return deserialize_model(model_dict)


def to_json(model, outfile, catboost_data: Pool = None):
    """Serialize a model to a json file.

    :param model: the model to serialize
    :param outfile: the json file to be created
    :param catboost_data: if `model` is a CatBoost model, the data `Pool` used to train it
    """
    model_dict = to_dict(model, catboost_data)
    dict_to_json(model_dict, outfile)


def from_json(infile):
    """Instantiate a previously serialized model from a json file.

    :param infile: json file containing the serialized model
    """
    model_dict = json_to_dict(infile)
    return deserialize_model(model_dict)


def dict_to_json(model_dict: Dict, outfile: str):
    """Write a serialized model to a json file.

    :param model_dict: serialized model
    :param outfile: json file to be created
    """
    with open(outfile, 'w') as model_json:
        json.dump(model_dict, model_json)


def json_to_dict(infile):
    """Obtain a serialized model from a json file.

    :param infile: json file to read the serialized model from
    """
    with open(infile, 'r') as model_json:
        model_dict = json.load(model_json)
    return model_dict


def serialize_version(model, model_dict):
    """Add version(s) of the libraries required to instantiate the model.

    :param model: model to check the dependencies of
    :param model_dict: serialized model to add the dependencies' versions to
    """
    # A user-supplied RandomState instance (e.g. RandomForestClassifier(random_state=
    # np.random.RandomState(0))) leaks straight through model.get_params() into
    # 'params' unconverted by every hand-written serializer, which is not
    # JSON-safe. Sanitize it here since every serialize_* branch funnels through
    # this single function before returning.
    if 'params' in model_dict:
        model_dict['params'] = {key: (serialize_random_state(value) if isinstance(value, RandomState) else value)
                                for key, value in model_dict['params'].items()}
    # Obtain library used to fit the model
    module = inspect.getmodule(model)
    if module is None:
        return model_dict
    module = sys.modules[module.__name__.partition('.')[0]]
    version = module.__version__ if hasattr(module, '__version__') else ''
    model_dict['versions'] = (module.__name__, version)
    return model_dict


def check_version(model_dict):
    """Check if the versions of the installed libraries and those the model was fitted with correspond.

    :param model_dict: serialized model
    """
    # Reverse of the RandomState sanitization done in serialize_version, so every
    # deserialize_* branch (which calls SomeClass(**model_dict['params'])) receives
    # back a real RandomState instance rather than its serialized dict form.
    if 'params' in model_dict:
        model_dict['params'] = {key: (deserialize_random_state(value)
                                      if isinstance(value, dict) and value.get('meta') == 'random_state' else value)
                                for key, value in model_dict['params'].items()}
    if 'versions' not in model_dict:
        return
    # Obtain module used to fit the model
    module_name, version = model_dict['versions']
    # Module is installed
    installed = importlib.util.find_spec(module_name) is not None
    if not installed:
        raise ModuleNotFoundError(f'Module {module_name} could not be found. Is it installed?')
    # Check version of the installed module
    if version == '':
        return
    installed_version = importlib.import_module(module_name).__version__
    if version != installed_version:
        warnings.warn(f'Version of the current {module_name} library ({installed_version}) '
                      f'does not match the version used to fit the serialized model ({version})')


class ModelNotSupported(Exception):
    """Custom class for unsupported model types."""
    pass
