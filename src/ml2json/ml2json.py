# -*- coding: utf-8 -*-

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
                                   OrdinalEncoder)
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
from .utils import is_model_fitted, recursive_inspection

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


def serialize_model(model, catboost_data: Pool = None) -> Dict:
    """Serialize a model into a dictionary.

    :param model: machine learning model to be serialized
    :param catboost_data: if `model` is a CatBoost model, the data `Pool` used to train it
    """
    # Verify model is fit
    if not is_model_fitted(model):
        return serialize_unfitted_model(model)

    # Classification
    if isinstance(model, LogisticRegression):
        model_dict = clf.serialize_logistic_regression(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, BernoulliNB):
        model_dict = clf.serialize_bernoulli_nb(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, GaussianNB):
        model_dict = clf.serialize_gaussian_nb(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, MultinomialNB):
        model_dict = clf.serialize_multinomial_nb(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, ComplementNB):
        model_dict = clf.serialize_complement_nb(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, discriminant_analysis.LinearDiscriminantAnalysis):
        model_dict = clf.serialize_lda(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, discriminant_analysis.QuadraticDiscriminantAnalysis):
        model_dict = clf.serialize_qda(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, svm.SVC):
        model_dict = clf.serialize_svm(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, Perceptron):
        model_dict = clf.serialize_perceptron(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, DecisionTreeClassifier):
        model_dict = clf.serialize_decision_tree(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, GradientBoostingClassifier):
        model_dict = clf.serialize_gradient_boosting(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, RandomForestClassifier):
        model_dict = clf.serialize_random_forest(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, MLPClassifier):
        model_dict = clf.serialize_mlp(model)
        return serialize_version(model, model_dict)
    elif 'XGBClassifier' in clf.__optionals__ and isinstance(model, XGBClassifier):
        model_dict = clf.serialize_xgboost_classifier(model)
        return serialize_version(model, model_dict)
    elif 'XGBRFClassifier' in clf.__optionals__ and isinstance(model, XGBRFClassifier):
        model_dict = clf.serialize_xgboost_rf_classifier(model)
        return serialize_version(model, model_dict)
    elif 'LGBMClassifier' in clf.__optionals__ and isinstance(model, LGBMClassifier):
        model_dict = clf.serialize_lightgbm_classifier(model)
        return serialize_version(model, model_dict)
    elif 'CatBoostClassifier' in clf.__optionals__ and isinstance(model, CatBoostClassifier):
        model_dict = clf.serialize_catboost_classifier(model, catboost_data)
        return serialize_version(model, model_dict)
    elif isinstance(model, AdaBoostClassifier):
        model_dict = clf.serialize_adaboost_classifier(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, BaggingClassifier):
        model_dict = clf.serialize_bagging_classifier(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, ExtraTreeClassifier):
        model_dict = clf.serialize_extra_tree_classifier(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, ExtraTreesClassifier):
        model_dict = clf.serialize_extratrees_classifier(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, IsolationForest):
        model_dict = clf.serialize_isolation_forest(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, RandomTreesEmbedding):
        model_dict = clf.serialize_random_trees_embedding(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, KNeighborsClassifier):
        model_dict = clf.serialize_nearest_neighbour_classifier(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, StackingClassifier):
        model_dict = clf.serialize_stacking_classifier(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, VotingClassifier):
        model_dict = clf.serialize_voting_classifier(model)
        return serialize_version(model, model_dict)

    # Regression
    elif isinstance(model, LinearRegression):
        model_dict = reg.serialize_linear_regressor(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, Lasso):
        model_dict = reg.serialize_lasso_regressor(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, ElasticNet):
        model_dict = reg.serialize_elastic_regressor(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, Ridge):
        model_dict = reg.serialize_ridge_regressor(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, SVR):
        model_dict = reg.serialize_svr(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, ExtraTreeRegressor):
        model_dict = reg.serialize_extra_tree_regressor(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, DecisionTreeRegressor):
        model_dict = reg.serialize_decision_tree_regressor(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, GradientBoostingRegressor):
        model_dict = reg.serialize_gradient_boosting_regressor(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, RandomForestRegressor):
        model_dict = reg.serialize_random_forest_regressor(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, ExtraTreesRegressor):
        model_dict = reg.serialize_extratrees_regressor(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, MLPRegressor):
        model_dict = reg.serialize_mlp_regressor(model)
        return serialize_version(model, model_dict)
    elif 'XGBRanker' in reg.__optionals__ and isinstance(model, XGBRanker):
        model_dict = reg.serialize_xgboost_ranker(model)
        return serialize_version(model, model_dict)
    elif 'XGBRegressor' in reg.__optionals__ and isinstance(model, XGBRegressor):
        model_dict = reg.serialize_xgboost_regressor(model)
        return serialize_version(model, model_dict)
    elif 'XGBRFRegressor' in reg.__optionals__ and isinstance(model, XGBRFRegressor):
        model_dict = reg.serialize_xgboost_rf_regressor(model)
        return serialize_version(model, model_dict)
    elif 'LGBMRegressor' in reg.__optionals__ and isinstance(model, LGBMRegressor):
        model_dict = reg.serialize_lightgbm_regressor(model)
        return serialize_version(model, model_dict)
    elif 'LGBMRanker' in reg.__optionals__ and isinstance(model, LGBMRanker):
        model_dict = reg.serialize_lightgbm_ranker(model)
        return serialize_version(model, model_dict)
    elif 'CatBoostRegressor' in reg.__optionals__ and isinstance(model, CatBoostRegressor):
        model_dict = reg.serialize_catboost_regressor(model, catboost_data)
        return serialize_version(model, model_dict)
    elif 'CatBoostRanker' in reg.__optionals__ and isinstance(model, CatBoostRanker):
        model_dict = reg.serialize_catboost_ranker(model, catboost_data)
        return serialize_version(model, model_dict)
    elif isinstance(model, AdaBoostRegressor):
        model_dict = reg.serialize_adaboost_regressor(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, BaggingRegressor):
        model_dict = reg.serialize_bagging_regressor(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, KNeighborsRegressor):
        model_dict = reg.serialize_nearest_neighbour_regressor(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, StackingRegressor):
        model_dict = reg.serialize_stacking_regressor(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, VotingRegressor):
        model_dict = reg.serialize_voting_regressor(model)
        return serialize_version(model, model_dict)

    # Clustering
    elif isinstance(model, FeatureAgglomeration):
        model_dict = clus.serialize_feature_agglomeration(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, AffinityPropagation):
        model_dict = clus.serialize_affinity_propagation(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, AgglomerativeClustering):
        model_dict = clus.serialize_agglomerative_clustering(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, DBSCAN):
        model_dict = clus.serialize_dbscan(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, MeanShift):
        model_dict = clus.serialize_meanshift(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, BisectingKMeans):
        model_dict = clus.serialize_bisecting_kmeans(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, MiniBatchKMeans):
        model_dict = clus.serialize_minibatch_kmeans(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, KMeans):
        model_dict = clus.serialize_kmeans(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, OPTICS):
        model_dict = clus.serialize_optics(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, SpectralClustering):
        model_dict = clus.serialize_spectral_clustering(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, SpectralBiclustering):
        model_dict = clus.serialize_spectral_biclustering(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, SpectralCoclustering):
        model_dict = clus.serialize_spectral_coclustering(model)
        return serialize_version(model, model_dict)
    elif 'KPrototypes' in clus.__optionals__ and isinstance(model, KPrototypes):
        model_dict = clus.serialize_kprototypes(model)
        return serialize_version(model, model_dict)
    elif 'KModes' in clus.__optionals__ and isinstance(model, KModes):
        model_dict = clus.serialize_kmodes(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, Birch):
        model_dict = clus.serialize_birch(model)
        return serialize_version(model, model_dict)
    elif 'HDBSCAN' in clus.__optionals__ and isinstance(model, HDBSCAN):
        model_dict = clus.serialize_hdbscan(model)
        return serialize_version(model, model_dict)

    # Cross-decomposition
    elif isinstance(model, CCA):
        model_dict = crdec.serialize_cca(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, PLSCanonical):
        model_dict = crdec.serialize_pls_canonical(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, PLSRegression):
        model_dict = crdec.serialize_pls_regression(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, PLSSVD):
        model_dict = crdec.serialize_pls_svd(model)
        return serialize_version(model, model_dict)

    # Decomposition
    elif isinstance(model, PCA):
        model_dict = dec.serialize_pca(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, KernelPCA):
        model_dict = dec.serialize_kernel_pca(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, IncrementalPCA):
        model_dict = dec.serialize_incremental_pca(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, MiniBatchSparsePCA):
        model_dict = dec.serialize_minibatch_sparse_pca(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, SparsePCA):
        model_dict = dec.serialize_sparse_pca(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, MiniBatchDictionaryLearning):
        model_dict = dec.serialize_minibatch_dictionary_learning(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, DictionaryLearning):
        model_dict = dec.serialize_dictionary_learning(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, FactorAnalysis):
        model_dict = dec.serialize_factor_analysis(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, FastICA):
        model_dict = dec.serialize_fast_ica(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, LatentDirichletAllocation):
        model_dict = dec.serialize_latent_dirichlet_allocation(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, MiniBatchNMF):
        model_dict = dec.serialize_minibatch_nmf(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, NMF):
        model_dict = dec.serialize_nmf(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, SparseCoder):
        model_dict = dec.serialize_sparse_coder(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, TruncatedSVD):
        model_dict = dec.serialize_truncated_svd(model)
        return serialize_version(model, model_dict)

    # Manifold
    elif isinstance(model, TSNE):
        model_dict = man.serialize_tsne(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, MDS):
        model_dict = man.serialize_mds(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, Isomap):
        model_dict = man.serialize_isomap(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, LocallyLinearEmbedding):
        model_dict = man.serialize_locally_linear_embedding(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, SpectralEmbedding):
        model_dict = man.serialize_spectral_embedding(model)
        return serialize_version(model, model_dict)
    elif 'UMAP' in man.__optionals__ and isinstance(model, UMAP):
        model_dict = man.serialize_umap(model)
        return serialize_version(model, model_dict)
    elif 'OpenTSNE' in man.__optionals__ and isinstance(model, (OpenTSNE, OpenTSNEsklearn)):
        model_dict = man.serialize_opentsne(model)
        return serialize_version(model, model_dict)
    elif 'OpenTSNE' in man.__optionals__ and isinstance(model, OpenTSNEEmbedding):
        model_dict = man.serialize_opentsne_embedding(model)
        return serialize_version(model, model_dict)
    elif 'OpenTSNE' in man.__optionals__ and isinstance(model, OpenPartialTSNEEmbedding):
        model_dict = man.serialize_opentsne_partial_embedding(model)
        return serialize_version(model, model_dict)

    # Neighbors
    elif isinstance(model, NearestNeighbors):
        model_dict = nei.serialize_nearest_neighbors(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, KDTree):
        model_dict = nei.serialize_kdtree(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, KernelDensity):
        model_dict = nei.serialize_kernel_density(model)
        return serialize_version(model, model_dict)
    elif 'NNDescent' in nei.__optionals__ and isinstance(model, NNDescent):
        model_dict = nei.serialize_nndescent(model)
        return serialize_version(model, model_dict)

    # Feature Extraction
    elif isinstance(model, DictVectorizer):
        model_dict = ext.serialize_dict_vectorizer(model)
        return serialize_version(model, model_dict)

    # Preprocess
    elif isinstance(model, LabelEncoder):
        model_dict = pre.serialize_label_encoder(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, LabelBinarizer):
        model_dict = pre.serialize_label_binarizer(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, MultiLabelBinarizer):
        model_dict = pre.serialize_multilabel_binarizer(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, MinMaxScaler):
        model_dict = pre.serialize_minmax_scaler(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, StandardScaler):
        model_dict = pre.serialize_standard_scaler(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, RobustScaler):
        model_dict = pre.serialize_robust_scaler(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, MaxAbsScaler):
        model_dict = pre.serialize_maxabs_scaler(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, KernelCenterer):
        model_dict = pre.serialize_kernel_centerer(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, OneHotEncoder):
        model_dict = pre.serialize_onehot_encoder(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, OrdinalEncoder):
        model_dict = pre.serialize_ordinal_encoder(model)
        return serialize_version(model, model_dict)

    # Applicability Domain
    elif isinstance(model, BoundingBoxApplicabilityDomain):
        model_dict = ad.serialize_bounding_box_applicability_domain(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, ConvexHullApplicabilityDomain):
        model_dict = ad.serialize_convex_hull_applicability_domain(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, PCABoundingBoxApplicabilityDomain):
        model_dict = ad.serialize_pca_bounding_box_applicability_domain(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, TopKatApplicabilityDomain):
        model_dict = ad.serialize_topkat_applicability_domain(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, LeverageApplicabilityDomain):
        model_dict = ad.serialize_leverage_applicability_domain(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, HotellingT2ApplicabilityDomain):
        model_dict = ad.serialize_hotelling_t2_applicability_domain(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, KernelDensityApplicabilityDomain):
        model_dict = ad.serialize_kernel_density_applicability_domain(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, IsolationForestApplicabilityDomain):
        model_dict = ad.serialize_isolation_forest_applicability_domain(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, CentroidDistanceApplicabilityDomain):
        model_dict = ad.serialize_centroid_distance_applicability_domain(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, KNNApplicabilityDomain):
        model_dict = ad.serialize_knn_applicability_domain(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, StandardizationApproachApplicabilityDomain):
        model_dict = ad.serialize_standardization_approach_applicability_domain(model)
        return serialize_version(model, model_dict)

    # Balancing
    elif 'imblearn' in ous.__optionals__ and isinstance(model, ClusterCentroids):
        model_dict = ous.serialize_cluster_centroids(model)
        return serialize_version(model, model_dict)
    elif 'imblearn' in ous.__optionals__ and isinstance(model, CondensedNearestNeighbour):
        model_dict = ous.serialize_condensed_nearest_neighbours(model)
        return serialize_version(model, model_dict)
    elif 'imblearn' in ous.__optionals__ and isinstance(model, EditedNearestNeighbours):
        model_dict = ous.serialize_edited_nearest_neighbours(model)
        return serialize_version(model, model_dict)
    elif 'imblearn' in ous.__optionals__ and isinstance(model, RepeatedEditedNearestNeighbours):
        model_dict = ous.serialize_repeated_edited_nearest_neighbours(model)
        return serialize_version(model, model_dict)
    elif 'imblearn' in ous.__optionals__ and isinstance(model, AllKNN):
        model_dict = ous.serialize_all_knn(model)
        return serialize_version(model, model_dict)
    elif 'imblearn' in ous.__optionals__ and isinstance(model, InstanceHardnessThreshold):
        model_dict = ous.serialize_instance_hardness_threshold(model)
        return serialize_version(model, model_dict)
    elif 'imblearn' in ous.__optionals__ and isinstance(model, NearMiss):
        model_dict = ous.serialize_near_miss(model)
        return serialize_version(model, model_dict)
    elif 'imblearn' in ous.__optionals__ and isinstance(model, NeighbourhoodCleaningRule):
        model_dict = ous.serialize_neighbourhood_cleaning_rule(model)
        return serialize_version(model, model_dict)
    elif 'imblearn' in ous.__optionals__ and isinstance(model, OneSidedSelection):
        model_dict = ous.serialize_one_sided_selection(model)
        return serialize_version(model, model_dict)
    elif 'imblearn' in ous.__optionals__ and isinstance(model, RandomUnderSampler):
        model_dict = ous.serialize_random_under_sampler(model)
        return serialize_version(model, model_dict)
    elif 'imblearn' in ous.__optionals__ and isinstance(model, TomekLinks):
        model_dict = ous.serialize_tomek_links(model)
        return serialize_version(model, model_dict)
    elif 'imblearn' in ous.__optionals__ and isinstance(model, RandomOverSampler):
        model_dict = ous.serialize_random_over_sampler(model)
        return serialize_version(model, model_dict)
    elif 'imblearn' in ous.__optionals__ and isinstance(model, SMOTENC):
        model_dict = ous.serialize_smotenc(model)
        return serialize_version(model, model_dict)
    elif 'imblearn' in ous.__optionals__ and isinstance(model, SMOTEN):
        model_dict = ous.serialize_smoten(model)
        return serialize_version(model, model_dict)
    elif 'imblearn' in ous.__optionals__ and isinstance(model, SMOTE):
        model_dict = ous.serialize_smote(model)
        return serialize_version(model, model_dict)
    elif 'imblearn' in ous.__optionals__ and isinstance(model, ADASYN):
        model_dict = ous.serialize_adasyn(model)
        return serialize_version(model, model_dict)
    elif 'imblearn' in ous.__optionals__ and isinstance(model, BorderlineSMOTE):
        model_dict = ous.serialize_borderline_smote(model)
        return serialize_version(model, model_dict)
    elif 'imblearn' in ous.__optionals__ and isinstance(model, KMeansSMOTE):
        model_dict = ous.serialize_kmeans_smote(model)
        return serialize_version(model, model_dict)
    elif 'imblearn' in ous.__optionals__ and isinstance(model, SVMSMOTE):
        model_dict = ous.serialize_svm_smote(model)
        return serialize_version(model, model_dict)
    elif 'imblearn' in ous.__optionals__ and isinstance(model, SMOTEENN):
        model_dict = ous.serialize_smote_enn(model)
        return serialize_version(model, model_dict)
    elif 'imblearn' in ous.__optionals__ and isinstance(model, SMOTETomek):
        model_dict = ous.serialize_smote_tomek(model)
        return serialize_version(model, model_dict)

    # Pipeline
    elif isinstance(model, Pipeline):
        model_dict = ppl.serialize_pipeline(model)
        return serialize_version(model, model_dict)

    # Otherwise
    else:
        raise ModelNotSupported('This model type is not currently supported. Email support@mlrequest.com to request a feature or report a bug.')


def deserialize_model(model_dict: Dict):
    """Instantiate a machine learning model from a previously serialized model.

    :param model_dict: dictionary of the previously serialized model
    """
    # Verify model is fitted
    if 'unfitted' in model_dict.keys() and model_dict['unfitted']:
        check_version(model_dict)
        return deserialize_unfitted_model(model_dict)

    # Classification
    if model_dict['meta'] == 'lr':
        check_version(model_dict)
        return clf.deserialize_logistic_regression(model_dict)
    elif model_dict['meta'] == 'bernoulli-nb':
        check_version(model_dict)
        return clf.deserialize_bernoulli_nb(model_dict)
    elif model_dict['meta'] == 'gaussian-nb':
        check_version(model_dict)
        return clf.deserialize_gaussian_nb(model_dict)
    elif model_dict['meta'] == 'multinomial-nb':
        check_version(model_dict)
        return clf.deserialize_multinomial_nb(model_dict)
    elif model_dict['meta'] == 'complement-nb':
        check_version(model_dict)
        return clf.deserialize_complement_nb(model_dict)
    elif model_dict['meta'] == 'lda':
        check_version(model_dict)
        return clf.deserialize_lda(model_dict)
    elif model_dict['meta'] == 'qda':
        check_version(model_dict)
        return clf.deserialize_qda(model_dict)
    elif model_dict['meta'] == 'svm':
        check_version(model_dict)
        return clf.deserialize_svm(model_dict)
    elif model_dict['meta'] == 'perceptron':
        check_version(model_dict)
        return clf.deserialize_perceptron(model_dict)
    elif model_dict['meta'] == 'decision-tree':
        check_version(model_dict)
        return clf.deserialize_decision_tree(model_dict)
    elif model_dict['meta'] == 'gb':
        check_version(model_dict)
        return clf.deserialize_gradient_boosting(model_dict)
    elif model_dict['meta'] == 'rf':
        check_version(model_dict)
        return clf.deserialize_random_forest(model_dict)
    elif model_dict['meta'] == 'mlp':
        check_version(model_dict)
        return clf.deserialize_mlp(model_dict)
    elif model_dict['meta'] == 'xgboost-classifier':
        check_version(model_dict)
        return clf.deserialize_xgboost_classifier(model_dict)
    elif model_dict['meta'] == 'xgboost-rf-classifier':
        check_version(model_dict)
        return clf.deserialize_xgboost_rf_classifier(model_dict)
    elif model_dict['meta'] == 'lightgbm-classifier':
        check_version(model_dict)
        return clf.deserialize_lightgbm_classifier(model_dict)
    elif model_dict['meta'] == 'catboost-classifier':
        check_version(model_dict)
        return clf.deserialize_catboost_classifier(model_dict)
    elif model_dict['meta'] == 'adaboost-classifier':
        check_version(model_dict)
        return clf.deserialize_adaboost_classifier(model_dict)
    elif model_dict['meta'] == 'bagging-classifier':
        check_version(model_dict)
        return clf.deserialize_bagging_classifier(model_dict)
    elif model_dict['meta'] == 'extra-tree-cls':
        check_version(model_dict)
        return clf.deserialize_extra_tree_classifier(model_dict)
    elif model_dict['meta'] == 'extratrees-classifier':
        check_version(model_dict)
        return clf.deserialize_extratrees_classifier(model_dict)
    elif model_dict['meta'] == 'isolation-forest':
        check_version(model_dict)
        return clf.deserialize_isolation_forest(model_dict)
    elif model_dict['meta'] == 'random-trees-embedding':
        check_version(model_dict)
        return clf.deserialize_random_trees_embedding(model_dict)
    elif model_dict['meta'] == 'nearest-neighbour-classifier':
        check_version(model_dict)
        return clf.deserialize_nearest_neighbour_classifier(model_dict)
    elif model_dict['meta'] == 'stacking-classifier':
        check_version(model_dict)
        return clf.deserialize_stacking_classifier(model_dict)
    elif model_dict['meta'] == 'voting-classifier':
        check_version(model_dict)
        return clf.deserialize_voting_classifier(model_dict)

    # Regression
    elif model_dict['meta'] == 'linear-regression':
        check_version(model_dict)
        return reg.deserialize_linear_regressor(model_dict)
    elif model_dict['meta'] == 'lasso-regression':
        check_version(model_dict)
        return reg.deserialize_lasso_regressor(model_dict)
    elif model_dict['meta'] == 'elasticnet-regression':
        check_version(model_dict)
        return reg.deserialize_elastic_regressor(model_dict)
    elif model_dict['meta'] == 'ridge-regression':
        check_version(model_dict)
        return reg.deserialize_ridge_regressor(model_dict)
    elif model_dict['meta'] == 'svr':
        check_version(model_dict)
        return reg.deserialize_svr(model_dict)
    elif model_dict['meta'] == 'decision-tree-regression':
        check_version(model_dict)
        return reg.deserialize_decision_tree_regressor(model_dict)
    elif model_dict['meta'] == 'gb-regression':
        check_version(model_dict)
        return reg.deserialize_gradient_boosting_regressor(model_dict)
    elif model_dict['meta'] == 'rf-regression':
        check_version(model_dict)
        return reg.deserialize_random_forest_regressor(model_dict)
    elif model_dict['meta'] == 'mlp-regression':
        check_version(model_dict)
        return reg.deserialize_mlp_regressor(model_dict)
    elif model_dict['meta'] == 'xgboost-ranker':
        check_version(model_dict)
        return reg.deserialize_xgboost_ranker(model_dict)
    elif model_dict['meta'] == 'xgboost-regressor':
        check_version(model_dict)
        return reg.deserialize_xgboost_regressor(model_dict)
    elif model_dict['meta'] == 'xgboost-rf-regressor':
        check_version(model_dict)
        return reg.deserialize_xgboost_rf_regressor(model_dict)
    elif model_dict['meta'] == 'lightgbm-regressor':
        check_version(model_dict)
        return reg.deserialize_lightgbm_regressor(model_dict)
    elif model_dict['meta'] == 'lightgbm-ranker':
        check_version(model_dict)
        return reg.deserialize_lightgbm_ranker(model_dict)
    elif model_dict['meta'] == 'catboost-regressor':
        check_version(model_dict)
        return reg.deserialize_catboost_regressor(model_dict)
    elif model_dict['meta'] == 'catboost-ranker':
        check_version(model_dict)
        return reg.deserialize_catboost_ranker(model_dict)
    elif model_dict['meta'] == 'adaboost-regressor':
        check_version(model_dict)
        return reg.deserialize_adaboost_regressor(model_dict)
    elif model_dict['meta'] == 'bagging-regression':
        check_version(model_dict)
        return reg.deserialize_bagging_regressor(model_dict)
    elif model_dict['meta'] == 'extra-tree-reg':
        check_version(model_dict)
        return reg.deserialize_extra_tree_regressor(model_dict)
    elif model_dict['meta'] == 'extratrees-regressor':
        check_version(model_dict)
        return reg.deserialize_extratrees_regressor(model_dict)
    elif model_dict['meta'] == 'nearest-neighbour-regressor':
        check_version(model_dict)
        return reg.deserialize_nearest_neighbour_regressor(model_dict)
    elif model_dict['meta'] == 'stacking-regressor':
        check_version(model_dict)
        return reg.deserialize_stacking_regressor(model_dict)
    elif model_dict['meta'] == 'voting-regressor':
        check_version(model_dict)
        return reg.deserialize_voting_regressor(model_dict)

    # Clustering
    elif model_dict['meta'] == 'affinity-propagation':
        check_version(model_dict)
        return clus.deserialize_affinity_propagation(model_dict)
    elif model_dict['meta'] == 'agglomerative-clustering':
        check_version(model_dict)
        return clus.deserialize_agglomerative_clustering(model_dict)
    elif model_dict['meta'] == 'feature-agglomeration':
        check_version(model_dict)
        return clus.deserialize_feature_agglomeration(model_dict)
    elif model_dict['meta'] == 'dbscan':
        check_version(model_dict)
        return clus.deserialize_dbscan(model_dict)
    elif model_dict['meta'] == 'meanshift':
        check_version(model_dict)
        return clus.deserialize_meanshift(model_dict)
    elif model_dict['meta'] == 'kmeans':
        check_version(model_dict)
        return clus.deserialize_kmeans(model_dict)
    elif model_dict['meta'] == 'minibatch-kmeans':
        check_version(model_dict)
        return clus.deserialize_minibatch_kmeans(model_dict)
    elif model_dict['meta'] == 'optics':
        check_version(model_dict)
        return clus.deserialize_optics(model_dict)
    elif model_dict['meta'] == 'spectral-clustering':
        check_version(model_dict)
        return clus.deserialize_spectral_clustering(model_dict)
    elif model_dict['meta'] == 'spectral-biclustering':
        check_version(model_dict)
        return clus.deserialize_spectral_biclustering(model_dict)
    elif model_dict['meta'] == 'spectral-coclustering':
        check_version(model_dict)
        return clus.deserialize_spectral_coclustering(model_dict)
    elif model_dict['meta'] == 'kmodes':
        check_version(model_dict)
        return clus.deserialize_kmodes(model_dict)
    elif model_dict['meta'] == 'kprototypes':
        check_version(model_dict)
        return clus.deserialize_kprototypes(model_dict)
    elif model_dict['meta'] == 'birch':
        check_version(model_dict)
        return clus.deserialize_birch(model_dict)
    elif model_dict['meta'] == 'bisecting-kmeans':
        check_version(model_dict)
        return clus.deserialize_bisecting_kmeans(model_dict)
    elif model_dict['meta'] == 'hdbscan':
        check_version(model_dict)
        return clus.deserialize_hdbscan(model_dict)

    # Cross-decomposition
    elif model_dict['meta'] == 'cca':
        check_version(model_dict)
        return crdec.deserialize_cca(model_dict)
    elif model_dict['meta'] == 'pls-canonical':
        check_version(model_dict)
        return crdec.deserialize_pls_canonical(model_dict)
    elif model_dict['meta'] == 'pls-regression':
        check_version(model_dict)
        return crdec.deserialize_pls_regression(model_dict)
    elif model_dict['meta'] == 'pls-svd':
        check_version(model_dict)
        return crdec.deserialize_pls_svd(model_dict)

    # Decomposition
    elif model_dict['meta'] == 'pca':
        check_version(model_dict)
        return dec.deserialize_pca(model_dict)
    elif model_dict['meta'] == 'kernel-pca':
        check_version(model_dict)
        return  dec.deserialize_kernel_pca(model_dict)
    elif model_dict['meta'] == 'incremental-pca':
        check_version(model_dict)
        return  dec.deserialize_incremental_pca(model_dict)
    elif model_dict['meta'] == 'sparse-pca':
        check_version(model_dict)
        return  dec.deserialize_sparse_pca(model_dict)
    elif model_dict['meta'] == 'minibatch-sparse-pca':
        check_version(model_dict)
        return  dec.deserialize_minibatch_sparse_pca(model_dict)
    elif model_dict['meta'] == 'dictionary-learning':
        check_version(model_dict)
        return  dec.deserialize_dictionary_learning(model_dict)
    elif model_dict['meta'] == 'minibatch-dictionary-learning':
        check_version(model_dict)
        return  dec.deserialize_minibatch_dictionary_learning(model_dict)
    elif model_dict['meta'] == 'factor-analysis':
        check_version(model_dict)
        return  dec.deserialize_factor_analysis(model_dict)
    elif model_dict['meta'] == 'fast-ica':
        check_version(model_dict)
        return  dec.deserialize_fast_ica(model_dict)
    elif model_dict['meta'] == 'latent-dirichlet-allocation':
        check_version(model_dict)
        return  dec.deserialize_latent_dirichlet_allocation(model_dict)
    elif model_dict['meta'] == 'nmf':
        check_version(model_dict)
        return  dec.deserialize_nmf(model_dict)
    elif model_dict['meta'] == 'minibatch-nmf':
        check_version(model_dict)
        return  dec.deserialize_minibatch_nmf(model_dict)
    elif model_dict['meta'] == 'sparse-coder':
        check_version(model_dict)
        return  dec.deserialize_sparse_coder(model_dict)
    elif model_dict['meta'] == 'truncated-svd':
        check_version(model_dict)
        return  dec.deserialize_truncated_svd(model_dict)

    # Manifold
    elif model_dict['meta'] == 'tsne':
        check_version(model_dict)
        return  man.deserialize_tsne(model_dict)
    elif model_dict['meta'] == 'mds':
        check_version(model_dict)
        return  man.deserialize_mds(model_dict)
    elif model_dict['meta'] == 'isomap':
        check_version(model_dict)
        return  man.deserialize_isomap(model_dict)
    elif model_dict['meta'] == 'locally-linear-embedding':
        check_version(model_dict)
        return  man.deserialize_locally_linear_embedding(model_dict)
    elif model_dict['meta'] == 'spectral-embedding':
        check_version(model_dict)
        return  man.deserialize_spectral_embedding(model_dict)
    elif model_dict['meta'] == 'umap':
        check_version(model_dict)
        return  man.deserialize_umap(model_dict)
    elif model_dict['meta'] == 'openTSNE':
        check_version(model_dict)
        return  man.deserialize_opentsne(model_dict)
    elif model_dict['meta'] == 'openTSNEEmbedding':
        check_version(model_dict)
        return  man.deserialize_opentsne_embedding(model_dict)
    elif model_dict['meta'] == 'openTSNEPartialEmbedding':
        check_version(model_dict)
        return  man.deserialize_opentsne_partial_embedding(model_dict)

    # Neighbors
    elif model_dict['meta'] == 'nearest-neighbors':
        check_version(model_dict)
        return  nei.deserialize_nearest_neighbors(model_dict)
    elif model_dict['meta'] == 'kdtree':
        check_version(model_dict)
        return  nei.deserialize_kdtree(model_dict)
    elif model_dict['meta'] == 'kernel-density':
        check_version(model_dict)
        return  nei.deserialize_kernel_density(model_dict)
    elif model_dict['meta'] == 'nn-descent':
        check_version(model_dict)
        return  nei.deserialize_nndescent(model_dict)

    # Feature Extraction
    elif model_dict['meta'] == 'dict-vectorizer':
        check_version(model_dict)
        return ext.deserialize_dict_vectorizer(model_dict)

    # Preprocess
    elif model_dict['meta'] == 'label-encoder':
        check_version(model_dict)
        return pre.deserialize_label_encoder(model_dict)
    elif model_dict['meta'] == 'label-binarizer':
        check_version(model_dict)
        return pre.deserialize_label_binarizer(model_dict)
    elif model_dict['meta'] == 'multilabel-binarizer':
        check_version(model_dict)
        return pre.deserialize_multilabel_binarizer(model_dict)
    elif model_dict['meta'] == 'minmax-scaler':
        check_version(model_dict)
        return pre.deserialize_minmax_scaler(model_dict)
    elif model_dict['meta'] == 'standard-scaler':
        check_version(model_dict)
        return pre.deserialize_standard_scaler(model_dict)
    elif model_dict['meta'] == 'robust-scaler':
        check_version(model_dict)
        return pre.deserialize_robust_scaler(model_dict)
    elif model_dict['meta'] == 'maxabs-scaler':
        check_version(model_dict)
        return pre.deserialize_maxabs_scaler(model_dict)
    elif model_dict['meta'] == 'kernel-centerer':
        check_version(model_dict)
        return pre.deserialize_kernel_centerer(model_dict)
    elif model_dict['meta'] == 'onehot-encoder':
        check_version(model_dict)
        return pre.deserialize_onehot_encoder(model_dict)
    elif model_dict['meta'] == 'ordinal-encoder':
        check_version(model_dict)
        return pre.deserialize_ordinal_encoder(model_dict)

    # Applicability Domain
    elif model_dict['meta'] == 'bounding-box-ad':
        check_version(model_dict)
        return ad.deserialize_bounding_box_applicability_domain(model_dict)
    elif model_dict['meta'] == 'convex-hull-ad':
        check_version(model_dict)
        return ad.deserialize_convex_hull_applicability_domain(model_dict)
    elif model_dict['meta'] == 'pca-bounding-box-ad':
        check_version(model_dict)
        return ad.deserialize_pca_bounding_box_applicability_domain(model_dict)
    elif model_dict['meta'] == 'topkat-ad':
        check_version(model_dict)
        return ad.deserialize_topkat_applicability_domain(model_dict)
    elif model_dict['meta'] == 'leverage-ad':
        check_version(model_dict)
        return ad.deserialize_leverage_applicability_domain(model_dict)
    elif model_dict['meta'] == 'hotelling-t2-ad':
        check_version(model_dict)
        return ad.deserialize_hotelling_t2_applicability_domain(model_dict)
    elif model_dict['meta'] == 'kernel-density-ad':
        check_version(model_dict)
        return ad.deserialize_kernel_density_applicability_domain(model_dict)
    elif model_dict['meta'] == 'isolation-forest-ad':
        check_version(model_dict)
        return ad.deserialize_isolation_forest_applicability_domain(model_dict)
    elif model_dict['meta'] == 'centroid-distance-ad':
        check_version(model_dict)
        return ad.deserialize_centroid_distance_applicability_domain(model_dict)
    elif model_dict['meta'] == 'knn-ad':
        check_version(model_dict)
        return ad.deserialize_knn_applicability_domain(model_dict)
    elif model_dict['meta'] == 'standardization-approach-ad':
        check_version(model_dict)
        return ad.deserialize_standardization_approach_applicability_domain(model_dict)

    # Balancing
    elif 'imblearn' in ous.__optionals__ and model_dict['meta'] == 'cluster-centroids':
        check_version(model_dict)
        return ous.deserialize_cluster_centroids(model_dict)
    elif 'imblearn' in ous.__optionals__ and model_dict['meta'] == 'condensed-nearest-neighbours':
        check_version(model_dict)
        return ous.deserialize_condensed_nearest_neighbours(model_dict)
    elif 'imblearn' in ous.__optionals__ and model_dict['meta'] == 'edited-nearest-neighbours':
        check_version(model_dict)
        return ous.deserialize_edited_nearest_neighbours(model_dict)
    elif 'imblearn' in ous.__optionals__ and model_dict['meta'] == 'repeated-edited-nearest-neighbours':
        check_version(model_dict)
        return ous.deserialize_repeated_edited_nearest_neighbours(model_dict)
    elif 'imblearn' in ous.__optionals__ and model_dict['meta'] == 'all-knn':
        check_version(model_dict)
        return ous.deserialize_all_knn(model_dict)
    elif 'imblearn' in ous.__optionals__ and model_dict['meta'] == 'instance-hardness-threshold':
        check_version(model_dict)
        return ous.deserialize_instance_hardness_threshold(model_dict)
    elif 'imblearn' in ous.__optionals__ and model_dict['meta'] == 'near-miss':
        check_version(model_dict)
        return ous.deserialize_near_miss(model_dict)
    elif 'imblearn' in ous.__optionals__ and model_dict['meta'] == 'neighbourhood-cleaning-rule':
        check_version(model_dict)
        return ous.deserialize_neighbourhood_cleaning_rule(model_dict)
    elif 'imblearn' in ous.__optionals__ and model_dict['meta'] == 'one-sided-selection':
        check_version(model_dict)
        return ous.deserialize_one_sided_selection(model_dict)
    elif 'imblearn' in ous.__optionals__ and model_dict['meta'] == 'random-under-sampler':
        check_version(model_dict)
        return ous.deserialize_random_under_sampler(model_dict)
    elif 'imblearn' in ous.__optionals__ and model_dict['meta'] == 'tomek-links':
        check_version(model_dict)
        return ous.deserialize_tomek_links(model_dict)
    elif 'imblearn' in ous.__optionals__ and model_dict['meta'] == 'random-over-sampler':
        check_version(model_dict)
        return ous.deserialize_random_over_sampler(model_dict)
    elif 'imblearn' in ous.__optionals__ and model_dict['meta'] == 'smotenc':
        check_version(model_dict)
        return ous.deserialize_smotenc(model_dict)
    elif 'imblearn' in ous.__optionals__ and model_dict['meta'] == 'smoten':
        check_version(model_dict)
        return ous.deserialize_smoten(model_dict)
    elif 'imblearn' in ous.__optionals__ and model_dict['meta'] == 'smote':
        check_version(model_dict)
        return ous.deserialize_smote(model_dict)
    elif 'imblearn' in ous.__optionals__ and model_dict['meta'] == 'adasyn':
        check_version(model_dict)
        return ous.deserialize_adasyn(model_dict)
    elif 'imblearn' in ous.__optionals__ and model_dict['meta'] == 'borderline-smote':
        check_version(model_dict)
        return ous.deserialize_borderline_smote(model_dict)
    elif 'imblearn' in ous.__optionals__ and model_dict['meta'] == 'kmeans-smote':
        check_version(model_dict)
        return ous.deserialize_kmeans_smote(model_dict)
    elif 'imblearn' in ous.__optionals__ and model_dict['meta'] == 'svm-smote':
        check_version(model_dict)
        return ous.deserialize_svm_smote(model_dict)
    elif 'imblearn' in ous.__optionals__ and model_dict['meta'] == 'smote-enn':
        check_version(model_dict)
        return ous.deserialize_smote_enn(model_dict)
    elif 'imblearn' in ous.__optionals__ and model_dict['meta'] == 'smote-tomek':
        check_version(model_dict)
        return ous.deserialize_smote_tomek(model_dict)

    # Pipeline
    elif model_dict['meta'] == 'pipeline':
        check_version(model_dict)
        return ppl.deserialize_pipeline(model_dict)

    # Otherwise
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
