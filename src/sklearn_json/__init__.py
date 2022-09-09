# -*- coding: utf-8 -*-

import json

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
                                   OneHotEncoder)
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
from sklearn.neighbors import NearestNeighbors, KDTree
from xgboost import XGBRegressor, XGBClassifier, XGBRFRegressor, XGBRFClassifier, XGBRanker
from lightgbm import LGBMClassifier, LGBMRegressor, LGBMRanker
from catboost import CatBoostClassifier, CatBoostRegressor, CatBoostRanker, Pool
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes
from hdbscan import HDBSCAN
from pynndescent import NNDescent
from umap import UMAP

from . import classification as clf
from . import regression as reg
from . import feature_extraction as ext
from . import preprocessing as pre
from . import cluster as clus
from . import decomposition as dec
from . import manifold as man
from . import neighbors as nei
from . import cross_decomposition as crdec


__version__ = '0.1.4'


def serialize_model(model, catboost_data: Pool = None):
    # Classification
    if isinstance(model, LogisticRegression):
        return clf.serialize_logistic_regression(model)
    elif isinstance(model, BernoulliNB):
        return clf.serialize_bernoulli_nb(model)
    elif isinstance(model, GaussianNB):
        return clf.serialize_gaussian_nb(model)
    elif isinstance(model, MultinomialNB):
        return clf.serialize_multinomial_nb(model)
    elif isinstance(model, ComplementNB):
        return clf.serialize_complement_nb(model)
    elif isinstance(model, discriminant_analysis.LinearDiscriminantAnalysis):
        return clf.serialize_lda(model)
    elif isinstance(model, discriminant_analysis.QuadraticDiscriminantAnalysis):
        return clf.serialize_qda(model)
    elif isinstance(model, svm.SVC):
        return clf.serialize_svm(model)
    elif isinstance(model, Perceptron):
        return clf.serialize_perceptron(model)
    elif isinstance(model, DecisionTreeClassifier):
        return clf.serialize_decision_tree(model)
    elif isinstance(model, GradientBoostingClassifier):
        return clf.serialize_gradient_boosting(model)
    elif isinstance(model, RandomForestClassifier):
        return clf.serialize_random_forest(model)
    elif isinstance(model, MLPClassifier):
        return clf.serialize_mlp(model)
    elif isinstance(model, XGBClassifier):
        return clf.serialize_xgboost_classifier(model)
    elif isinstance(model, XGBRFClassifier):
        return clf.serialize_xgboost_rf_classifier(model)
    elif isinstance(model, LGBMClassifier):
        return clf.serialize_lightgbm_classifier(model)
    elif isinstance(model, CatBoostClassifier):
        return clf.serialize_catboost_classifier(model, catboost_data)
    elif isinstance(model, AdaBoostClassifier):
        return clf.serialize_adaboost_classifier(model)
    elif isinstance(model, BaggingClassifier):
        return clf.serialize_bagging_classifier(model)
    elif isinstance(model, ExtraTreeClassifier):
        return clf.serialize_extra_tree_classifier(model)
    elif isinstance(model, ExtraTreesClassifier):
        return clf.serialize_extratrees_classifier(model)
    elif isinstance(model, IsolationForest):
        return clf.serialize_isolation_forest(model)
    elif isinstance(model, RandomTreesEmbedding):
        return clf.serialize_random_trees_embedding(model)

    # Regression
    elif isinstance(model, LinearRegression):
        return reg.serialize_linear_regressor(model)
    elif isinstance(model, Lasso):
        return reg.serialize_lasso_regressor(model)
    elif isinstance(model, ElasticNet):
        return reg.serialize_elastic_regressor(model)
    elif isinstance(model, Ridge):
        return reg.serialize_ridge_regressor(model)
    elif isinstance(model, SVR):
        return reg.serialize_svr(model)
    elif isinstance(model, ExtraTreeRegressor):
        return reg.serialize_extra_tree_regressor(model)
    elif isinstance(model, DecisionTreeRegressor):
        return reg.serialize_decision_tree_regressor(model)
    elif isinstance(model, GradientBoostingRegressor):
        return reg.serialize_gradient_boosting_regressor(model)
    elif isinstance(model, RandomForestRegressor):
        return reg.serialize_random_forest_regressor(model)
    elif isinstance(model, ExtraTreesRegressor):
        return reg.serialize_extratrees_regressor(model)
    elif isinstance(model, MLPRegressor):
        return reg.serialize_mlp_regressor(model)
    elif isinstance(model, XGBRanker):
        return reg.serialize_xgboost_ranker(model)
    elif isinstance(model, XGBRegressor):
        return reg.serialize_xgboost_regressor(model)
    elif isinstance(model, XGBRFRegressor):
        return reg.serialize_xgboost_rf_regressor(model)
    elif isinstance(model, LGBMRegressor):
        return reg.serialize_lightgbm_regressor(model)
    elif isinstance(model, LGBMRanker):
        return reg.serialize_lightgbm_ranker(model)
    elif isinstance(model, CatBoostRegressor):
        return reg.serialize_catboost_regressor(model, catboost_data)
    elif isinstance(model, CatBoostRanker):
        return reg.serialize_catboost_ranker(model, catboost_data)
    elif isinstance(model, AdaBoostRegressor):
        return reg.serialize_adaboost_regressor(model)
    elif isinstance(model, BaggingRegressor):
        return reg.serialize_bagging_regressor(model)

    # Clustering
    elif isinstance(model, FeatureAgglomeration):
        return clus.serialize_feature_agglomeration(model)
    elif isinstance(model, AffinityPropagation):
        return clus.serialize_affinity_propagation(model)
    elif isinstance(model, AgglomerativeClustering):
        return clus.serialize_agglomerative_clustering(model)
    elif isinstance(model, DBSCAN):
        return clus.serialize_dbscan(model)
    elif isinstance(model, MeanShift):
        return clus.serialize_meanshift(model)
    elif isinstance(model, BisectingKMeans):
        return clus.serialize_bisecting_kmeans(model)
    elif isinstance(model, MiniBatchKMeans):
        return clus.serialize_minibatch_kmeans(model)
    elif isinstance(model, KMeans):
        return clus.serialize_kmeans(model)
    elif isinstance(model, OPTICS):
        return clus.serialize_optics(model)
    elif isinstance(model, SpectralClustering):
        return clus.serialize_spectral_clustering(model)
    elif isinstance(model, SpectralBiclustering):
        return clus.serialize_spectral_biclustering(model)
    elif isinstance(model, SpectralCoclustering):
        return clus.serialize_spectral_coclustering(model)
    elif isinstance(model, KPrototypes):
        return clus.serialize_kprototypes(model)
    elif isinstance(model, KModes):
        return clus.serialize_kmodes(model)
    elif isinstance(model, Birch):
        return clus.serialize_birch(model)
    elif isinstance(model, HDBSCAN):
        return clus.serialize_hdbscan(model)

    # Decomposition
    elif isinstance(model, CCA):
        return crdec.serialize_cca(model)
    elif isinstance(model, PLSCanonical):
        return crdec.serialize_pls_canonical(model)
    elif isinstance(model, PLSRegression):
        return crdec.serialize_pls_regression(model)
    elif isinstance(model, PLSSVD):
        return crdec.serialize_pls_svd(model)

    # Decomposition
    elif isinstance(model, PCA):
        return dec.serialize_pca(model)
    elif isinstance(model, KernelPCA):
        return dec.serialize_kernel_pca(model)
    elif isinstance(model, IncrementalPCA):
        return dec.serialize_incremental_pca(model)
    elif isinstance(model, MiniBatchSparsePCA):
        return dec.serialize_minibatch_sparse_pca(model)
    elif isinstance(model, SparsePCA):
        return dec.serialize_sparse_pca(model)
    elif isinstance(model, MiniBatchDictionaryLearning):
        return dec.serialize_minibatch_dictionary_learning(model)
    elif isinstance(model, DictionaryLearning):
        return dec.serialize_dictionary_learning(model)
    elif isinstance(model, FactorAnalysis):
        return dec.serialize_factor_analysis(model)
    elif isinstance(model, FastICA):
        return dec.serialize_fast_ica(model)
    elif isinstance(model, LatentDirichletAllocation):
        return dec.serialize_latent_dirichlet_allocation(model)
    elif isinstance(model, MiniBatchNMF):
        return dec.serialize_minibatch_nmf(model)
    elif isinstance(model, NMF):
        return dec.serialize_nmf(model)
    elif isinstance(model, SparseCoder):
        return dec.serialize_sparse_coder(model)
    elif isinstance(model, TruncatedSVD):
        return dec.serialize_truncated_svd(model)

    # Manifold
    elif isinstance(model, TSNE):
        return man.serialize_tsne(model)
    elif isinstance(model, MDS):
        return man.serialize_mds(model)
    elif isinstance(model, Isomap):
        return man.serialize_isomap(model)
    elif isinstance(model, LocallyLinearEmbedding):
        return man.serialize_locally_linear_embedding(model)
    elif isinstance(model, SpectralEmbedding):
        return man.serialize_spectral_embedding(model)
    elif isinstance(model, UMAP):
        return man.serialize_umap(model)

    # Neighbors
    elif isinstance(model, NearestNeighbors):
        return nei.serialize_nearest_neighbors(model)
    elif isinstance(model, KDTree):
        return nei.serialize_kdtree(model)
    elif isinstance(model, NNDescent):
        return nei.serialize_nndescent(model)

    # Feature Extraction
    elif isinstance(model, DictVectorizer):
        return ext.serialize_dict_vectorizer(model)

    # Preprocess
    elif isinstance(model, LabelEncoder):
        return pre.serialize_label_encoder(model)
    elif isinstance(model, LabelBinarizer):
        return pre.serialize_label_binarizer(model)
    elif isinstance(model, MultiLabelBinarizer):
        return pre.serialize_multilabel_binarizer(model)
    elif isinstance(model, MinMaxScaler):
        return pre.serialize_minmax_scaler(model)
    elif isinstance(model, StandardScaler):
        return pre.serialize_standard_scaler(model)
    elif isinstance(model, KernelCenterer):
        return pre.serialize_kernel_centerer(model)
    elif isinstance(model, OneHotEncoder):
        return pre.serialize_onehot_encoder(model)

    # Otherwise
    else:
        raise ModellNotSupported('This model type is not currently supported. Email support@mlrequest.com to request a feature or report a bug.')


def deserialize_model(model_dict):
    # Classification
    if model_dict['meta'] == 'lr':
        return clf.deserialize_logistic_regression(model_dict)
    elif model_dict['meta'] == 'bernoulli-nb':
        return clf.deserialize_bernoulli_nb(model_dict)
    elif model_dict['meta'] == 'gaussian-nb':
        return clf.deserialize_gaussian_nb(model_dict)
    elif model_dict['meta'] == 'multinomial-nb':
        return clf.deserialize_multinomial_nb(model_dict)
    elif model_dict['meta'] == 'complement-nb':
        return clf.deserialize_complement_nb(model_dict)
    elif model_dict['meta'] == 'lda':
        return clf.deserialize_lda(model_dict)
    elif model_dict['meta'] == 'qda':
        return clf.deserialize_qda(model_dict)
    elif model_dict['meta'] == 'svm':
        return clf.deserialize_svm(model_dict)
    elif model_dict['meta'] == 'perceptron':
        return clf.deserialize_perceptron(model_dict)
    elif model_dict['meta'] == 'decision-tree':
        return clf.deserialize_decision_tree(model_dict)
    elif model_dict['meta'] == 'gb':
        return clf.deserialize_gradient_boosting(model_dict)
    elif model_dict['meta'] == 'rf':
        return clf.deserialize_random_forest(model_dict)
    elif model_dict['meta'] == 'mlp':
        return clf.deserialize_mlp(model_dict)
    elif model_dict['meta'] == 'xgboost-classifier':
        return clf.deserialize_xgboost_classifier(model_dict)
    elif model_dict['meta'] == 'xgboost-rf-classifier':
        return clf.deserialize_xgboost_rf_classifier(model_dict)
    elif model_dict['meta'] == 'lightgbm-classifier':
        return clf.deserialize_lightgbm_classifier(model_dict)
    elif model_dict['meta'] == 'catboost-classifier':
        return clf.deserialize_catboost_classifier(model_dict)
    elif model_dict['meta'] == 'adaboost-classifier':
        return clf.deserialize_adaboost_classifier(model_dict)
    elif model_dict['meta'] == 'bagging-classifier':
        return clf.deserialize_bagging_classifier(model_dict)
    elif model_dict['meta'] == 'extra-tree-cls':
        return clf.deserialize_extra_tree_classifier(model_dict)
    elif model_dict['meta'] == 'extratrees-classifier':
        return clf.deserialize_extratrees_classifier(model_dict)
    elif model_dict['meta'] == 'isolation-forest':
        return clf.deserialize_isolation_forest(model_dict)
    elif model_dict['meta'] == 'random-trees-embedding':
        return clf.deserialize_random_trees_embedding(model_dict)

    # Regression
    elif model_dict['meta'] == 'linear-regression':
        return reg.deserialize_linear_regressor(model_dict)
    elif model_dict['meta'] == 'lasso-regression':
        return reg.deserialize_lasso_regressor(model_dict)
    elif model_dict['meta'] == 'elasticnet-regression':
        return reg.deserialize_elastic_regressor(model_dict)
    elif model_dict['meta'] == 'ridge-regression':
        return reg.deserialize_ridge_regressor(model_dict)
    elif model_dict['meta'] == 'svr':
        return reg.deserialize_svr(model_dict)
    elif model_dict['meta'] == 'decision-tree-regression':
        return reg.deserialize_decision_tree_regressor(model_dict)
    elif model_dict['meta'] == 'gb-regression':
        return reg.deserialize_gradient_boosting_regressor(model_dict)
    elif model_dict['meta'] == 'rf-regression':
        return reg.deserialize_random_forest_regressor(model_dict)
    elif model_dict['meta'] == 'mlp-regression':
        return reg.deserialize_mlp_regressor(model_dict)
    elif model_dict['meta'] == 'xgboost-ranker':
        return reg.deserialize_xgboost_ranker(model_dict)
    elif model_dict['meta'] == 'xgboost-regressor':
        return reg.deserialize_xgboost_regressor(model_dict)
    elif model_dict['meta'] == 'xgboost-rf-regressor':
        return reg.deserialize_xgboost_rf_regressor(model_dict)
    elif model_dict['meta'] == 'lightgbm-regressor':
        return reg.deserialize_lightgbm_regressor(model_dict)
    elif model_dict['meta'] == 'lightgbm-ranker':
        return reg.deserialize_lightgbm_ranker(model_dict)
    elif model_dict['meta'] == 'catboost-regressor':
        return reg.deserialize_catboost_regressor(model_dict)
    elif model_dict['meta'] == 'catboost-ranker':
        return reg.deserialize_catboost_ranker(model_dict)
    elif model_dict['meta'] == 'adaboost-regressor':
        return reg.deserialize_adaboost_regressor(model_dict)
    elif model_dict['meta'] == 'bagging-regression':
        return reg.deserialize_bagging_regressor(model_dict)
    elif model_dict['meta'] == 'extra-tree-reg':
        return reg.deserialize_extra_tree_regressor(model_dict)
    elif model_dict['meta'] == 'extratrees-regressor':
        return reg.deserialize_extratrees_regressor(model_dict)

    # Clustering
    elif model_dict['meta'] == 'affinity-propagation':
        return clus.deserialize_affinity_propagation(model_dict)
    elif model_dict['meta'] == 'agglomerative-clustering':
        return clus.deserialize_agglomerative_clustering(model_dict)
    elif model_dict['meta'] == 'feature-agglomeration':
        return clus.deserialize_feature_agglomeration(model_dict)
    elif model_dict['meta'] == 'dbscan':
        return clus.deserialize_dbscan(model_dict)
    elif model_dict['meta'] == 'meanshift':
        return clus.deserialize_meanshift(model_dict)
    elif model_dict['meta'] == 'kmeans':
        return clus.deserialize_kmeans(model_dict)
    elif model_dict['meta'] == 'minibatch-kmeans':
        return clus.deserialize_minibatch_kmeans(model_dict)
    elif model_dict['meta'] == 'optics':
        return clus.deserialize_optics(model_dict)
    elif model_dict['meta'] == 'spectral-clustering':
        return clus.deserialize_spectral_clustering(model_dict)
    elif model_dict['meta'] == 'spectral-biclustering':
        return clus.deserialize_spectral_biclustering(model_dict)
    elif model_dict['meta'] == 'spectral-coclustering':
        return clus.deserialize_spectral_coclustering(model_dict)
    elif model_dict['meta'] == 'kmodes':
        return clus.deserialize_kmodes(model_dict)
    elif model_dict['meta'] == 'kprototypes':
        return clus.deserialize_kprototypes(model_dict)
    elif model_dict['meta'] == 'birch':
        return clus.deserialize_birch(model_dict)
    elif model_dict['meta'] == 'bisecting-kmeans':
        return clus.deserialize_bisecting_kmeans(model_dict)
    elif model_dict['meta'] == 'hdbscan':
        return clus.deserialize_hdbscan(model_dict)

    # Cross-decomposition
    elif model_dict['meta'] == 'cca':
        return crdec.deserialize_cca(model_dict)
    elif model_dict['meta'] == 'pls-canonical':
        return crdec.deserialize_pls_canonical(model_dict)
    elif model_dict['meta'] == 'pls-regression':
        return crdec.deserialize_pls_regression(model_dict)
    elif model_dict['meta'] == 'pls-svd':
        return crdec.deserialize_pls_svd(model_dict)

    # Decomposition
    elif model_dict['meta'] == 'pca':
        return dec.deserialize_pca(model_dict)
    elif model_dict['meta'] == 'kernel-pca':
        return  dec.deserialize_kernel_pca(model_dict)
    elif model_dict['meta'] == 'incremental-pca':
        return  dec.deserialize_incremental_pca(model_dict)
    elif model_dict['meta'] == 'sparse-pca':
        return  dec.deserialize_sparse_pca(model_dict)
    elif model_dict['meta'] == 'minibatch-sparse-pca':
        return  dec.deserialize_minibatch_sparse_pca(model_dict)
    elif model_dict['meta'] == 'dictionary-learning':
        return  dec.deserialize_dictionary_learning(model_dict)
    elif model_dict['meta'] == 'minibatch-dictionary-learning':
        return  dec.deserialize_minibatch_dictionary_learning(model_dict)
    elif model_dict['meta'] == 'factor-analysis':
        return  dec.deserialize_factor_analysis(model_dict)
    elif model_dict['meta'] == 'fast-ica':
        return  dec.deserialize_fast_ica(model_dict)
    elif model_dict['meta'] == 'latent-dirichlet-allocation':
        return  dec.deserialize_latent_dirichlet_allocation(model_dict)
    elif model_dict['meta'] == 'nmf':
        return  dec.deserialize_nmf(model_dict)
    elif model_dict['meta'] == 'minibatch-nmf':
        return  dec.deserialize_minibatch_nmf(model_dict)
    elif model_dict['meta'] == 'sparse-coder':
        return  dec.deserialize_sparse_coder(model_dict)
    elif model_dict['meta'] == 'truncated-svd':
        return  dec.deserialize_truncated_svd(model_dict)

    # Manifold
    elif model_dict['meta'] == 'tsne':
        return  man.deserialize_tsne(model_dict)
    elif model_dict['meta'] == 'mds':
        return  man.deserialize_mds(model_dict)
    elif model_dict['meta'] == 'isomap':
        return  man.deserialize_isomap(model_dict)
    elif model_dict['meta'] == 'locally-linear-embedding':
        return  man.deserialize_locally_linear_embedding(model_dict)
    elif model_dict['meta'] == 'spectral-embedding':
        return  man.deserialize_spectral_embedding(model_dict)
    elif model_dict['meta'] == 'umap':
        return  man.deserialize_umap(model_dict)

    # Neighbors
    elif model_dict['meta'] == 'nearest-neighbors':
        return  nei.deserialize_nearest_neighbors(model_dict)
    elif model_dict['meta'] == 'kdtree':
        return  nei.deserialize_kdtree(model_dict)
    elif model_dict['meta'] == 'nn-descent':
        return  nei.deserialize_nndescent(model_dict)

    # Feature Extraction
    elif model_dict['meta'] == 'dict-vectorizer':
        return ext.deserialize_dict_vectorizer(model_dict)

    # Preprocess
    elif model_dict['meta'] == 'label-encoder':
        return pre.deserialize_label_encoder(model_dict)
    elif model_dict['meta'] == 'label-binarizer':
        return pre.deserialize_label_binarizer(model_dict)
    elif model_dict['meta'] == 'multilabel-binarizer':
        return pre.deserialize_multilabel_binarizer(model_dict)
    elif model_dict['meta'] == 'minmax-scaler':
        return pre.deserialize_minmax_scaler(model_dict)
    elif model_dict['meta'] == 'standard-scaler':
        return pre.deserialize_standard_scaler(model_dict)
    elif model_dict['meta'] == 'kernel-centerer':
        return pre.deserialize_kernel_centerer(model_dict)
    elif model_dict['meta'] == 'onehot-encoder':
        return pre.deserialize_onehot_encoder(model_dict)

    # Otherwise
    else:
        raise ModellNotSupported('Model type not supported or corrupt JSON file.')


def to_dict(model, catboost_data: Pool = None):
    return serialize_model(model, catboost_data)


def from_dict(model_dict):
    return deserialize_model(model_dict)


def to_json(model, model_name, catboost_data: Pool = None):
    with open(model_name, 'w') as model_json:
        json.dump(serialize_model(model, catboost_data), model_json)


def from_json(model_name):
    with open(model_name, 'r') as model_json:
        model_dict = json.load(model_json)
        return deserialize_model(model_dict)

class ModellNotSupported(Exception):
    pass
