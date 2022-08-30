# -*- coding: utf-8 -*-

import json

from sklearn import svm, discriminant_analysis, dummy
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor, _gb_losses
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB, ComplementNB
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer, MinMaxScaler, StandardScaler
from sklearn.svm import SVR
from sklearn.cluster import (AffinityPropagation, AgglomerativeClustering,
                             Birch, DBSCAN, FeatureAgglomeration, KMeans,
                             MiniBatchKMeans, MeanShift, OPTICS, SpectralClustering,
                             SpectralBiclustering, SpectralCoclustering)
from sklearn.decomposition import PCA

from . import classification as clf
from . import regression as reg
from . import feature_extraction as ext
from . import preprocessing as pre
from . import cluster as clus
from . import decomposition as dec


__version__ = '0.1.4'


def serialize_model(model):
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
    elif isinstance(model, DecisionTreeRegressor):
        return reg.serialize_decision_tree_regressor(model)
    elif isinstance(model, GradientBoostingRegressor):
        return reg.serialize_gradient_boosting_regressor(model)
    elif isinstance(model, RandomForestRegressor):
        return reg.serialize_random_forest_regressor(model)
    elif isinstance(model, MLPRegressor):
        return reg.serialize_mlp_regressor(model)

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
    elif isinstance(model, KMeans):
        return clus.serialize_kmeans(model)
    elif isinstance(model, MiniBatchKMeans):
        return clus.serialize_minibatch_kmeans(model)
    elif isinstance(model, OPTICS):
        return clus.serialize_optics(model)
    elif isinstance(model, SpectralClustering):
        return clus.serialize_spectral_clustering(model)
    elif isinstance(model, SpectralBiclustering):
        return clus.serialize_spectral_biclustering(model)
    elif isinstance(model, SpectralCoclustering):
        return clus.serialize_spectral_coclustering(model)

    # Decomposition
    elif isinstance(model, PCA):
        return dec.serialize_pca(model)

    # Feature Extraction
    elif isinstance(model, DictVectorizer):
        return ext.serialize_dict_vectorizer(model)

    # Preprocess
    elif isinstance(model, LabelBinarizer):
        return pre.serialize_label_binarizer(model)
    elif isinstance(model, MultiLabelBinarizer):
        return pre.serialize_multilabel_binarizer(model)
    elif isinstance(model, MinMaxScaler):
        return pre.serialize_minmax_scaler(model)
    elif isinstance(model, StandardScaler):
        return pre.serialize_standard_scaler(model)
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

    # Decomposition
    elif model_dict['meta'] == 'pca':
        return dec.deserialize_pca(model_dict)

    # Feature Extraction
    elif model_dict['meta'] == 'dict-vectorizer':
        return ext.deserialize_dict_vectorizer(model_dict)

    # Preprocess
    elif model_dict['meta'] == 'label-binarizer':
        return pre.deserialize_label_binarizer(model_dict)
    elif model_dict['meta'] == 'multilabel-binarizer':
        return pre.deserialize_multilabel_binarizer(model_dict)
    elif model_dict['meta'] == 'minmax-scaler':
        return pre.deserialize_minmax_scaler(model_dict)
    elif model_dict['meta'] == 'standard-scaler':
        return pre.deserialize_standard_scaler(model_dict)
    else:
        raise ModellNotSupported('Model type not supported or corrupt JSON file.')


def to_dict(model):
    return serialize_model(model)


def from_dict(model_dict):
    return deserialize_model(model_dict)


def to_json(model, model_name):
    with open(model_name, 'w') as model_json:
        json.dump(serialize_model(model), model_json)


def from_json(model_name):
    with open(model_name, 'r') as model_json:
        model_dict = json.load(model_json)
        return deserialize_model(model_dict)

class ModellNotSupported(Exception):
    pass
