# -*- coding: utf-8 -*-

import os
import uuid
import inspect
import importlib

import numpy as np
import scipy as sp
from sklearn import svm, discriminant_analysis, dummy
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.tree._tree import Tree
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, RandomForestClassifier,
                              StackingClassifier, VotingClassifier, IsolationForest,
                              HistGradientBoostingClassifier, _gb_losses,
                              RandomTreesEmbedding)
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB, ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier


# Allow additional dependencies to be optional
__optionals__ = []
try:
    from xgboost import XGBClassifier, XGBRFClassifier
    __optionals__.extend(['XGBClassifier', 'XGBRFClassifier'])
except:
    pass
try:
    from lightgbm import LGBMClassifier, Booster as LGBMBooster
    __optionals__.append('LGBMClassifier')
except:
    pass
try:
    from catboost import CatBoostClassifier
    __optionals__.append('CatBoostClassifier')
except:
    pass

from . import regression
from .utils import csr
from .neighbors import serialize_kdtree, deserialize_kdtree
from .preprocessing import (serialize_label_binarizer, deserialize_label_binarizer,
                            serialize_label_encoder, deserialize_label_encoder,
                            serialize_onehot_encoder, deserialize_onehot_encoder)


def serialize_logistic_regression(model):
    serialized_model = {
        'meta': 'lr',
        'classes_': model.classes_.tolist(),
        'coef_': model.coef_.tolist(),
        'intercept_': model.intercept_.tolist(),
        'n_iter_': model.n_iter_.tolist(),
        'params': model.get_params()
    }

    if 'feature_names_in' in model.__dict__:
        serialized_model['feature_names_in'] = model.feature_names_in.tolist()

    return serialized_model


def deserialize_logistic_regression(model_dict):
    model = LogisticRegression(model_dict['params'])

    model.classes_ = np.array(model_dict['classes_'])
    model.coef_ = np.array(model_dict['coef_'])
    model.intercept_ = np.array(model_dict['intercept_'])
    model.n_iter_ = np.array(model_dict['intercept_'])

    if 'feature_names_in' in model_dict.keys():
        model.feature_names_in = np.array(model_dict['feature_names_in'])

    return model


def serialize_bernoulli_nb(model):
    serialized_model = {
        'meta': 'bernoulli-nb',
        'classes_': model.classes_.tolist(),
        'class_count_': model.class_count_.tolist(),
        'class_log_prior_': model.class_log_prior_.tolist(),
        'feature_count_': model.feature_count_.tolist(),
        'feature_log_prob_': model.feature_log_prob_.tolist(),
        'params': model.get_params()
    }

    if 'feature_names_in' in model.__dict__:
        serialized_model['feature_names_in'] = model.feature_names_in.tolist()

    return serialized_model


def deserialize_bernoulli_nb(model_dict):
    model = BernoulliNB(**model_dict['params'])

    model.classes_ = np.array(model_dict['classes_'])
    model.class_count_ = np.array(model_dict['class_count_'])
    model.class_log_prior_ = np.array(model_dict['class_log_prior_'])
    model.feature_count_= np.array(model_dict['feature_count_'])
    model.feature_log_prob_ = np.array(model_dict['feature_log_prob_'])

    if 'feature_names_in' in model_dict.keys():
        model.feature_names_in = np.array(model_dict['feature_names_in'])

    return model


def serialize_gaussian_nb(model):
    serialized_model = {
        'meta': 'gaussian-nb',
        'classes_': model.classes_.tolist(),
        'class_count_': model.class_count_.tolist(),
        'class_prior_': model.class_prior_.tolist(),
        'theta_': model.theta_.tolist(),
        'var_': model.var_.tolist(),
        'epsilon_': model.epsilon_,
        'params': model.get_params()
    }

    if 'feature_names_in' in model.__dict__:
        serialized_model['feature_names_in'] = model.feature_names_in.tolist()

    return serialized_model


def deserialize_gaussian_nb(model_dict):
    model = GaussianNB(**model_dict['params'])

    model.classes_ = np.array(model_dict['classes_'])
    model.class_count_ = np.array(model_dict['class_count_'])
    model.class_prior_ = np.array(model_dict['class_prior_'])
    model.theta_ = np.array(model_dict['theta_'])
    model.var_ = np.array(model_dict['var_'])
    model.epsilon_ = model_dict['epsilon_']

    if 'feature_names_in' in model_dict.keys():
        model.feature_names_in = np.array(model_dict['feature_names_in'])

    return model


def serialize_multinomial_nb(model):
    serialized_model = {
        'meta': 'multinomial-nb',
        'classes_': model.classes_.tolist(),
        'class_count_': model.class_count_.tolist(),
        'class_log_prior_': model.class_log_prior_.tolist(),
        'feature_count_': model.feature_count_.tolist(),
        'feature_log_prob_': model.feature_log_prob_.tolist(),
        'params': model.get_params()
    }

    if 'feature_names_in' in model.__dict__:
        serialized_model['feature_names_in'] = model.feature_names_in.tolist()

    return serialized_model


def deserialize_multinomial_nb(model_dict):
    model = MultinomialNB(**model_dict['params'])

    model.classes_ = np.array(model_dict['classes_'])
    model.class_count_ = np.array(model_dict['class_count_'])
    model.class_log_prior_ = np.array(model_dict['class_log_prior_'])
    model.feature_count_= np.array(model_dict['feature_count_'])
    model.feature_log_prob_ = np.array(model_dict['feature_log_prob_'])

    if 'feature_names_in' in model_dict.keys():
        model.feature_names_in = np.array(model_dict['feature_names_in'])

    return model


def serialize_complement_nb(model):
    serialized_model = {
        'meta': 'complement-nb',
        'classes_': model.classes_.tolist(),
        'class_count_': model.class_count_.tolist(),
        'class_log_prior_': model.class_log_prior_.tolist(),
        'feature_count_': model.feature_count_.tolist(),
        'feature_log_prob_': model.feature_log_prob_.tolist(),
        'feature_all_': model.feature_all_.tolist(),
        'params': model.get_params()
    }

    if 'feature_names_in' in model.__dict__:
        serialized_model['feature_names_in'] = model.feature_names_in.tolist()

    return serialized_model


def deserialize_complement_nb(model_dict):
    model = ComplementNB(**model_dict['params'])

    model.classes_ = np.array(model_dict['classes_'])
    model.class_count_ = np.array(model_dict['class_count_'])
    model.class_log_prior_ = np.array(model_dict['class_log_prior_'])
    model.feature_count_= np.array(model_dict['feature_count_'])
    model.feature_log_prob_ = np.array(model_dict['feature_log_prob_'])
    model.feature_all_ = np.array(model_dict['feature_all_'])

    if 'feature_names_in' in model_dict.keys():
        model.feature_names_in = np.array(model_dict['feature_names_in'])

    return model


def serialize_lda(model):
    serialized_model = {
        'meta': 'lda',
        'coef_': model.coef_.tolist(),
        'intercept_': model.intercept_.tolist(),
        'explained_variance_ratio_': model.explained_variance_ratio_.tolist(),
        'means_': model.means_.tolist(),
        'priors_': model.priors_.tolist(),
        'scalings_': model.scalings_.tolist(),
        'xbar_': model.xbar_.tolist(),
        'classes_': model.classes_.tolist(),
        'params': model.get_params()
    }
    if 'covariance_' in model.__dict__:
        serialized_model['covariance_'] = model.covariance_.tolist()

    if 'feature_names_in' in model.__dict__:
        serialized_model['feature_names_in'] = model.feature_names_in.tolist()

    return serialized_model


def deserialize_lda(model_dict):
    model = discriminant_analysis.LinearDiscriminantAnalysis(**model_dict['params'])

    model.coef_ = np.array(model_dict['coef_']).astype(np.float64)
    model.intercept_ = np.array(model_dict['intercept_']).astype(np.float64)
    model.explained_variance_ratio_ = np.array(model_dict['explained_variance_ratio_']).astype(np.float64)
    model.means_ = np.array(model_dict['means_']).astype(np.float64)
    model.priors_ = np.array(model_dict['priors_']).astype(np.float64)
    model.scalings_ = np.array(model_dict['scalings_']).astype(np.float64)
    model.xbar_ = np.array(model_dict['xbar_']).astype(np.float64)
    model.classes_ = np.array(model_dict['classes_']).astype(np.int64)

    if 'feature_names_in' in model_dict.keys():
        model.feature_names_in = np.array(model_dict['feature_names_in'])

    return model


def serialize_qda(model):
    serialized_model = {
        'meta': 'qda',
        'means_': model.means_.tolist(),
        'priors_': model.priors_.tolist(),
        'scalings_': [array.tolist() for array in model.scalings_],
        'rotations_': [array.tolist() for array in model.rotations_],
        'classes_': model.classes_.tolist(),
        'params': model.get_params()
    }
    if 'covariance_' in model.__dict__:
        serialized_model['covariance_'] = model.covariance_.tolist()

    if 'feature_names_in' in model.__dict__:
        serialized_model['feature_names_in'] = model.feature_names_in.tolist()

    return serialized_model


def deserialize_qda(model_dict):
    model = discriminant_analysis.QuadraticDiscriminantAnalysis(**model_dict['params'])

    model.means_ = np.array(model_dict['means_']).astype(np.float64)
    model.priors_ = np.array(model_dict['priors_']).astype(np.float64)
    model.scalings_ = np.array(model_dict['scalings_']).astype(np.float64)
    model.rotations_ = np.array(model_dict['rotations_']).astype(np.float64)
    model.classes_ = np.array(model_dict['classes_']).astype(np.int64)

    if 'feature_names_in' in model_dict.keys():
        model.feature_names_in = np.array(model_dict['feature_names_in'])

    return model


def serialize_svm(model):
    serialized_model = {
        'meta': 'svm',
        'class_weight_': model.class_weight_.tolist(),
        'classes_': model.classes_.tolist(),
        'support_': model.support_.tolist(),
        '_n_support': model._n_support.tolist(),
        'intercept_': model.intercept_.tolist(),
        '_probA': model._probA.tolist(),
        '_probB': model._probB.tolist(),
        '_intercept_': model._intercept_.tolist(),
        'shape_fit_': model.shape_fit_,
        '_gamma': model._gamma,
        'params': model.get_params()
    }

    if isinstance(model.support_vectors_, sp.sparse.csr_matrix):
        serialized_model['support_vectors_'] = csr.serialize_csr_matrix(model.support_vectors_)
    elif isinstance(model.support_vectors_, np.ndarray):
        serialized_model['support_vectors_'] = model.support_vectors_.tolist()

    if isinstance(model.dual_coef_, sp.sparse.csr_matrix):
        serialized_model['dual_coef_'] = csr.serialize_csr_matrix(model.dual_coef_)
    elif isinstance(model.dual_coef_, np.ndarray):
        serialized_model['dual_coef_'] = model.dual_coef_.tolist()

    if isinstance(model._dual_coef_, sp.sparse.csr_matrix):
        serialized_model['_dual_coef_'] = csr.serialize_csr_matrix(model._dual_coef_)
    elif isinstance(model._dual_coef_, np.ndarray):
        serialized_model['_dual_coef_'] = model._dual_coef_.tolist()

    if 'feature_names_in' in model.__dict__:
        serialized_model['feature_names_in'] = model.feature_names_in.tolist()

    return serialized_model


def deserialize_svm(model_dict):
    model = svm.SVC(**model_dict['params'])
    model.shape_fit_ = model_dict['shape_fit_']
    model._gamma = model_dict['_gamma']

    model.class_weight_ = np.array(model_dict['class_weight_']).astype(np.float64)
    model.classes_ = np.array(model_dict['classes_'])
    model.support_ = np.array(model_dict['support_']).astype(np.int32)
    model._n_support = np.array(model_dict['_n_support']).astype(np.int32)
    model.intercept_ = np.array(model_dict['intercept_']).astype(np.float64)
    model._probA = np.array(model_dict['_probA']).astype(np.float64)
    model._probB = np.array(model_dict['_probB']).astype(np.float64)
    model._intercept_ = np.array(model_dict['_intercept_']).astype(np.float64)

    if 'meta' in model_dict['support_vectors_'] and model_dict['support_vectors_']['meta'] == 'csr':
        model.support_vectors_ = csr.deserialize_csr_matrix(model_dict['support_vectors_'])
        model._sparse = True
    else:
        model.support_vectors_ = np.array(model_dict['support_vectors_']).astype(np.float64)
        model._sparse = False

    if 'meta' in model_dict['dual_coef_'] and model_dict['dual_coef_']['meta'] == 'csr':
        model.dual_coef_ = csr.deserialize_csr_matrix(model_dict['dual_coef_'])
    else:
        model.dual_coef_ = np.array(model_dict['dual_coef_']).astype(np.float64)

    if 'meta' in model_dict['_dual_coef_'] and model_dict['_dual_coef_']['meta'] == 'csr':
        model._dual_coef_ = csr.deserialize_csr_matrix(model_dict['_dual_coef_'])
    else:
        model._dual_coef_ = np.array(model_dict['_dual_coef_']).astype(np.float64)

    if 'feature_names_in' in model_dict.keys():
        model.feature_names_in = np.array(model_dict['feature_names_in'])

    return model


def serialize_dummy_classifier(model):
    if isinstance(model.classes_, np.ndarray):
        model.classes_ = model.classes_.tolist()
    else:
        model.classes_ = model.classes_
    if isinstance(model.class_prior_, np.ndarray):
        model.class_prior_ = model.class_prior_.tolist()
    else:
        model.class_prior_ = model.class_prior_
    return model.__dict__


def serialize_tree(tree):
    serialized_tree = tree.__getstate__()

    dtypes = serialized_tree['nodes'].dtype
    serialized_tree['nodes'] = serialized_tree['nodes'].tolist()
    serialized_tree['values'] = serialized_tree['values'].tolist()

    return serialized_tree, dtypes


def deserialize_tree(tree_dict, n_features, n_classes, n_outputs):
    tree_dict['nodes'] = [tuple(lst) for lst in tree_dict['nodes']]

    names = ['left_child', 'right_child', 'feature', 'threshold', 'impurity', 'n_node_samples', 'weighted_n_node_samples']
    tree_dict['nodes'] = np.array(tree_dict['nodes'], dtype=np.dtype({'names': names, 'formats': tree_dict['nodes_dtype']}))
    tree_dict['values'] = np.array(tree_dict['values'])

    tree = Tree(n_features, np.array([n_classes], dtype=np.intp), n_outputs)
    tree.__setstate__(tree_dict)

    return tree


def serialize_decision_tree(model):
    tree, dtypes = serialize_tree(model.tree_)
    serialized_model = {
        'meta': 'decision-tree',
        'feature_importances_': model.feature_importances_.tolist(),
        'max_features_': model.max_features_,
        'n_classes_': int(model.n_classes_),
        'n_features_in_': model.n_features_in_,
        'n_outputs_': model.n_outputs_,
        'tree_': tree,
        'classes_': model.classes_.tolist(),
        'params': model.get_params()
    }

    tree_dtypes = []
    for i in range(0, len(dtypes)):
        tree_dtypes.append(dtypes[i].str)

    serialized_model['tree_']['nodes_dtype'] = tree_dtypes

    if 'feature_names_in' in model.__dict__:
        serialized_model['feature_names_in'] = model.feature_names_in.tolist()

    return serialized_model


def deserialize_decision_tree(model_dict):
    deserialized_model = DecisionTreeClassifier(**model_dict['params'])

    deserialized_model.classes_ = np.array(model_dict['classes_'])
    deserialized_model.max_features_ = model_dict['max_features_']
    deserialized_model.n_classes_ = model_dict['n_classes_']
    deserialized_model.n_features_in_ = model_dict['n_features_in_']
    deserialized_model.n_outputs_ = model_dict['n_outputs_']

    tree = deserialize_tree(model_dict['tree_'], model_dict['n_features_in_'], model_dict['n_classes_'], model_dict['n_outputs_'])
    deserialized_model.tree_ = tree

    return deserialized_model


def serialize_gradient_boosting(model):
    serialized_model = {
        'meta': 'gb',
        'classes_': model.classes_.tolist(),
        'max_features_': model.max_features_,
        'n_classes_': model.n_classes_,
        'n_features_in_': model.n_features_in_,
        'train_score_': model.train_score_.tolist(),
        'params': model.get_params(),
        'estimators_shape': list(model.estimators_.shape),
        'estimators_': []
    }

    if  isinstance(model.init_, dummy.DummyClassifier):
        serialized_model['init_'] = serialize_dummy_classifier(model.init_)
        serialized_model['init_']['meta'] = 'dummy'
    elif isinstance(model.init_, str):
        serialized_model['init_'] = model.init_

    if isinstance(model._loss, _gb_losses.BinomialDeviance):
        serialized_model['_loss'] = 'deviance'
    elif isinstance(model._loss, _gb_losses.ExponentialLoss):
        serialized_model['_loss'] = 'exponential'
    elif isinstance(model._loss, _gb_losses.MultinomialDeviance):
        serialized_model['_loss'] = 'multinomial'

    if 'priors' in model.init_.__dict__:
        serialized_model['priors'] = model.init_.priors.tolist()

    serialized_model['estimators_'] = [regression.serialize_decision_tree_regressor(regression_tree) for regression_tree in model.estimators_.reshape(-1, )]

    if 'feature_names_in' in model.__dict__:
        serialized_model['feature_names_in'] = model.feature_names_in.tolist()

    return serialized_model


def deserialize_gradient_boosting(model_dict):
    model = GradientBoostingClassifier(**model_dict['params'])
    estimators = [regression.deserialize_decision_tree_regressor(tree) for tree in model_dict['estimators_']]
    model.estimators_ = np.array(estimators).reshape(model_dict['estimators_shape'])
    if 'init_' in model_dict and model_dict['init_']['meta'] == 'dummy':
        model.init_ = dummy.DummyClassifier()
        model.init_.__dict__ = model_dict['init_']
        model.init_.__dict__.pop('meta')

    model.classes_ = np.array(model_dict['classes_'])
    model.train_score_ = np.array(model_dict['train_score_'])
    model.max_features_ = model_dict['max_features_']
    model.n_classes_ = model_dict['n_classes_']
    model.n_features_in_ = model_dict['n_features_in_']
    if model_dict['_loss'] == 'deviance':
        model._loss = _gb_losses.BinomialDeviance(model.n_classes_)
    elif model_dict['_loss'] == 'exponential':
        model._loss = _gb_losses.ExponentialLoss(model.n_classes_)
    elif model_dict['_loss'] == 'multinomial':
        model._loss = _gb_losses.MultinomialDeviance(model.n_classes_)

    if 'priors' in model_dict:
        model.init_.priors = np.array(model_dict['priors'])

    if 'feature_names_in' in model_dict.keys():
        model.feature_names_in = np.array(model_dict['feature_names_in'])

    return model


def serialize_random_forest(model):
    serialized_model = {
        'meta': 'rf',
        'max_depth': model.max_depth,
        'min_samples_split': model.min_samples_split,
        'min_samples_leaf': model.min_samples_leaf,
        'min_weight_fraction_leaf': model.min_weight_fraction_leaf,
        'max_features': model.max_features,
        'max_leaf_nodes': model.max_leaf_nodes,
        'min_impurity_decrease': model.min_impurity_decrease,
        'n_features_in_': model.n_features_in_,
        'n_outputs_': model.n_outputs_,
        'classes_': model.classes_.tolist(),
        'estimators_': [serialize_decision_tree(decision_tree) for decision_tree in model.estimators_],
        'params': model.get_params()
    }

    if 'oob_score_' in model.__dict__:
        serialized_model['oob_score_'] = model.oob_score_
    if 'oob_decision_function_' in model.__dict__:
        serialized_model['oob_decision_function_'] = model.oob_decision_function_.tolist()

    if isinstance(model.n_classes_, int):
        serialized_model['n_classes_'] = model.n_classes_
    else:
        serialized_model['n_classes_'] = model.n_classes_.tolist()

    if 'feature_names_in' in model.__dict__:
        serialized_model['feature_names_in'] = model.feature_names_in.tolist()

    return serialized_model


def deserialize_random_forest(model_dict):
    model = RandomForestClassifier(**model_dict['params'])
    estimators = [deserialize_decision_tree(decision_tree) for decision_tree in model_dict['estimators_']]
    model.estimators_ = np.array(estimators)

    model.classes_ = np.array(model_dict['classes_'])
    model.n_features_in_ = model_dict['n_features_in_']
    model.n_outputs_ = model_dict['n_outputs_']
    model.max_depth = model_dict['max_depth']
    model.min_samples_split = model_dict['min_samples_split']
    model.min_samples_leaf = model_dict['min_samples_leaf']
    model.min_weight_fraction_leaf = model_dict['min_weight_fraction_leaf']
    model.max_features = model_dict['max_features']
    model.max_leaf_nodes = model_dict['max_leaf_nodes']
    model.min_impurity_decrease = model_dict['min_impurity_decrease']

    if 'oob_score_' in model_dict:
        model.oob_score_ = model_dict['oob_score_']
    if 'oob_decision_function_' in model_dict:
        model.oob_decision_function_ = model_dict['oob_decision_function_']

    if isinstance(model_dict['n_classes_'], list):
        model.n_classes_ = np.array(model_dict['n_classes_'])
    else:
        model.n_classes_ = model_dict['n_classes_']

    if 'feature_names_in' in model_dict.keys():
        model.feature_names_in = np.array(model_dict['feature_names_in'])

    return model


def serialize_perceptron(model):
    serialized_model = {
        'meta': 'perceptron',
        'coef_': model.coef_.tolist(),
        'intercept_': model.intercept_.tolist(),
        'n_iter_': model.n_iter_,
        'classes_': model.classes_.tolist(),
        'params': model.get_params()
    }
    if 'covariance_' in model.__dict__:
        serialized_model['covariance_'] = model.covariance_.tolist()

    if 'feature_names_in' in model.__dict__:
        serialized_model['feature_names_in'] = model.feature_names_in.tolist()

    return serialized_model


def deserialize_perceptron(model_dict):
    model = Perceptron(**model_dict['params'])

    model.coef_ = np.array(model_dict['coef_']).astype(np.float64)
    model.intercept_ = np.array(model_dict['intercept_']).astype(np.float64)
    model.n_iter_ = np.array(model_dict['n_iter_']).astype(np.float64)
    model.classes_ = np.array(model_dict['classes_']).astype(np.int64)

    if 'feature_names_in' in model_dict.keys():
        model.feature_names_in = np.array(model_dict['feature_names_in'])

    return model


def serialize_mlp(model):
    serialized_model = {
        'meta': 'mlp',
        'coefs_': [array.tolist() for array in model.coefs_],
        'loss_': model.loss_,
        'intercepts_': [array.tolist() for array in model.intercepts_],
        'n_iter_': model.n_iter_,
        'n_layers_': model.n_layers_,
        'n_outputs_': model.n_outputs_,
        'out_activation_': model.out_activation_,
        '_label_binarizer': serialize_label_binarizer(model._label_binarizer),
        'params': model.get_params()
    }

    if isinstance(model.classes_, list):
        serialized_model['classes_'] = [array.tolist() for array in model.classes_]
    else:
        serialized_model['classes_'] = model.classes_.tolist()

    if 'feature_names_in' in model.__dict__:
        serialized_model['feature_names_in'] = model.feature_names_in.tolist()

    return serialized_model


def deserialize_mlp(model_dict):
    model = MLPClassifier(**model_dict['params'])

    model.coefs_ = [np.array(coefs) for coefs in model_dict['coefs_']]
    model.loss_ = model_dict['loss_']
    model.intercepts_ = [np.array(intercepts) for intercepts in model_dict['intercepts_']]
    model.n_iter_ = model_dict['n_iter_']
    model.n_layers_ = model_dict['n_layers_']
    model.n_outputs_ = model_dict['n_outputs_']
    model.out_activation_ = model_dict['out_activation_']
    model._label_binarizer = deserialize_label_binarizer(model_dict['_label_binarizer'])

    model.classes_ = np.array(model_dict['classes_'])

    if 'feature_names_in' in model_dict.keys():
        model.feature_names_in = np.array(model_dict['feature_names_in'])

    return model


def serialize_xgboost_classifier(model):
    serialized_model = {
        'meta': 'xgboost-classifier',
        'params': model.get_params()
    }

    filename = f'{str(uuid.uuid4())}.json'
    model.save_model(filename)
    with open(filename, 'r') as fh:
        serialized_model['advanced-params'] = fh.read()
    os.remove(filename)

    if 'feature_names_in' in model.__dict__:
        serialized_model['feature_names_in'] = model.feature_names_in.tolist()

    return serialized_model


if 'XGBClassifier' in __optionals__:
    def deserialize_xgboost_classifier(model_dict):
        model = XGBClassifier(**model_dict['params'])

        filename = f'{str(uuid.uuid4())}.json'
        with open(filename, 'w') as fh:
            fh.write(model_dict['advanced-params'])
        model.load_model(filename)
        os.remove(filename)

        if 'feature_names_in' in model_dict.keys():
            model.feature_names_in = np.array(model_dict['feature_names_in'])

        return model


if 'XGBRFClassifier' in __optionals__:
    def serialize_xgboost_rf_classifier(model):
        serialized_model = {
            'meta': 'xgboost-rf-classifier',
            'params': model.get_params()
        }

        filename = f'{str(uuid.uuid4())}.json'
        model.save_model(filename)
        with open(filename, 'r') as fh:
            serialized_model['advanced-params'] = fh.read()
        os.remove(filename)

        if 'feature_names_in' in model.__dict__:
            serialized_model['feature_names_in'] = model.feature_names_in.tolist()

        return serialized_model


    def deserialize_xgboost_rf_classifier(model_dict):
        model = XGBRFClassifier(**model_dict['params'])

        filename = f'{str(uuid.uuid4())}.json'
        with open(filename, 'w') as fh:
            fh.write(model_dict['advanced-params'])
        model.load_model(filename)
        os.remove(filename)

        if 'feature_names_in' in model_dict.keys():
            model.feature_names_in = np.array(model_dict['feature_names_in'])

        return model


if 'LGBMClassifier' in __optionals__:
    def serialize_lightgbm_classifier(model):
        serialized_model = {
            'meta': 'lightgbm-classifier',
            'params': model.get_params(),
            '_other_params': model._other_params
        }
        serialized_model['params'].update({
            '_Booster': model.booster_.model_to_string(),
            'fitted_': model.fitted_,
            '_evals_result': model._evals_result,
            '_best_score': model._best_score,
            '_best_iteration': model._best_iteration,
            '_objective': model._objective,
            'class_weight': model.class_weight,
            '_class_weight': model._class_weight,
            '_n_features': model._n_features,
            '_n_features_in': model._n_features_in,
            '_n_classes': model._n_classes,
            '_le': serialize_label_encoder(model._le)
        })

        if hasattr(model, '_class_map') and model._class_map is not None:
            serialized_model['params']['_class_map'] = {int(key): int(value) for key, value in model._class_map.items()}
        if hasattr(model, '_classes') and model._classes is not None:
            serialized_model['params']['_classes'] = model._classes.astype(int).tolist()

        if 'feature_names_in' in model.__dict__:
            serialized_model['feature_names_in'] = model.feature_names_in.tolist()

        return serialized_model


    def deserialize_lightgbm_classifier(model_dict):
        params = model_dict['params']
        params['_Booster'] = LGBMBooster(model_str=params['_Booster'])
        params['_le'] = deserialize_label_encoder(params['_le'])

        if '_class_map' in params and params['_class_map'] is not None:
            params['_class_map'] = {np.int32(key): np.int64(value) for key, value in params['_class_map'].items()}
        if '_classes' in params and params['_classes'] is not None:
            params['_classes'] = np.array(params['_classes'], dtype=np.int32)

        model = LGBMClassifier().set_params(**params)
        model._other_params = model_dict['_other_params']

        if 'feature_names_in' in model_dict.keys():
            model.feature_names_in = np.array(model_dict['feature_names_in'])

        return model


if 'CatBoostClassifier' in __optionals__:
    def serialize_catboost_classifier(model, catboost_data):
        serialized_model = {
            'meta': 'catboost-classifier',
            'params': model.get_params()
        }

        filename = f'{str(uuid.uuid4())}.json'
        model.save_model(filename, format='json', pool=catboost_data)
        with open(filename, 'r') as fh:
            serialized_model['advanced-params'] = fh.read()
        os.remove(filename)

        if 'feature_names_in' in model.__dict__:
            serialized_model['feature_names_in'] = model.feature_names_in.tolist()

        return serialized_model


    def deserialize_catboost_classifier(model_dict):
        model = CatBoostClassifier(**model_dict['params'])

        filename = f'{str(uuid.uuid4())}.json'
        with open(filename, 'w') as fh:
            fh.write(model_dict['advanced-params'])
        model.load_model(filename, format='json')
        os.remove(filename)

        if 'feature_names_in' in model_dict.keys():
            model.feature_names_in = np.array(model_dict['feature_names_in'])

        return model


def serialize_adaboost_classifier(model):
    serialized_model = {
        'meta': 'adaboost-classifier',
        'estimators_': [serialize_decision_tree(decision_tree) for decision_tree in model.estimators_],
        'classes_': model.classes_.tolist(),
        'n_classes_': model.n_classes_,
        'estimator_weights_': model.estimator_weights_.tolist(),
        'estimator_errors_': model.estimator_errors_.tolist(),
        'n_features_in_': model.n_features_in_,
        'params': model.get_params()
    }

    if 'base_estimator_' in model.__dict__ and model.base_estimator_ is not None:
        serialized_model['base_estimator_'] = (inspect.getmodule(model.base_estimator_).__name__,
                                               type(model.base_estimator_).__name__,
                                               model.base_estimator_.get_params())
    else:
        serialized_model['base_estimator_'] = None

    if 'feature_names_in' in model.__dict__:
        serialized_model['feature_names_in'] = model.feature_names_in.tolist()

    return serialized_model


def deserialize_adaboost_classifier(model_dict):

    if model_dict['base_estimator_'] is not None:
        model_dict['params']['base_estimator'] = getattr(importlib.import_module(model_dict['base_estimator_'][0]),
                                                         model_dict['base_estimator_'][1])(
            **model_dict['base_estimator_'][2])
    else:
        model_dict['params']['base_estimator'] = None

    model = AdaBoostClassifier(**model_dict['params'])
    model.base_estimator_ = model_dict['params']['base_estimator']
    model.estimators_ = [deserialize_decision_tree(decision_tree) for decision_tree in model_dict['estimators_']]
    model.classes_ = np.array(model_dict['classes_'])
    model.n_classes_ = model_dict['n_classes_']
    model.estimator_weights_ = np.array(model_dict['estimator_weights_'])
    model.estimator_errors_ = np.array(model_dict['estimator_errors_'])
    model.n_features_in_ = model_dict['n_features_in_']

    if 'feature_names_in' in model_dict.keys():
        model.feature_names_in = np.array(model_dict['feature_names_in'])

    return model


def serialize_bagging_classifier(model):
    serialized_model = {
        'meta': 'bagging-classifier',
        '_max_samples': model._max_samples,
        '_n_samples': model._n_samples,
        '_max_features': model._max_features,
        'n_features_in_': model.n_features_in_,
        'classes_': model.classes_.tolist(),
        '_seeds': model._seeds.tolist(),
        'estimators_': [serialize_decision_tree(decision_tree) for decision_tree in model.estimators_],
        'estimator_params': model.estimator_params,
        'estimators_features_': [array.tolist() for array in model.estimators_features_],
        'params': model.get_params()
    }

    if 'base_estimator_' in model.__dict__ and model.base_estimator_ is not None:
        serialized_model['base_estimator_'] = (inspect.getmodule(model.base_estimator_).__name__,
                                               type(model.base_estimator_).__name__,
                                               model.base_estimator_.get_params())
    else:
        serialized_model['base_estimator_'] = None

    if 'oob_score_' in model.__dict__:
        serialized_model['oob_score_'] = model.oob_score_
    if 'oob_decision_function_' in model.__dict__:
        serialized_model['oob_decision_function_'] = model.oob_decision_function_.tolist()


    if isinstance(model.n_classes_, int):
        serialized_model['n_classes_'] = model.n_classes_
    else:
        serialized_model['n_classes_'] = model.n_classes_.tolist()

    if 'feature_names_in' in model.__dict__:
        serialized_model['feature_names_in'] = model.feature_names_in.tolist()

    return serialized_model


def deserialize_bagging_classifier(model_dict):

    if model_dict['base_estimator_'] is not None:
        model_dict['params']['base_estimator'] = getattr(importlib.import_module(model_dict['base_estimator_'][0]),
                                                         model_dict['base_estimator_'][1])(
            **model_dict['base_estimator_'][2])
    else:
        model_dict['params']['base_estimator'] = None

    model = BaggingClassifier(**model_dict['params'])

    model.base_estimator_ = model_dict['params']['base_estimator']
    model.estimators_ = [deserialize_decision_tree(decision_tree) for decision_tree in model_dict['estimators_']]
    model._max_samples = model_dict['_max_samples']
    model._n_samples = model_dict['_n_samples']
    model._max_features = model_dict['_max_features']
    model.n_features_in_ = model_dict['n_features_in_']
    model.classes_ = np.array(model_dict['classes_'])
    model._seeds = np.array(model_dict['_seeds'])
    model.estimator_params = model_dict['estimator_params']
    model.estimators_features_ = [np.array(array) for array in model_dict['estimators_features_']]

    if 'oob_score_' in model_dict:
        model.oob_score_ = model_dict['oob_score_']
    if 'oob_decision_function_' in model_dict:
        model.oob_decision_function_ = model_dict['oob_decision_function_']

    if isinstance(model_dict['n_classes_'], list):
        model.n_classes_ = np.array(model_dict['n_classes_'])
    else:
        model.n_classes_ = model_dict['n_classes_']

    if 'feature_names_in' in model_dict.keys():
        model.feature_names_in = np.array(model_dict['feature_names_in'])

    return model


def serialize_extra_tree_classifier(model):
    tree, dtypes = serialize_tree(model.tree_)
    serialized_model = {
        'meta': 'extra-tree-cls',
        'max_features_': model.max_features_,
        'n_classes_': int(model.n_classes_),
        'n_features_in_': model.n_features_in_,
        'n_outputs_': model.n_outputs_,
        'tree_': tree,
        'classes_': model.classes_.tolist(),
        'params': model.get_params()
    }

    tree_dtypes = []
    for i in range(0, len(dtypes)):
        tree_dtypes.append(dtypes[i].str)

    serialized_model['tree_']['nodes_dtype'] = tree_dtypes

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()

    return serialized_model


def deserialize_extra_tree_classifier(model_dict):
    deserialized_model = ExtraTreeClassifier(**model_dict['params'])

    deserialized_model.classes_ = np.array(model_dict['classes_'])
    deserialized_model.max_features_ = model_dict['max_features_']
    deserialized_model.n_classes_ = model_dict['n_classes_']
    deserialized_model.n_features_in_ = model_dict['n_features_in_']
    deserialized_model.n_outputs_ = model_dict['n_outputs_']

    tree = deserialize_tree(model_dict['tree_'], model_dict['n_features_in_'], model_dict['n_classes_'], model_dict['n_outputs_'])
    deserialized_model.tree_ = tree

    if 'feature_names_in' in model_dict.keys():
        deserialized_model.feature_names_in = np.array(model_dict['feature_names_in'])

    return deserialized_model


def serialize_extratrees_classifier(model):
    serialized_model = {
        'meta': 'extratrees-classifier',
        'n_features_in_': model.n_features_in_,
        'n_outputs_': model.n_outputs_,
        'classes_': model.classes_.tolist(),
        'estimators_': [serialize_extra_tree_classifier(extra_tree) for extra_tree in model.estimators_],
        'params': model.get_params()
    }

    if 'base_estimator_' in model.__dict__ and model.base_estimator_ is not None:
        serialized_model['base_estimator_'] = (inspect.getmodule(model.base_estimator_).__name__,
                                               type(model.base_estimator_).__name__,
                                               model.base_estimator_.get_params())
    else:
        serialized_model['base_estimator_'] = None

    if 'oob_score_' in model.__dict__:
        serialized_model['oob_score_'] = model.oob_score_
    if 'oob_decision_function_' in model.__dict__:
        serialized_model['oob_decision_function_'] = model.oob_decision_function_.tolist()


    if isinstance(model.n_classes_, int):
        serialized_model['n_classes_'] = model.n_classes_
    else:
        serialized_model['n_classes_'] = model.n_classes_.tolist()

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()

    return serialized_model


def deserialize_extratrees_classifier(model_dict):
    model = ExtraTreesClassifier(**model_dict['params'])

    if model_dict['base_estimator_'] is not None:
        model_dict['base_estimator_'] = getattr(importlib.import_module(model_dict['base_estimator_'][0]),
                                                model_dict['base_estimator_'][1])(
            **model_dict['base_estimator_'][2])
    else:
        model_dict['base_estimator_'] = None

    model.base_estimator_ = model_dict['base_estimator_']
    model.estimators_ = [deserialize_extra_tree_classifier(decision_tree) for decision_tree in model_dict['estimators_']]
    model.n_features_in_ = model_dict['n_features_in_']
    model.n_outputs_ = model_dict['n_outputs_']
    model.classes_ = np.array(model_dict['classes_'])

    if 'oob_score_' in model_dict:
        model.oob_score_ = model_dict['oob_score_']
    if 'oob_decision_function_' in model_dict:
        model.oob_decision_function_ = model_dict['oob_decision_function_']

    if isinstance(model_dict['n_classes_'], list):
        model.n_classes_ = np.array(model_dict['n_classes_'])
    else:
        model.n_classes_ = model_dict['n_classes_']

    if 'feature_names_in' in model_dict.keys():
        model.feature_names_in = np.array(model_dict['feature_names_in'])

    return model


def serialize_isolation_forest(model):
    serialized_model = {
        'meta': 'isolation-forest',
        'n_features_in_': model.n_features_in_,
        '_max_features': model._max_features,
        'max_samples_': model.max_samples_,
        '_max_samples': model._max_samples,
        '_n_samples': model._n_samples,
        'offset_': model.offset_,
        'oob_score': model.oob_score,
        'bootstrap_features': model.bootstrap_features,
        '_seeds': model._seeds.tolist(),
        'estimators_': [regression.serialize_extra_tree_regressor(extra_tree) for extra_tree in model.estimators_],
        'estimators_features_': [array.tolist() for array in model.estimators_features_],
        'estimator_params': list(model.estimator_params),
        'params': model.get_params()
    }

    if 'base_estimator_' in model.__dict__ and model.base_estimator_ is not None:
        serialized_model['base_estimator_'] = (inspect.getmodule(model.base_estimator_).__name__,
                                               type(model.base_estimator_).__name__,
                                               model.base_estimator_.get_params())
    else:
        serialized_model['base_estimator_'] = None

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()

    return serialized_model


def deserialize_isolation_forest(model_dict):
    model = IsolationForest(**model_dict['params'])

    if model_dict['base_estimator_'] is not None:
        model_dict['base_estimator_'] = getattr(importlib.import_module(model_dict['base_estimator_'][0]),
                                                model_dict['base_estimator_'][1])(
            **model_dict['base_estimator_'][2])
    else:
        model_dict['base_estimator_'] = None

    model.base_estimator_ = model_dict['base_estimator_']
    model.estimators_ = [regression.deserialize_extra_tree_regressor(decision_tree) for decision_tree in model_dict['estimators_']]
    model.n_features_in_ = model_dict['n_features_in_']
    model._max_features = model_dict['_max_features']
    model.max_samples_ = model_dict['max_samples_']
    model._max_samples = model_dict['_max_samples']
    model._n_samples = model_dict['_n_samples']
    model.offset_ = model_dict['offset_']
    model.oob_score = model_dict['oob_score']
    model.bootstrap_features = model_dict['bootstrap_features']
    model._seeds = np.array(model_dict['_seeds'])
    model.estimators_features_ = [np.array(array) for array in model_dict['estimators_features_']]
    model.estimator_params = tuple(model_dict['estimator_params'])

    if 'feature_names_in' in model_dict.keys():
        model.feature_names_in = np.array(model_dict['feature_names_in'])

    return model


def serialize_random_trees_embedding(model):
    serialized_model = {
        'meta': 'random-trees-embedding',
        'n_features_in_': model.n_features_in_,
        '_n_features_out': model._n_features_out,
        'max_samples': model.max_samples,
        'n_outputs_': model.n_outputs_,
        'oob_score': model.oob_score,
        'bootstrap': model.bootstrap,
        'class_weight': model.class_weight,
        'one_hot_encoder_': serialize_onehot_encoder(model.one_hot_encoder_),
        'estimators_': [regression.serialize_extra_tree_regressor(extra_tree) for extra_tree in model.estimators_],
        'estimator_params': list(model.estimator_params),
        'params': model.get_params()
    }

    if 'base_estimator_' in model.__dict__ and model.base_estimator_ is not None:
        serialized_model['base_estimator_'] = (inspect.getmodule(model.base_estimator_).__name__,
                                               type(model.base_estimator_).__name__,
                                               model.base_estimator_.get_params())
    else:
        serialized_model['base_estimator_'] = None

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()

    return serialized_model


def deserialize_random_trees_embedding(model_dict):
    model = RandomTreesEmbedding(**model_dict['params'])

    if model_dict['base_estimator_'] is not None:
        model_dict['base_estimator_'] = getattr(importlib.import_module(model_dict['base_estimator_'][0]),
                                                model_dict['base_estimator_'][1])(
            **model_dict['base_estimator_'][2])
    else:
        model_dict['base_estimator_'] = None

    model.base_estimator_ = model_dict['base_estimator_']
    model.estimators_ = [regression.deserialize_extra_tree_regressor(decision_tree) for decision_tree in model_dict['estimators_']]
    model.n_features_in_ = model_dict['n_features_in_']
    model._n_features_out = model_dict['_n_features_out']
    model.max_samples = model_dict['max_samples']
    model.n_outputs_ = model_dict['n_outputs_']
    model.oob_score = model_dict['oob_score']
    model.bootstrap = model_dict['bootstrap']
    model.class_weight = model_dict['class_weight']
    model.one_hot_encoder_ = deserialize_onehot_encoder(model_dict['one_hot_encoder_'])
    model.estimator_params = tuple(model_dict['estimator_params'])

    if 'feature_names_in' in model_dict.keys():
        model.feature_names_in = np.array(model_dict['feature_names_in'])

    return model


def serialize_nearest_neighbour_classifier(model):
    serialized_model = {
        'meta': 'nearest-neighbour-classifier',
        'radius': model.radius,
        'n_features_in_': model.n_features_in_,
        'outputs_2d_': model.outputs_2d_,
        'classes_': model.classes_.tolist(),
        '_y': model._y.tolist(),
        'effective_metric_params_': model.effective_metric_params_,
        'effective_metric_': model.effective_metric_,
        '_fit_method': model._fit_method,
        'n_samples_fit_': model.n_samples_fit_,
        '_fit_X': model._fit_X.tolist(),
        'params': model.get_params()
    }

    if '_tree' in model.__dict__ and model.__dict__['_tree'] is not None:
        serialized_model['_tree'] = serialize_kdtree(model._tree)
    else:
        serialized_model['_tree'] = None

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()

    return serialized_model


def deserialize_nearest_neighbour_classifier(model_dict):
    model = KNeighborsClassifier(**model_dict['params'])

    model.radius = model_dict['radius']
    model.n_features_in_ = model_dict['n_features_in_']
    model.outputs_2d_ = model_dict['outputs_2d_']
    model.classes_ = np.array(model_dict['classes_'])
    model._y = np.array(model_dict['_y'])
    model.effective_metric_params_ = model_dict['effective_metric_params_']
    model.effective_metric_ = model_dict['effective_metric_']
    model._fit_method = model_dict['_fit_method']
    model._fit_X = np.array(model_dict['_fit_X'])
    model.n_samples_fit_ = model_dict['n_samples_fit_']

    if model_dict['_tree'] is not None:
        model._tree = deserialize_kdtree(model_dict['_tree'])
    else:
        model._tree = None

    if 'feature_names_in' in model_dict.keys():
        model.feature_names_in = np.array(model_dict['feature_names_in'])

    return model


def serialize_stacking_classifier(model):
    # Import here to avoid circular imports
    from . import serialize_model

    serialized_model = {
        'meta': 'stacking-classifier',
        '_n_feature_outs': model._n_feature_outs,
        'classes_': model.classes_.tolist(),
        '_le': serialize_label_encoder(model._le),
        'estimators_': [serialize_model(submodel) for submodel in model.estimators_],
        'final_estimator_': serialize_model(model.final_estimator_),
        'stack_method_': model.stack_method_,
        'named_estimators_': {model_name: serialize_model(submodel) for model_name, submodel in model.named_estimators_.items()},
        'params': {key:value for key, value in model.get_params().items() if key.split('__')[0] not in ['final_estimator'] + list(model.named_estimators_.keys())}
    }

    # Serialize the estimators in params
    serialized_model['params']['estimators'] = [(name, serialize_model(model)) for name, model in
                                                serialized_model['params']['estimators']]

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()

    return serialized_model

def deserialize_stacking_classifier(model_dict):
    # Import here to avoid circular imports
    from . import deserialize_model

    model_dict['params']['estimators'] = [(name, deserialize_model(model)) for name, model in
                                          model_dict['params']['estimators']]

    model = StackingClassifier(**model_dict['params'])

    model._n_feature_outs = model_dict['_n_feature_outs']
    model.classes_ = np.array(model_dict['classes_'])
    model._le = deserialize_label_encoder(model_dict['_le'])
    model.estimators_ = [deserialize_model(submodel) for submodel in model_dict['estimators_']]
    model.final_estimator_ = deserialize_model(model_dict['final_estimator_'])
    model.stack_method_ = model_dict['stack_method_']
    model.named_estimators_ = {model_name: deserialize_model(submodel) for model_name, submodel in model_dict['named_estimators_'].items()}

    if 'feature_names_in' in model_dict.keys():
        model.feature_names_in = np.array(model_dict['feature_names_in'])

    return model


def serialize_voting_classifier(model):
    # Import here to avoid circular imports
    from . import serialize_model

    serialized_model = {
        'meta': 'voting-classifier',
        'classes_': model.classes_.tolist(),
        'le_': serialize_label_encoder(model.le_),
        'estimators_': [serialize_model(submodel) for submodel in model.estimators_],
        'named_estimators_': {model_name: serialize_model(submodel) for model_name, submodel in
                              model.named_estimators_.items()},
        'params': {key: value for key, value in model.get_params().items() if
                   key.split('__')[0] not in list(zip(*model.get_params()['estimators']))[0]}
    }

    # Serialize the estimators in params
    serialized_model['params']['estimators'] = [(name, serialize_model(model)) for name, model in
                                                serialized_model['params']['estimators']]

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()

    return serialized_model

def deserialize_voting_classifier(model_dict):
    # Import here to avoid circular imports
    from . import deserialize_model

    model_dict['params']['estimators'] = [(name, deserialize_model(model)) for name, model in
                                          model_dict['params']['estimators']]

    model = VotingClassifier(**model_dict['params'])

    model.classes_ = np.array(model_dict['classes_'])
    model.le_ = deserialize_label_encoder(model_dict['le_'])
    model.estimators_ = [deserialize_model(submodel) for submodel in model_dict['estimators_']]
    model.named_estimators_ = {model_name: deserialize_model(submodel) for model_name, submodel in model_dict['named_estimators_'].items()}

    if 'feature_names_in' in model_dict.keys():
        model.feature_names_in = np.array(model_dict['feature_names_in'])

    return model
