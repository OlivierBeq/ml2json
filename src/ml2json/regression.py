# -*- coding: utf-8 -*-

import os
import uuid
import inspect
import importlib

import numpy as np
import scipy as sp
import sklearn
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import (AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor, RandomForestRegressor,
                              StackingRegressor, VotingRegressor,
                              HistGradientBoostingRegressor)
from sklearn._loss import loss
from sklearn.neural_network import MLPRegressor
from sklearn.tree._tree import Tree
from sklearn.svm import SVR
from sklearn import dummy
from sklearn.neighbors import KNeighborsRegressor

from .neighbors import serialize_kdtree, deserialize_kdtree

# Allow additional dependencies to be optional
__optionals__ = []
try:
    from xgboost import XGBRegressor, XGBRFRegressor, XGBRanker
    __optionals__.extend(['XGBRegressor', 'XGBRFRegressor', 'XGBRanker'])
except:
    pass
try:
    from lightgbm import LGBMRegressor, LGBMRanker, Booster as LGBMBooster
    __optionals__.extend(['LGBMRegressor', 'LGBMRanker'])
except:
    pass
try:
    from catboost import CatBoostRegressor, CatBoostRanker
    __optionals__.extend(['CatBoostRegressor', 'CatBoostRanker'])
except:
    pass


from .utils import csr


def serialize_linear_regressor(model):
    serialized_model = {
        'meta': 'linear-regression',
        'coef_': model.coef_.tolist(),
        'intercept_': model.intercept_.tolist(),
        'params': model.get_params()
    }

    return serialized_model


def deserialize_linear_regressor(model_dict):
    model = LinearRegression(**model_dict['params'])

    model.coef_ = np.array(model_dict['coef_'])
    model.intercept_ = np.array(model_dict['intercept_'])

    return model


def serialize_lasso_regressor(model):
    serialized_model = {
        'meta': 'lasso-regression',
        'coef_': model.coef_.tolist(),
        'params': model.get_params()
    }

    if isinstance(model.n_iter_, int):
        serialized_model['n_iter_'] = model.n_iter_
    else:
        serialized_model['n_iter_'] = model.n_iter_.tolist()

    if isinstance(model.n_iter_, float):
        serialized_model['intercept_'] = model.intercept_
    else:
        serialized_model['intercept_'] = model.intercept_.tolist()

    return serialized_model


def deserialize_lasso_regressor(model_dict):
    model = Lasso(model_dict['params'])

    model.coef_ = np.array(model_dict['coef_'])

    if isinstance(model_dict['n_iter_'], list):
        model.n_iter_ = np.array(model_dict['n_iter_'])
    else:
        model.n_iter_ = int(model_dict['n_iter_'])

    if isinstance(model_dict['intercept_'], list):
        model.intercept_ = np.array(model_dict['intercept_'])
    else:
        model.intercept_ = float(model_dict['intercept_'])

    return model


def serialize_elastic_regressor(model):
    serialized_model = {
        'meta': 'elasticnet-regression',
        'coef_': model.coef_.tolist(),
        'alpha': model.alpha,
        'params': model.get_params()
    }
    if isinstance(model.n_iter_, int):
        serialized_model['n_iter_'] = model.n_iter_
    else:
        serialized_model['n_iter_'] = model.n_iter_.tolist()
    if isinstance(model.intercept_, float):
        serialized_model['intercept_'] = model.intercept_
    else:
        serialized_model['intercept_'] = model.intercept_.tolist()

    return serialized_model


def deserialize_elastic_regressor(model_dict):
    model = ElasticNet(model_dict['params'])

    model.coef_ = np.array(model_dict['coef_'])
    model.alpha = np.array(model_dict['alpha'])

    if isinstance(model_dict['n_iter_'], list):
        model.n_iter_ = np.array(model_dict['n_iter_'])
    else:
        model.n_iter_ = int(model_dict['n_iter_'])
    if isinstance(model_dict['intercept_'], list):
        model.intercept_ = np.array(model_dict['intercept_'])
    else:
        model.intercept_ = float(model_dict['intercept_'])
    return model


def serialize_ridge_regressor(model):
    serialized_model = {
        'meta': 'ridge-regression',
        'coef_': model.coef_.tolist(),
        'params': model.get_params()
    }

    if model.n_iter_:
        serialized_model['n_iter_'] = model.n_iter_.tolist()

    if isinstance(model.n_iter_, float):
        serialized_model['intercept_'] = model.intercept_
    else:
        serialized_model['intercept_'] = model.intercept_.tolist()

    return serialized_model


def deserialize_ridge_regressor(model_dict):
    model = Ridge(model_dict['params'])

    model.coef_ = np.array(model_dict['coef_'])

    if 'n_iter_' in model_dict:
        model.n_iter_ = np.array(model_dict['n_iter_'])

    if isinstance(model_dict['intercept_'], list):
        model.intercept_ = np.array(model_dict['intercept_'])
    else:
        model.intercept_ = float(model_dict['intercept_'])

    return model


def serialize_svr(model):
    serialized_model = {
        'meta': 'svr',
        'support_': model.support_.tolist(),
        '_n_support': model._n_support.tolist(),
        '_probA': model._probA.tolist(),
        '_probB': model._probB.tolist(),
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

    if hasattr(model, 'class_weight_') and sklearn.__version__ < '1.2.0':
            serialized_model['class_weight_'] = model.class_weight_.tolist(),

    if hasattr(model, 'intercept_'):
        serialized_model['intercept_'] = model.intercept_.tolist()
    if hasattr(model, '_intercept'):
        serialized_model['_intercept'] = model._intercept.tolist()

    return serialized_model


def deserialize_svr(model_dict):
    model = SVR(**model_dict['params'])
    model.shape_fit_ = model_dict['shape_fit_']
    model._gamma = model_dict['_gamma']


    model.support_ = np.array(model_dict['support_']).astype(np.int32)
    model._n_support = np.array(model_dict['_n_support']).astype(np.int32)
    model._probA = np.array(model_dict['_probA']).astype(np.float64)
    model._probB = np.array(model_dict['_probB']).astype(np.float64)

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

    if 'class_weight_' in model_dict:
        model.class_weight_ = np.array(model_dict['class_weight_']).astype(np.float64)
    if '_intercept' in model_dict:
        model._intercept = np.array(model_dict['_intercept']).astype(np.float64)
        model._intercept_ = model._intercept.copy()
    if 'intercept_' in model_dict:
        model.intercept_ = np.array(model_dict['intercept_']).astype(np.float64)
        model._intercept_ = model.intercept_.copy()

    return model


def serialize_tree(tree):
    serialized_tree = tree.__getstate__()
    dtypes = serialized_tree['nodes'].dtype
    serialized_tree['nodes'] = serialized_tree['nodes'].tolist()
    serialized_tree['values'] = serialized_tree['values'].tolist()

    return serialized_tree, dtypes


def deserialize_tree(tree_dict, n_features, n_outputs):
    tree_dict['nodes'] = [tuple(lst) for lst in tree_dict['nodes']]

    names = ['left_child', 'right_child', 'feature', 'threshold', 'impurity', 'n_node_samples', 'weighted_n_node_samples']
    if sklearn.__version__ >= '1.3':
        names.append('missing_go_to_left')
    tree_dict['nodes'] = np.array(tree_dict['nodes'], dtype=np.dtype({'names': names, 'formats': tree_dict['nodes_dtype']}))
    tree_dict['values'] = np.array(tree_dict['values'])

    # Dummy classes
    dummy_classes = np.array([1] * n_outputs, dtype=np.intp)

    tree = Tree(n_features, dummy_classes, n_outputs)
    tree.__setstate__(tree_dict)

    return tree


def serialize_decision_tree_regressor(model):
    tree, dtypes = serialize_tree(model.tree_)
    serialized_model = {
        'meta': 'decision-tree-regression',
        'feature_importances_': model.feature_importances_.tolist(),
        'max_features_': model.max_features_,
        'n_features_in_': model.n_features_in_,
        'n_outputs_': model.n_outputs_,
        'tree_': tree
    }

    tree_dtypes = []
    for i in range(0, len(dtypes)):
        tree_dtypes.append(dtypes[i].str)

    serialized_model['tree_']['nodes_dtype'] = tree_dtypes

    return serialized_model


def deserialize_decision_tree_regressor(model_dict):
    deserialized_decision_tree = DecisionTreeRegressor()

    deserialized_decision_tree.max_features_ = model_dict['max_features_']
    deserialized_decision_tree.n_features_in_ = model_dict['n_features_in_']
    deserialized_decision_tree.n_outputs_ = model_dict['n_outputs_']

    tree = deserialize_tree(model_dict['tree_'], model_dict['n_features_in_'], model_dict['n_outputs_'])
    deserialized_decision_tree.tree_ = tree

    return deserialized_decision_tree


def serialize_dummy_regressor(model):
    if isinstance(model.constant_, np.ndarray):
        model.constant = model.constant_.tolist()
    else:
        model.constant = model.constant_
    return model.__dict__


def serialize_gradient_boosting_regressor(model):

    serialized_model = {
        'meta': 'gb-regression',
        'max_features_': model.max_features_,
        'n_features_in_': model.n_features_in_,
        'train_score_': model.train_score_.tolist(),
        'params': model.get_params(),
        'estimators_shape': list(model.estimators_.shape),
        'estimators_': []
    }
    if  isinstance(model.init_, dummy.DummyRegressor):
        serialized_model['init_'] = serialize_dummy_regressor(model.init_)
        serialized_model['init_']['meta'] = 'dummy'
    elif isinstance(model.init_, str):
        serialized_model['init_'] = model.init_

    if sklearn.__version__ >= '1.4.0':
        if isinstance(model._loss, loss.HalfSquaredError):
            serialized_model['_loss'] = 'ls'
        elif isinstance(model._loss, loss.AbsoluteError):
            serialized_model['_loss'] = 'lad'
        elif isinstance(model._loss, loss.HuberLoss):
            serialized_model['_loss'] = 'huber'
            serialized_model['quantile'] = model._loss.quantile
        elif isinstance(model._loss, loss.PinballLoss):
            serialized_model['_loss'] = 'quantile'
            serialized_model['quantile'] = model._loss.closs.quantile
    else:
        from sklearn.ensemble import _gb_losses
        if isinstance(model._loss, _gb_losses.LeastSquaresError):
            serialized_model['_loss'] = 'ls'
        elif isinstance(model._loss, _gb_losses.LeastAbsoluteError):
            serialized_model['_loss'] = 'lad'
        elif isinstance(model._loss, _gb_losses.HuberLossFunction):
            serialized_model['_loss'] = 'huber'
        elif isinstance(model._loss, _gb_losses.QuantileLossFunction):
            serialized_model['quantile'] = model._loss.quantile
        elif isinstance(model._loss, loss.PinballLoss):
            serialized_model['_loss'] = 'quantile'

    if 'priors' in model.init_.__dict__:
        serialized_model['priors'] = model.init_.priors.tolist()

    for tree in model.estimators_.reshape((-1,)):
        serialized_model['estimators_'].append(serialize_decision_tree_regressor(tree))

    serialized_model['init_'] = {key: value.tolist() if isinstance(value, np.ndarray) else value
                                 for key, value in serialized_model['init_'].items()}

    return serialized_model


def deserialize_gradient_boosting_regressor(model_dict):
    model = GradientBoostingRegressor(**model_dict['params'])
    trees = [deserialize_decision_tree_regressor(tree) for tree in model_dict['estimators_']]
    model.estimators_ = np.array(trees).reshape(model_dict['estimators_shape'])

    if 'init_' in model_dict:
        model_dict['init_'] = {key: np.array(value) if isinstance(value, list) else value
                               for key, value in model_dict['init_'].items()}
        if model_dict['init_']['meta'] == 'dummy':
            model.init_ = dummy.DummyRegressor()
        model.init_.__dict__ = model_dict['init_']
        model.init_.__dict__.pop('meta')

    model.train_score_ = np.array(model_dict['train_score_'])
    model.max_features_ = model_dict['max_features_']
    model.n_features_in_ = model_dict['n_features_in_']

    if sklearn.__version__ >= '1.4.0':
        if model_dict['_loss'] == 'ls':
            model._loss = loss.HalfSquaredError()
        elif model_dict['_loss'] == 'lad':
            model._loss = loss.AbsoluteError()
        elif model_dict['_loss'] == 'huber':
            model._loss = loss.HuberLoss(quantile=model_dict['quantile'])
        elif model_dict['_loss'] == 'quantile':
            model._loss = loss.PinballLoss(quantile=model_dict['quantile'])
    else:
        from sklearn.ensemble import _gb_losses
        if model_dict['_loss'] == 'ls':
            model._loss = _gb_losses.LeastSquaresError()
        elif model_dict['_loss'] == 'lad':
            model._loss = _gb_losses.LeastAbsoluteError()
        elif model_dict['_loss'] == 'huber':
            model._loss = _gb_losses.HuberLossFunction(1)
        elif model_dict['_loss'] == 'quantile':
            model._loss = _gb_losses.QuantileLossFunction(1)

    if 'priors' in model_dict:
        model.init_.priors = np.array(model_dict['priors'])

    return model


def serialize_random_forest_regressor(model):

    serialized_model = {
        'meta': 'rf-regression',
        'estimators_': [serialize_decision_tree_regressor(decision_tree) for decision_tree in model.estimators_],
        'n_features_in_': model.n_features_in_,
        'n_outputs_': model.n_outputs_,
        'params': model.get_params()
    }

    if 'oob_score_' in model.__dict__:
        serialized_model['oob_score_'] = model.oob_score_
    if 'oob_decision_function_' in model.__dict__:
        serialized_model['oob_prediction_'] = model.oob_prediction_.tolist()
    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist(),

    return serialized_model


def deserialize_random_forest_regressor(model_dict):
    model = RandomForestRegressor(**model_dict['params'])
    estimators = [deserialize_decision_tree_regressor(decision_tree) for decision_tree in model_dict['estimators_']]
    model.estimators_ = np.array(estimators)

    model.n_features_in_ = model_dict['n_features_in_']
    model.n_outputs_ = model_dict['n_outputs_']

    if 'oob_score_' in model_dict:
        model.oob_score_ = model_dict['oob_score_']
    if 'oob_prediction_' in model_dict:
        model.oob_prediction_ =np.array(model_dict['oob_prediction_'])

    return model


def serialize_mlp_regressor(model):
    serialized_model = {
        'meta': 'mlp-regression',
        'coefs_': [array.tolist() for array in model.coefs_],
        'loss_': model.loss_,
        'intercepts_': [array.tolist() for array in model.intercepts_],
        'n_iter_': model.n_iter_,
        'n_layers_': model.n_layers_,
        'n_outputs_': model.n_outputs_,
        'out_activation_': model.out_activation_,
        'params': model.get_params()
    }

    return serialized_model


def deserialize_mlp_regressor(model_dict):
    model = MLPRegressor(**model_dict['params'])

    model.coefs_ = [np.array(array) for array in model_dict['coefs_']]
    model.loss_ = model_dict['loss_']
    model.intercepts_ = [np.array(array) for array in model_dict['intercepts_']]
    model.n_iter_ = model_dict['n_iter_']
    model.n_layers_ = model_dict['n_layers_']
    model.n_outputs_ = model_dict['n_outputs_']
    model.out_activation_ = model_dict['out_activation_']

    return model


if 'XGBRanker' in __optionals__:
    def serialize_xgboost_ranker(model):
        serialized_model = {
            'meta': 'xgboost-ranker',
            'params': model.get_params()
        }

        filename = f'{str(uuid.uuid4())}.json'
        model.save_model(filename)
        with open(filename, 'r') as fh:
            serialized_model['advanced-params'] = fh.read()
        os.remove(filename)

        return serialized_model


    def deserialize_xgboost_ranker(model_dict):
        model = XGBRanker(**model_dict['params'])

        filename = f'{str(uuid.uuid4())}.json'
        with open(filename, 'w') as fh:
            fh.write(model_dict['advanced-params'])
        model.load_model(filename)
        os.remove(filename)

        return model

if 'XGBRegressor' in __optionals__:
    def serialize_xgboost_regressor(model):
        serialized_model = {
            'meta': 'xgboost-regressor',
            'params': model.get_params()
        }

        filename = f'{str(uuid.uuid4())}.json'
        model.save_model(filename)
        with open(filename, 'r') as fh:
            serialized_model['advanced-params'] = fh.read()
        os.remove(filename)

        return serialized_model


    def deserialize_xgboost_regressor(model_dict):
        model = XGBRegressor(**model_dict['params'])

        filename = f'{str(uuid.uuid4())}.json'
        with open(filename, 'w') as fh:
            fh.write(model_dict['advanced-params'])
        model.load_model(filename)
        os.remove(filename)

        return model

if 'XGBRFRegressor' in __optionals__:
    def serialize_xgboost_rf_regressor(model):
        serialized_model = {
            'meta': 'xgboost-rf-regressor',
            'params': model.get_params()
        }

        filename = f'{str(uuid.uuid4())}.json'
        model.save_model(filename)
        with open(filename, 'r') as fh:
            serialized_model['advanced-params'] = fh.read()
        os.remove(filename)

        return serialized_model


    def deserialize_xgboost_rf_regressor(model_dict):
        model = XGBRFRegressor(**model_dict['params'])

        filename = f'{str(uuid.uuid4())}.json'
        with open(filename, 'w') as fh:
            fh.write(model_dict['advanced-params'])
        model.load_model(filename)
        os.remove(filename)

        return model


if 'LGBMRegressor' in __optionals__:
    def serialize_lightgbm_regressor(model):
        serialized_model = {
            'meta': 'lightgbm-regressor',
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
            '_n_classes': model._n_classes
        })

        if hasattr(model, '_class_map') and model._class_map is not None:
            serialized_model['params']['_class_map'] = {int(key): int(value) for key, value in model._class_map.items()}
        if hasattr(model, '_classes') and model._classes is not None:
            serialized_model['params']['_classes'] = model._classes.astype(int).tolist()

        return serialized_model


    def deserialize_lightgbm_regressor(model_dict):
        params = model_dict['params']
        params['_Booster'] = LGBMBooster(model_str=params['_Booster'])

        if '_class_map' in params and params['_class_map'] is not None:
            params['_class_map'] = {np.int32(key): np.int64(value) for key, value in model_dict['_class_map'].items()}
        if '_classes' in params and params['_classes'] is not None:
            params['_classes'] = np.array(model_dict['_classes'], dtype=np.int32)

        model = LGBMRegressor().set_params(**params)
        model._other_params = model_dict['_other_params']

        return model

if 'LGBMRanker' in __optionals__:
    def serialize_lightgbm_ranker(model):
        serialized_model = {
            'meta': 'lightgbm-ranker',
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
            '_n_classes': model._n_classes
        })

        if hasattr(model, '_class_map') and model._class_map is not None:
            serialized_model['params']['_class_map'] = {int(key): int(value) for key, value in model._class_map.items()}
        if hasattr(model, '_classes') and model._classes is not None:
            serialized_model['params']['_classes'] = model._classes.astype(int).tolist()

        return serialized_model


    def deserialize_lightgbm_ranker(model_dict):
        params = model_dict['params']
        params['_Booster'] = LGBMBooster(model_str=params['_Booster'])

        if '_class_map' in params:
            params['_class_map'] = {np.int32(key): np.int64(value) for key, value in model_dict['_class_map'].items()}
        if '_classes' in params:
            params['_classes'] = np.array(model_dict['_classes'], dtype=np.int32)


        model = LGBMRanker().set_params(**params)
        model._other_params = model_dict['_other_params']

        return model


if 'CatBoostRegressor' in __optionals__:
    def serialize_catboost_regressor(model, catboost_data):
        serialized_model = {
            'meta': 'catboost-regressor',
            'params': model.get_params()
        }

        filename = f'{str(uuid.uuid4())}.json'
        model.save_model(filename, format='json', pool=catboost_data)
        with open(filename, 'r') as fh:
            serialized_model['advanced-params'] = fh.read()
        os.remove(filename)

        return serialized_model


    def deserialize_catboost_regressor(model_dict):
        model = CatBoostRegressor(**model_dict['params'])

        filename = f'{str(uuid.uuid4())}.json'
        with open(filename, 'w') as fh:
            fh.write(model_dict['advanced-params'])
        model.load_model(filename, format='json')
        os.remove(filename)

        return model


if 'CatBoostRanker' in __optionals__:
    def serialize_catboost_ranker(model: CatBoostRanker, catboost_data):
        serialized_model = {
            'meta': 'catboost-ranker',
            'params': model.get_params()
        }

        filename = f'{str(uuid.uuid4())}.json'
        model.save_model(filename, format='json', pool=catboost_data)
        with open(filename, 'r') as fh:
            serialized_model['advanced-params'] = fh.read()
        os.remove(filename)

        return serialized_model


    def deserialize_catboost_ranker(model_dict):
        model = CatBoostRanker(**model_dict['params'])

        filename = f'{str(uuid.uuid4())}.json'
        with open(filename, 'w') as fh:
            fh.write(model_dict['advanced-params'])
        model.load_model(filename, format='json')
        os.remove(filename)

        return model


def serialize_adaboost_regressor(model):
    serialized_model = {
        'meta': 'adaboost-regressor',
        'estimators_': [serialize_decision_tree_regressor(decision_tree) for decision_tree in model.estimators_],
        'estimator_weights_': model.estimator_weights_.tolist(),
        'estimator_errors_': model.estimator_errors_.tolist(),
        'estimator_params': model.estimator_params,
        'n_features_in_': model.n_features_in_,
        'params': model.get_params()
    }

    if 'base_estimator_' in model.__dict__ and model.base_estimator_ is not None:
        serialized_model['base_estimator_'] = (inspect.getmodule(model.base_estimator_).__name__,
                                               type(model.base_estimator_).__name__,
                                               model.base_estimator_.get_params())
    elif sklearn.__version__ < '1.2.0':
        serialized_model['base_estimator_'] = None
    elif '_estimator' in model.__dict__ and model._estimator is not None:
        serialized_model['_estimator'] = (inspect.getmodule(model._estimator).__name__,
                                          type(model._estimator).__name__,
                                          model._estimator.get_params())
    else:
        serialized_model['_estimator'] = None

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()

    return serialized_model


def deserialize_adaboost_regressor(model_dict):
    if model_dict.get('base_estimator_') is not None:
        model_dict['params']['base_estimator'] = getattr(importlib.import_module(model_dict['base_estimator_'][0]),
                                                         model_dict['base_estimator_'][1])(
            **model_dict['base_estimator_'][2])
    elif sklearn.__version__ < '1.2.0':
        model_dict['params']['base_estimator'] = None
    elif model_dict.get('_estimator') is not None:
        model_dict['params']['estimator'] = getattr(importlib.import_module(model_dict['_estimator'][0]),
                                                    model_dict['_estimator'][1])(**model_dict['_estimator'][2])
    else:
        model_dict['params']['estimator'] = None

    model = AdaBoostRegressor(**model_dict['params'])
    if sklearn.__version__ < '1.2.0':
        model.base_estimator_ = model_dict['params']['base_estimator']
    else:
        model._estimator = model_dict['params']['estimator']
    model.estimators_ = [deserialize_decision_tree_regressor(decision_tree) for decision_tree in model_dict['estimators_']]
    model.estimator_weights_ = np.array(model_dict['estimator_weights_'])
    model.estimator_errors_ = np.array(model_dict['estimator_errors_'])
    model.estimator_params = tuple(model_dict['estimator_params'])
    model.n_features_in_ = model_dict['n_features_in_']

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

    return model


def serialize_bagging_regressor(model):
    serialized_model = {
        'meta': 'bagging-regression',
        '_max_samples': model._max_samples,
        '_n_samples': model._n_samples,
        '_max_features': model._max_features,
        'n_features_in_': model.n_features_in_,
        '_seeds': model._seeds.tolist(),
        'estimators_': [serialize_decision_tree_regressor(decision_tree) for decision_tree in model.estimators_],
        'estimator_params': model.estimator_params,
        'estimators_features_': [array.tolist() for array in model.estimators_features_],
        'params': model.get_params()
    }

    if 'base_estimator_' in model.__dict__ and model.base_estimator_ is not None:
        serialized_model['base_estimator_'] = (inspect.getmodule(model.base_estimator_).__name__,
                                               type(model.base_estimator_).__name__,
                                               model.base_estimator_.get_params())
    elif sklearn.__version__ < '1.2.0':
        serialized_model['base_estimator_'] = None
    elif '_estimator' in model.__dict__ and model._estimator is not None:
        serialized_model['_estimator'] = (inspect.getmodule(model._estimator).__name__,
                                          type(model._estimator).__name__,
                                          model._estimator.get_params())
    else:
        serialized_model['_estimator'] = None

    if 'oob_score_' in model.__dict__:
        serialized_model['oob_score_'] = model.oob_score_
    if 'oob_decision_function_' in model.__dict__:
        serialized_model['oob_decision_function_'] = model.oob_decision_function_.tolist()

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()

    return serialized_model


def deserialize_bagging_regressor(model_dict):
    if model_dict.get('base_estimator_') is not None:
        model_dict['params']['base_estimator'] = getattr(importlib.import_module(model_dict['base_estimator_'][0]),
                                                         model_dict['base_estimator_'][1])(
            **model_dict['base_estimator_'][2])
    elif sklearn.__version__ < '1.2.0':
        model_dict['params']['base_estimator'] = None
    elif model_dict.get('_estimator') is not None:
        model_dict['params']['estimator'] = getattr(importlib.import_module(model_dict['_estimator'][0]),
                                                     model_dict['_estimator'][1])(**model_dict['_estimator'][2])
    else:
        model_dict['params']['estimator'] = None

    model = BaggingRegressor(**model_dict['params'])

    if sklearn.__version__ < '1.2.0':
        model.base_estimator_ = model_dict['params']['base_estimator']
    else:
        model._estimator = model_dict['params']['estimator']
    model.estimators_ = [deserialize_decision_tree_regressor(decision_tree) for decision_tree in model_dict['estimators_']]
    model._max_samples = model_dict['_max_samples']
    model._n_samples = model_dict['_n_samples']
    model._max_features = model_dict['_max_features']
    model.n_features_in_ = model_dict['n_features_in_']
    model._seeds = np.array(model_dict['_seeds'])
    model.estimator_params = model_dict['estimator_params']
    model.estimators_features_ = [np.array(array) for array in model_dict['estimators_features_']]

    if 'oob_score_' in model_dict:
        model.oob_score_ = model_dict['oob_score_']
    if 'oob_decision_function_' in model_dict:
        model.oob_decision_function_ = model_dict['oob_decision_function_']

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

    return model


def serialize_extra_tree_regressor(model):
    tree, dtypes = serialize_tree(model.tree_)
    serialized_model = {
        'meta': 'extra-tree-reg',
        'max_features_': model.max_features_,
        'n_features_in_': model.n_features_in_,
        'n_outputs_': model.n_outputs_,
        'tree_': tree,
        'params': model.get_params()
    }

    tree_dtypes = []
    for i in range(0, len(dtypes)):
        tree_dtypes.append(dtypes[i].str)

    serialized_model['tree_']['nodes_dtype'] = tree_dtypes

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()

    return serialized_model


def deserialize_extra_tree_regressor(model_dict):
    deserialized_model = ExtraTreeRegressor(**model_dict['params'])

    deserialized_model.max_features_ = model_dict['max_features_']
    deserialized_model.n_features_in_ = model_dict['n_features_in_']
    deserialized_model.n_outputs_ = model_dict['n_outputs_']

    tree = deserialize_tree(model_dict['tree_'], model_dict['n_features_in_'], model_dict['n_outputs_'])
    deserialized_model.tree_ = tree

    if 'feature_names_in_' in model_dict.keys():
        deserialized_model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

    return deserialized_model


def serialize_extratrees_regressor(model):
    serialized_model = {
        'meta': 'extratrees-regressor',
        'n_features_in_': model.n_features_in_,
        'n_outputs_': model.n_outputs_,
        'estimators_': [serialize_extra_tree_regressor(extra_tree) for extra_tree in model.estimators_],
        'params': model.get_params()
    }

    if 'base_estimator_' in model.__dict__ and model.base_estimator_ is not None:
        serialized_model['base_estimator_'] = (inspect.getmodule(model.base_estimator_).__name__,
                                               type(model.base_estimator_).__name__,
                                               model.base_estimator_.get_params())
    elif sklearn.__version__ < '1.2.0':
        serialized_model['base_estimator_'] = None
    elif '_estimator' in model.__dict__ and model._estimator is not None:
        serialized_model['_estimator'] = (inspect.getmodule(model._estimator).__name__,
                                          type(model._estimator).__name__,
                                          model._estimator.get_params())
    else:
        serialized_model['_estimator'] = None

    if 'oob_score_' in model.__dict__:
        serialized_model['oob_score_'] = model.oob_score_
    if 'oob_prediction_' in model.__dict__:
        serialized_model['oob_prediction_'] = model.oob_prediction_.tolist()

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()

    return serialized_model


def deserialize_extratrees_regressor(model_dict):
    model = ExtraTreesRegressor(**model_dict['params'])

    if model_dict.get('base_estimator_') is not None:
        model_dict['params']['base_estimator'] = getattr(importlib.import_module(model_dict['base_estimator_'][0]),
                                                         model_dict['base_estimator_'][1])(
            **model_dict['base_estimator_'][2])
    elif sklearn.__version__ < '1.2.0':
        model_dict['params']['base_estimator'] = None
    elif model_dict.get('_estimator') is not None:
        model_dict['params']['estimator'] = getattr(importlib.import_module(model_dict['_estimator'][0]),
                                                    model_dict['_estimator'][1])(**model_dict['_estimator'][2])
    else:
        model_dict['params']['estimator'] = None

    if sklearn.__version__ < '1.2.0':
        model.base_estimator_ = model_dict['params']['base_estimator']
    else:
        model._estimator = model_dict['params']['estimator']
    model.estimators_ = [deserialize_extra_tree_regressor(decision_tree) for decision_tree in model_dict['estimators_']]
    model.n_features_in_ = model_dict['n_features_in_']
    model.n_outputs_ = model_dict['n_outputs_']

    if 'oob_score_' in model_dict:
        model.oob_score_ = model_dict['oob_score_']
    if 'oob_prediction_' in model_dict:
        model.oob_prediction_ = model_dict['oob_prediction_']

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

    return model

def serialize_nearest_neighbour_regressor(model):
    serialized_model = {
        'meta': 'nearest-neighbour-regressor',
        'radius': model.radius,
        'n_features_in_': model.n_features_in_,
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


def deserialize_nearest_neighbour_regressor(model_dict):
    model = KNeighborsRegressor(**model_dict['params'])

    model.radius = model_dict['radius']
    model.n_features_in_ = model_dict['n_features_in_']
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

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

    return model


def serialize_stacking_regressor(model):
    # Import here to avoid circular imports
    from . import serialize_model

    serialized_model = {
        'meta': 'stacking-regressor',
        '_n_feature_outs': model._n_feature_outs,
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


def deserialize_stacking_regressor(model_dict):
    # Import here to avoid circular imports
    from . import deserialize_model

    model_dict['params']['estimators'] = [(name, deserialize_model(model)) for name, model in
                                          model_dict['params']['estimators']]

    model = StackingRegressor(**model_dict['params'])

    model._n_feature_outs = model_dict['_n_feature_outs']
    model.estimators_ = [deserialize_model(submodel) for submodel in model_dict['estimators_']]
    model.final_estimator_ = deserialize_model(model_dict['final_estimator_'])
    model.stack_method_ = model_dict['stack_method_']
    model.named_estimators_ = {model_name: deserialize_model(submodel) for model_name, submodel in model_dict['named_estimators_'].items()}

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

    return model


def serialize_voting_regressor(model):
    # Import here to avoid circular imports
    from . import serialize_model

    serialized_model = {
        'meta': 'voting-regressor',
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


def deserialize_voting_regressor(model_dict):
    # Import here to avoid circular imports
    from . import deserialize_model

    model_dict['params']['estimators'] = [(name, deserialize_model(model)) for name, model in
                                          model_dict['params']['estimators']]

    model = VotingRegressor(**model_dict['params'])

    model.estimators_ = [deserialize_model(submodel) for submodel in model_dict['estimators_']]
    model.named_estimators_ = {model_name: deserialize_model(submodel) for model_name, submodel in
                               model_dict['named_estimators_'].items()}

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

    return model
