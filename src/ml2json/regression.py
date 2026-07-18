# -*- coding: utf-8 -*-

import os
import uuid

import numpy as np
import scipy as sp
import sklearn
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from . import _base

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
    model._effective_probability = model.probability is True


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


def serialize_decision_tree_regressor(model):
    return _base.serialize_model_generic(model)


def deserialize_decision_tree_regressor(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_gradient_boosting_regressor(model):
    return _base.serialize_model_generic(model)


def deserialize_gradient_boosting_regressor(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_random_forest_regressor(model):
    return _base.serialize_model_generic(model)


def deserialize_random_forest_regressor(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_mlp_regressor(model):
    serialized_model = {
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
    return _base.serialize_model_generic(model)


def deserialize_adaboost_regressor(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_bagging_regressor(model):
    return _base.serialize_model_generic(model)


def deserialize_bagging_regressor(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_extra_tree_regressor(model):
    return _base.serialize_model_generic(model)


def deserialize_extra_tree_regressor(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_extratrees_regressor(model):
    return _base.serialize_model_generic(model)


def deserialize_extratrees_regressor(model_dict):
    return _base.deserialize_model_generic(model_dict)

def serialize_nearest_neighbour_regressor(model):
    return _base.serialize_model_generic(model)


def deserialize_nearest_neighbour_regressor(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_stacking_regressor(model):
    # Import here to avoid circular imports
    from . import serialize_model

    serialized_model = {
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


# The classes below all hold plain attributes (arrays, scalars, nested
# already-supported estimators) with no exotic Cython/compiled state, so the
# generic recursive engine handles them directly - same pattern as
# AdaBoostRegressor/BaggingRegressor above.

def serialize_ard_regression(model):
    return _base.serialize_model_generic(model)


def deserialize_ard_regression(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_bayesian_ridge(model):
    return _base.serialize_model_generic(model)


def deserialize_bayesian_ridge(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_elasticnet_cv(model):
    return _base.serialize_model_generic(model)


def deserialize_elasticnet_cv(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_lasso_cv(model):
    return _base.serialize_model_generic(model)


def deserialize_lasso_cv(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_multitask_elasticnet(model):
    return _base.serialize_model_generic(model)


def deserialize_multitask_elasticnet(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_multitask_elasticnet_cv(model):
    return _base.serialize_model_generic(model)


def deserialize_multitask_elasticnet_cv(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_multitask_lasso(model):
    return _base.serialize_model_generic(model)


def deserialize_multitask_lasso(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_multitask_lasso_cv(model):
    return _base.serialize_model_generic(model)


def deserialize_multitask_lasso_cv(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_gamma_regressor(model):
    return _base.serialize_model_generic(model)


def deserialize_gamma_regressor(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_poisson_regressor(model):
    return _base.serialize_model_generic(model)


def deserialize_poisson_regressor(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_tweedie_regressor(model):
    return _base.serialize_model_generic(model)


def deserialize_tweedie_regressor(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_huber_regressor(model):
    return _base.serialize_model_generic(model)


def deserialize_huber_regressor(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_lars(model):
    return _base.serialize_model_generic(model)


def deserialize_lars(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_lars_cv(model):
    return _base.serialize_model_generic(model)


def deserialize_lars_cv(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_lasso_lars(model):
    return _base.serialize_model_generic(model)


def deserialize_lasso_lars(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_lasso_lars_cv(model):
    return _base.serialize_model_generic(model)


def deserialize_lasso_lars_cv(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_lasso_lars_ic(model):
    return _base.serialize_model_generic(model)


def deserialize_lasso_lars_ic(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_orthogonal_matching_pursuit(model):
    return _base.serialize_model_generic(model)


def deserialize_orthogonal_matching_pursuit(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_orthogonal_matching_pursuit_cv(model):
    return _base.serialize_model_generic(model)


def deserialize_orthogonal_matching_pursuit_cv(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_passive_aggressive_regressor(model):
    return _base.serialize_model_generic(model)


def deserialize_passive_aggressive_regressor(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_quantile_regressor(model):
    return _base.serialize_model_generic(model)


def deserialize_quantile_regressor(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_ransac_regressor(model):
    return _base.serialize_model_generic(model)


def deserialize_ransac_regressor(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_ridge_cv(model):
    return _base.serialize_model_generic(model)


def deserialize_ridge_cv(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_sgd_regressor(model):
    return _base.serialize_model_generic(model)


def deserialize_sgd_regressor(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_theilsen_regressor(model):
    return _base.serialize_model_generic(model)


def deserialize_theilsen_regressor(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_linear_svr(model):
    return _base.serialize_model_generic(model)


def deserialize_linear_svr(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_nu_svr(model):
    return _base.serialize_model_generic(model)


def deserialize_nu_svr(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_radius_neighbors_regressor(model):
    return _base.serialize_model_generic(model)


def deserialize_radius_neighbors_regressor(model_dict):
    return _base.deserialize_model_generic(model_dict)
