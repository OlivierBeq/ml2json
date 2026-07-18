# -*- coding: utf-8 -*-

import os
import uuid

import numpy as np
import scipy as sp
from sklearn import svm, discriminant_analysis
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB, ComplementNB
from sklearn.neural_network import MLPClassifier

from . import _base


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
try:
    from imblearn.ensemble import (EasyEnsembleClassifier, RUSBoostClassifier, BalancedBaggingClassifier,
                                   BalancedRandomForestClassifier)
    __optionals__.extend(['imblearn'])
except:
    pass


from .utils import csr
from .preprocessing import (serialize_label_binarizer, deserialize_label_binarizer,
                            serialize_label_encoder, deserialize_label_encoder)


def serialize_logistic_regression(model):
    serialized_model = {
        'classes_': model.classes_.tolist(),
        'coef_': model.coef_.tolist(),
        'intercept_': model.intercept_.tolist(),
        'n_iter_': model.n_iter_.tolist(),
        'params': model.get_params()
    }

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()

    return serialized_model


def deserialize_logistic_regression(model_dict):
    model = LogisticRegression(model_dict['params'])

    model.classes_ = np.array(model_dict['classes_'])
    model.coef_ = np.array(model_dict['coef_'])
    model.intercept_ = np.array(model_dict['intercept_'])
    model.n_iter_ = np.array(model_dict['intercept_'])

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

    return model


def serialize_bernoulli_nb(model):
    serialized_model = {
        'classes_': model.classes_.tolist(),
        'class_count_': model.class_count_.tolist(),
        'class_log_prior_': model.class_log_prior_.tolist(),
        'feature_count_': model.feature_count_.tolist(),
        'feature_log_prob_': model.feature_log_prob_.tolist(),
        'params': model.get_params()
    }

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()

    return serialized_model


def deserialize_bernoulli_nb(model_dict):
    model = BernoulliNB(**model_dict['params'])

    model.classes_ = np.array(model_dict['classes_'])
    model.class_count_ = np.array(model_dict['class_count_'])
    model.class_log_prior_ = np.array(model_dict['class_log_prior_'])
    model.feature_count_= np.array(model_dict['feature_count_'])
    model.feature_log_prob_ = np.array(model_dict['feature_log_prob_'])

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

    return model


def serialize_gaussian_nb(model):
    serialized_model = {
        'classes_': model.classes_.tolist(),
        'class_count_': model.class_count_.tolist(),
        'class_prior_': model.class_prior_.tolist(),
        'theta_': model.theta_.tolist(),
        'var_': model.var_.tolist(),
        'epsilon_': model.epsilon_,
        'params': model.get_params()
    }

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()

    return serialized_model


def deserialize_gaussian_nb(model_dict):
    model = GaussianNB(**model_dict['params'])

    model.classes_ = np.array(model_dict['classes_'])
    model.class_count_ = np.array(model_dict['class_count_'])
    model.class_prior_ = np.array(model_dict['class_prior_'])
    model.theta_ = np.array(model_dict['theta_'])
    model.var_ = np.array(model_dict['var_'])
    model.epsilon_ = model_dict['epsilon_']

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

    return model


def serialize_multinomial_nb(model):
    serialized_model = {
        'classes_': model.classes_.tolist(),
        'class_count_': model.class_count_.tolist(),
        'class_log_prior_': model.class_log_prior_.tolist(),
        'feature_count_': model.feature_count_.tolist(),
        'feature_log_prob_': model.feature_log_prob_.tolist(),
        'params': model.get_params()
    }

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()

    return serialized_model


def deserialize_multinomial_nb(model_dict):
    model = MultinomialNB(**model_dict['params'])

    model.classes_ = np.array(model_dict['classes_'])
    model.class_count_ = np.array(model_dict['class_count_'])
    model.class_log_prior_ = np.array(model_dict['class_log_prior_'])
    model.feature_count_= np.array(model_dict['feature_count_'])
    model.feature_log_prob_ = np.array(model_dict['feature_log_prob_'])

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

    return model


def serialize_complement_nb(model):
    serialized_model = {
        'classes_': model.classes_.tolist(),
        'class_count_': model.class_count_.tolist(),
        'class_log_prior_': model.class_log_prior_.tolist(),
        'feature_count_': model.feature_count_.tolist(),
        'feature_log_prob_': model.feature_log_prob_.tolist(),
        'feature_all_': model.feature_all_.tolist(),
        'params': model.get_params()
    }

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()

    return serialized_model


def deserialize_complement_nb(model_dict):
    model = ComplementNB(**model_dict['params'])

    model.classes_ = np.array(model_dict['classes_'])
    model.class_count_ = np.array(model_dict['class_count_'])
    model.class_log_prior_ = np.array(model_dict['class_log_prior_'])
    model.feature_count_= np.array(model_dict['feature_count_'])
    model.feature_log_prob_ = np.array(model_dict['feature_log_prob_'])
    model.feature_all_ = np.array(model_dict['feature_all_'])

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

    return model


def serialize_lda(model):
    serialized_model = {
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

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()

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

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

    return model


def serialize_qda(model):
    serialized_model = {
        'means_': model.means_.tolist(),
        'priors_': model.priors_.tolist(),
        'scalings_': [array.tolist() for array in model.scalings_],
        'rotations_': [array.tolist() for array in model.rotations_],
        'classes_': model.classes_.tolist(),
        'params': model.get_params()
    }
    if 'covariance_' in model.__dict__:
        serialized_model['covariance_'] = model.covariance_.tolist()

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()

    return serialized_model


def deserialize_qda(model_dict):
    model = discriminant_analysis.QuadraticDiscriminantAnalysis(**model_dict['params'])

    model.means_ = np.array(model_dict['means_']).astype(np.float64)
    model.priors_ = np.array(model_dict['priors_']).astype(np.float64)
    model.scalings_ = np.array(model_dict['scalings_']).astype(np.float64)
    model.rotations_ = np.array(model_dict['rotations_']).astype(np.float64)
    model.classes_ = np.array(model_dict['classes_']).astype(np.int64)

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

    return model


def serialize_svm(model):
    serialized_model = {
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

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()

    return serialized_model


def deserialize_svm(model_dict):
    model = svm.SVC(**model_dict['params'])
    model.shape_fit_ = model_dict['shape_fit_']
    model._gamma = model_dict['_gamma']
    model._effective_probability = model.probability is True

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

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

    return model


def serialize_decision_tree(model):
    return _base.serialize_model_generic(model)


def deserialize_decision_tree(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_gradient_boosting(model):
    return _base.serialize_model_generic(model)


def deserialize_gradient_boosting(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_random_forest(model):
    return _base.serialize_model_generic(model)


def deserialize_random_forest(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_perceptron(model):
    serialized_model = {
        'coef_': model.coef_.tolist(),
        'intercept_': model.intercept_.tolist(),
        'n_iter_': model.n_iter_,
        'classes_': model.classes_.tolist(),
        'params': model.get_params()
    }
    if 'covariance_' in model.__dict__:
        serialized_model['covariance_'] = model.covariance_.tolist()

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()

    return serialized_model


def deserialize_perceptron(model_dict):
    model = Perceptron(**model_dict['params'])

    model.coef_ = np.array(model_dict['coef_']).astype(np.float64)
    model.intercept_ = np.array(model_dict['intercept_']).astype(np.float64)
    model.n_iter_ = np.array(model_dict['n_iter_']).astype(np.float64)
    model.classes_ = np.array(model_dict['classes_']).astype(np.int64)

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

    return model


def serialize_mlp(model):
    serialized_model = {
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

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()

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

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

    return model


def serialize_xgboost_classifier(model):
    serialized_model = {
        'params': model.get_params()
    }

    filename = f'{str(uuid.uuid4())}.json'
    model.save_model(filename)
    with open(filename, 'r') as fh:
        serialized_model['advanced-params'] = fh.read()
    os.remove(filename)

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()

    return serialized_model


if 'XGBClassifier' in __optionals__:
    def deserialize_xgboost_classifier(model_dict):
        model = XGBClassifier(**model_dict['params'])

        filename = f'{str(uuid.uuid4())}.json'
        with open(filename, 'w') as fh:
            fh.write(model_dict['advanced-params'])
        model.load_model(filename)
        os.remove(filename)

        if 'feature_names_in_' in model_dict.keys():
            model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

        return model


if 'XGBRFClassifier' in __optionals__:
    def serialize_xgboost_rf_classifier(model):
        serialized_model = {
            'params': model.get_params()
        }

        filename = f'{str(uuid.uuid4())}.json'
        model.save_model(filename)
        with open(filename, 'r') as fh:
            serialized_model['advanced-params'] = fh.read()
        os.remove(filename)

        if 'feature_names_in_' in model.__dict__:
            serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()

        return serialized_model


    def deserialize_xgboost_rf_classifier(model_dict):
        model = XGBRFClassifier(**model_dict['params'])

        filename = f'{str(uuid.uuid4())}.json'
        with open(filename, 'w') as fh:
            fh.write(model_dict['advanced-params'])
        model.load_model(filename)
        os.remove(filename)

        if 'feature_names_in_' in model_dict.keys():
            model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

        return model


if 'LGBMClassifier' in __optionals__:
    def serialize_lightgbm_classifier(model):
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
            '_n_classes': model._n_classes,
            '_le': serialize_label_encoder(model._le)
        })

        if hasattr(model, '_class_map') and model._class_map is not None:
            serialized_model['params']['_class_map'] = {int(key): int(value) for key, value in model._class_map.items()}
        if hasattr(model, '_classes') and model._classes is not None:
            serialized_model['params']['_classes'] = model._classes.astype(int).tolist()

        if 'feature_names_in_' in model.__dict__:
            serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()

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

        if 'feature_names_in_' in model_dict.keys():
            model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

        return model


if 'CatBoostClassifier' in __optionals__:
    def serialize_catboost_classifier(model, catboost_data):
        serialized_model = {
            'params': model.get_params()
        }

        filename = f'{str(uuid.uuid4())}.json'
        model.save_model(filename, format='json', pool=catboost_data)
        with open(filename, 'r') as fh:
            serialized_model['advanced-params'] = fh.read()
        os.remove(filename)

        if 'feature_names_in_' in model.__dict__:
            serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()

        return serialized_model


    def deserialize_catboost_classifier(model_dict):
        model = CatBoostClassifier(**model_dict['params'])

        filename = f'{str(uuid.uuid4())}.json'
        with open(filename, 'w') as fh:
            fh.write(model_dict['advanced-params'])
        model.load_model(filename, format='json')
        os.remove(filename)

        if 'feature_names_in_' in model_dict.keys():
            model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

        return model


def serialize_adaboost_classifier(model):
    return _base.serialize_model_generic(model)


def deserialize_adaboost_classifier(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_bagging_classifier(model):
    return _base.serialize_model_generic(model)


def deserialize_bagging_classifier(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_extra_tree_classifier(model):
    return _base.serialize_model_generic(model)


def deserialize_extra_tree_classifier(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_extratrees_classifier(model):
    return _base.serialize_model_generic(model)


def deserialize_extratrees_classifier(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_isolation_forest(model):
    return _base.serialize_model_generic(model)


def deserialize_isolation_forest(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_random_trees_embedding(model):
    return _base.serialize_model_generic(model)


def deserialize_random_trees_embedding(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_nearest_neighbour_classifier(model):
    return _base.serialize_model_generic(model)


def deserialize_nearest_neighbour_classifier(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_stacking_classifier(model):
    # Import here to avoid circular imports
    from . import serialize_model

    serialized_model = {
        '_n_feature_outs': model._n_feature_outs,
        'classes_': model.classes_.tolist(),
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
    if '_le' in model.__dict__:
        serialized_model['_le'] = serialize_label_encoder(model._le)
    else:
        serialized_model['_label_encoder'] = serialize_label_encoder(model._label_encoder)

    return serialized_model


def deserialize_stacking_classifier(model_dict):
    # Import here to avoid circular imports
    from . import deserialize_model

    model_dict['params']['estimators'] = [(name, deserialize_model(model)) for name, model in
                                          model_dict['params']['estimators']]

    model = StackingClassifier(**model_dict['params'])

    model._n_feature_outs = model_dict['_n_feature_outs']
    model.classes_ = np.array(model_dict['classes_'])
    model.estimators_ = [deserialize_model(submodel) for submodel in model_dict['estimators_']]
    model.final_estimator_ = deserialize_model(model_dict['final_estimator_'])
    model.stack_method_ = model_dict['stack_method_']
    model.named_estimators_ = {model_name: deserialize_model(submodel) for model_name, submodel in model_dict['named_estimators_'].items()}

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])
    if '_le' in model_dict.keys():
        model._le = deserialize_label_encoder(model_dict['_le'])
    else:
        model._label_encoder = deserialize_label_encoder(model_dict['_label_encoder'])

    return model


def serialize_voting_classifier(model):
    # Import here to avoid circular imports
    from . import serialize_model

    serialized_model = {
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

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

    return model


if 'imblearn' in __optionals__:
    # These wrap the generic recursive engine rather than hand-enumerating
    # attributes (like AdaBoostClassifier/BaggingClassifier above): they hold
    # no exotic Cython/compiled state, just plain arrays and nested estimators
    # already handled by the generic engine's own recursion.
    def serialize_easy_ensemble_classifier(model):
        return _base.serialize_model_generic(model)


    def deserialize_easy_ensemble_classifier(model_dict):
        return _base.deserialize_model_generic(model_dict)


    def serialize_rusboost_classifier(model):
        return _base.serialize_model_generic(model)


    def deserialize_rusboost_classifier(model_dict):
        return _base.deserialize_model_generic(model_dict)


    def serialize_balanced_bagging_classifier(model):
        return _base.serialize_model_generic(model)


    def deserialize_balanced_bagging_classifier(model_dict):
        return _base.deserialize_model_generic(model_dict)


    def serialize_balanced_random_forest_classifier(model):
        return _base.serialize_model_generic(model)


    def deserialize_balanced_random_forest_classifier(model_dict):
        return _base.deserialize_model_generic(model_dict)
