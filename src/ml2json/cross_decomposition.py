# -*- coding: utf-8 -*-

import numpy as np
from sklearn.cross_decomposition import (CCA, PLSCanonical,
                                         PLSRegression, PLSSVD)


def serialize_cca(model):
    serialized_model = {
        'meta': 'cca',
        'x_weights_': model.x_weights_.tolist(),
        'y_weights_': model.y_weights_.tolist(),
        'x_loadings_': model.x_loadings_.tolist(),
        'y_loadings_': model.y_loadings_.tolist(),
        'x_rotations_': model.x_rotations_.tolist(),
        'y_rotations_': model.y_rotations_.tolist(),
        '_coef_': model._coef_.tolist(),
        'intercept_': model.intercept_.tolist(),
        'n_iter_': model.n_iter_,
        'n_features_in_': model.n_features_in_,
        '_x_mean': model._x_mean.tolist(),
        '_y_mean': model._y_mean.tolist(),
        '_x_std': model._x_std.tolist(),
        '_y_std': model._y_std.tolist(),
        '_x_scores': model._x_scores.tolist(),
        '_y_scores': model._y_scores.tolist(),
        '_norm_y_weights': model._norm_y_weights,
        '_n_features_out': model._n_features_out,
        'deflation_mode': model.deflation_mode,
        'mode': model.mode,
        'algorithm': model.algorithm,
        'params': model.get_params()
    }

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()

    return serialized_model


def deserialize_cca(model_dict):
    model = CCA(**model_dict['params'])

    model.x_weights_ = np.array(model_dict['x_weights_'])
    model.y_weights_ = np.array(model_dict['y_weights_'])
    model.x_loadings_ = np.array(model_dict['x_loadings_'])
    model.y_loadings_ = np.array(model_dict['y_loadings_'])
    model.x_rotations_ = np.array(model_dict['x_rotations_'])
    model.y_rotations_ = np.array(model_dict['y_rotations_'])
    model._coef_ = np.array(model_dict['_coef_'])
    model.intercept_ = np.array(model_dict['intercept_'])
    model.n_iter_ = model_dict['n_iter_']
    model.n_features_in_ = model_dict['n_features_in_']

    model._x_mean = np.array(model_dict['_x_mean'])
    model._y_mean = np.array(model_dict['_y_mean'])
    model._x_std = np.array(model_dict['_x_std'])
    model._y_std = np.array(model_dict['_y_std'])
    model._x_scores = np.array(model_dict['_x_scores'])
    model._y_scores = np.array(model_dict['_y_scores'])
    model._norm_y_weights = model_dict['_norm_y_weights']
    model._n_features_out = model_dict['_n_features_out']
    model.deflation_mode = model_dict['deflation_mode']
    model.mode = model_dict['mode']
    model.algorithm = model_dict['algorithm']

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

    return model


def serialize_pls_canonical(model):
    serialized_model = {
        'meta': 'pls-canonical',
        'x_weights_': model.x_weights_.tolist(),
        'y_weights_': model.y_weights_.tolist(),
        'x_loadings_': model.x_loadings_.tolist(),
        'y_loadings_': model.y_loadings_.tolist(),
        'x_rotations_': model.x_rotations_.tolist(),
        'y_rotations_': model.y_rotations_.tolist(),
        '_coef_': model._coef_.tolist(),
        'intercept_': model.intercept_.tolist(),
        'n_iter_': model.n_iter_,
        'n_features_in_': model.n_features_in_,
        '_x_mean': model._x_mean.tolist(),
        '_y_mean': model._y_mean.tolist(),
        '_x_std': model._x_std.tolist(),
        '_y_std': model._y_std.tolist(),
        '_x_scores': model._x_scores.tolist(),
        '_y_scores': model._y_scores.tolist(),
        '_norm_y_weights': model._norm_y_weights,
        '_n_features_out': model._n_features_out,
        'deflation_mode': model.deflation_mode,
        'mode': model.mode,
        'params': model.get_params()
    }

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()

    return serialized_model


def deserialize_pls_canonical(model_dict):
    model = PLSCanonical(**model_dict['params'])

    model.x_weights_ = np.array(model_dict['x_weights_'])
    model.y_weights_ = np.array(model_dict['y_weights_'])
    model.x_loadings_ = np.array(model_dict['x_loadings_'])
    model.y_loadings_ = np.array(model_dict['y_loadings_'])
    model.x_rotations_ = np.array(model_dict['x_rotations_'])
    model.y_rotations_ = np.array(model_dict['y_rotations_'])
    model._coef_ = np.array(model_dict['_coef_'])
    model.intercept_ = np.array(model_dict['intercept_'])
    model.n_iter_ = model_dict['n_iter_']
    model.n_features_in_ = model_dict['n_features_in_']

    model._x_mean = np.array(model_dict['_x_mean'])
    model._y_mean = np.array(model_dict['_y_mean'])
    model._x_std = np.array(model_dict['_x_std'])
    model._y_std = np.array(model_dict['_y_std'])
    model._x_scores = np.array(model_dict['_x_scores'])
    model._y_scores = np.array(model_dict['_y_scores'])
    model._norm_y_weights = model_dict['_norm_y_weights']
    model._n_features_out = model_dict['_n_features_out']
    model.deflation_mode = model_dict['deflation_mode']
    model.mode = model_dict['mode']

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

    return model


def serialize_pls_regression(model):
    serialized_model = {
        'meta': 'pls-regression',
        'x_weights_': model.x_weights_.tolist(),
        'y_weights_': model.y_weights_.tolist(),
        'x_loadings_': model.x_loadings_.tolist(),
        'y_loadings_': model.y_loadings_.tolist(),
        'x_rotations_': model.x_rotations_.tolist(),
        'y_rotations_': model.y_rotations_.tolist(),
        '_coef_': model._coef_.tolist(),
        'intercept_': model.intercept_.tolist(),
        'n_iter_': model.n_iter_,
        'n_features_in_': model.n_features_in_,
        '_x_mean': model._x_mean.tolist(),
        '_y_mean': model._y_mean.tolist(),
        '_x_std': model._x_std.tolist(),
        '_y_std': model._y_std.tolist(),
        'x_scores_': model.x_scores_.tolist(),
        'y_scores_': model.y_scores_.tolist(),
        '_x_scores': model._x_scores.tolist(),
        '_y_scores': model._y_scores.tolist(),
        '_norm_y_weights': model._norm_y_weights,
        '_n_features_out': model._n_features_out,
        'deflation_mode': model.deflation_mode,
        'mode': model.mode,
        'params': model.get_params()
    }

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()

    return serialized_model


def deserialize_pls_regression(model_dict):
    model = PLSRegression(**model_dict['params'])

    model.x_weights_ = np.array(model_dict['x_weights_'])
    model.y_weights_ = np.array(model_dict['y_weights_'])
    model.x_loadings_ = np.array(model_dict['x_loadings_'])
    model.y_loadings_ = np.array(model_dict['y_loadings_'])
    model.x_rotations_ = np.array(model_dict['x_rotations_'])
    model.y_rotations_ = np.array(model_dict['y_rotations_'])
    model._coef_ = np.array(model_dict['_coef_'])
    model.intercept_ = np.array(model_dict['intercept_'])
    model.n_iter_ = model_dict['n_iter_']
    model.n_features_in_ = model_dict['n_features_in_']

    model._x_mean = np.array(model_dict['_x_mean'])
    model._y_mean = np.array(model_dict['_y_mean'])
    model._x_std = np.array(model_dict['_x_std'])
    model._y_std = np.array(model_dict['_y_std'])
    model.x_scores_ = np.array(model_dict['x_scores_'])
    model.y_scores_ = np.array(model_dict['y_scores_'])
    model._x_scores = np.array(model_dict['_x_scores'])
    model._y_scores = np.array(model_dict['_y_scores'])
    model._norm_y_weights = model_dict['_norm_y_weights']
    model._n_features_out = model_dict['_n_features_out']
    model.deflation_mode = model_dict['deflation_mode']
    model.mode = model_dict['mode']

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

    return model


def serialize_pls_svd(model):
    serialized_model = {
        'meta': 'pls-svd',
        'x_weights_': model.x_weights_.tolist(),
        'y_weights_': model.y_weights_.tolist(),
        '_x_mean': model._x_mean.tolist(),
        '_y_mean': model._y_mean.tolist(),
        '_x_std': model._x_std.tolist(),
        '_y_std': model._y_std.tolist(),
        'n_features_in_': model.n_features_in_,
        '_n_features_out': model._n_features_out,
        'params': model.get_params()
    }

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()

    return serialized_model


def deserialize_pls_svd(model_dict):
    model = PLSSVD(**model_dict['params'])

    model.x_weights_ = np.array(model_dict['x_weights_'])
    model.y_weights_ = np.array(model_dict['y_weights_'])
    model._x_mean = np.array(model_dict['_x_mean'])
    model._y_mean = np.array(model_dict['_y_mean'])
    model._x_std = np.array(model_dict['_x_std'])
    model._y_std = np.array(model_dict['_y_std'])
    model.n_features_in_ = model_dict['n_features_in_']
    model._n_features_out = model_dict['_n_features_out']

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

    return model
