# -*- coding: utf-8 -*-

import numpy as np
from sklearn.decomposition import PCA


def serialize_pca(model):
    serialized_model = {
        'meta': 'pca',
        'components_': model.components_.tolist(),
        'explained_variance_': model.explained_variance_.tolist(),
        'explained_variance_ratio_': model.explained_variance_ratio_.tolist(),
        'singular_values_': model.singular_values_.tolist(),
        'mean_': model.mean_.tolist(),
        'n_components_': model.n_components_,
        'n_features_': model.n_features_,
        'n_samples_': model.n_samples_,
        'noise_variance_': model.noise_variance_,
        'n_features_in_': model.n_features_in_,
        '_fit_svd_solver': model._fit_svd_solver,
        'params': model.get_params(),
    }

    if 'feature_names_in' in model.__dict__:
        serialized_model['feature_names_in'] = model.feature_names_in.tolist()

    return serialized_model


def deserialize_pca(model_dict):
    model = PCA(**model_dict['params'])

    model.components_ = np.array(model_dict['components_'])
    model.explained_variance_ = np.array(model_dict['explained_variance_'])
    model.explained_variance_ratio_ = np.array(model_dict['explained_variance_ratio_'])
    model.singular_values_ = np.array(model_dict['singular_values_'])
    model.mean_ = np.array(model_dict['mean_'])
    model.n_components_ = model_dict['n_components_']
    model.n_features_ = model_dict['n_features_']
    model.n_samples_ = model_dict['n_samples_']
    model.n_features_in_ = model_dict['n_features_in_']
    model.noise_variance_ = model_dict['noise_variance_']
    model._fit_svd_solver = model_dict['_fit_svd_solver']

    if 'feature_names_in' in model_dict.keys():
        model.feature_names_in = np.array(model_dict['feature_names_in'])

    return model
