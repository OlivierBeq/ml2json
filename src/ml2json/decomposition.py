# -*- coding: utf-8 -*-

import numpy as np
from sklearn.decomposition import (PCA, KernelPCA, DictionaryLearning, FactorAnalysis, FastICA, IncrementalPCA,
                                   LatentDirichletAllocation, MiniBatchDictionaryLearning, MiniBatchSparsePCA, NMF,
                                   MiniBatchNMF, SparsePCA, SparseCoder, TruncatedSVD)

from .preprocessing import serialize_kernel_centerer, deserialize_kernel_centerer
from .utils.random_state import serialize_random_state, deserialize_random_state


def serialize_pca(model):
    serialized_model = {
        'meta': 'pca',
        'components_': model.components_.tolist(),
        'explained_variance_': model.explained_variance_.tolist(),
        'explained_variance_ratio_': model.explained_variance_ratio_.tolist(),
        'singular_values_': model.singular_values_.tolist(),
        'mean_': model.mean_.tolist(),
        'n_components_': model.n_components_,
        'n_samples_': model.n_samples_,
        'noise_variance_': model.noise_variance_,
        'n_features_in_': model.n_features_in_,
        '_fit_svd_solver': model._fit_svd_solver,
        'params': model.get_params(),
    }

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()
    if 'n_features_' in model.__dict__:
        serialized_model['n_features_'] = model.n_features_

    return serialized_model


def deserialize_pca(model_dict):
    model = PCA(**model_dict['params'])

    model.components_ = np.array(model_dict['components_'])
    model.explained_variance_ = np.array(model_dict['explained_variance_'])
    model.explained_variance_ratio_ = np.array(model_dict['explained_variance_ratio_'])
    model.singular_values_ = np.array(model_dict['singular_values_'])
    model.mean_ = np.array(model_dict['mean_'])
    model.n_components_ = model_dict['n_components_']
    model.n_samples_ = model_dict['n_samples_']
    model.n_features_in_ = model_dict['n_features_in_']
    model.noise_variance_ = model_dict['noise_variance_']
    model._fit_svd_solver = model_dict['_fit_svd_solver']

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])
    if 'n_features_' in model_dict.keys():
        model.n_features_ = model_dict['n_features_']

    return model


def serialize_kernel_pca(model):
    serialized_model = {
        'meta': 'kernel-pca',
        'eigenvalues_': model.eigenvalues_.tolist(),
        'eigenvectors_': model.eigenvectors_.tolist(),
        'n_features_in_': model.n_features_in_,
        'X_fit_': model.X_fit_.tolist(),
        '_centerer': serialize_kernel_centerer(model._centerer),
        'params': model.get_params(),
    }

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()

    if 'gamma_' in model.__dict__:
        serialized_model['gamma_'] = model.gamma_

    return serialized_model


def deserialize_kernel_pca(model_dict):
    model = KernelPCA(**model_dict['params'])

    model.eigenvalues_ = np.array(model_dict['eigenvalues_'])
    model.eigenvectors_ = np.array(model_dict['eigenvectors_'])
    model.n_features_in_ = model_dict['n_features_in_']
    model.X_fit_ = np.array(model_dict['X_fit_'])
    model._centerer = deserialize_kernel_centerer(model_dict['_centerer'])

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

    if 'gamma_' in model_dict.keys():
        model.gamma_ = model_dict['gamma_']

    return model


def serialize_dictionary_learning(model):
    serialized_model = {
        'meta': 'dictionary-learning',
        'components_': model.components_.tolist(),
        'n_iter_': model.n_iter_,
        'error_': model.error_,
        'n_features_in_': model.n_features_in_,
        'params': model.get_params(),
    }

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()

    return serialized_model


def deserialize_dictionary_learning(model_dict):
    model = DictionaryLearning(**model_dict['params'])

    model.components_ = np.array(model_dict['components_'])
    model.n_iter_ = model_dict['n_iter_']
    model.error_ = model_dict['error_']
    model.n_features_in_ = model_dict['n_features_in_']

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

    return model


def serialize_factor_analysis(model):
    serialized_model = {
        'meta': 'factor-analysis',
        'components_': model.components_.tolist(),
        'loglike_': model.loglike_,
        'noise_variance_': model.noise_variance_.tolist(),
        'mean_': model.mean_.tolist(),
        'n_iter_': model.n_iter_,
        'n_features_in_': model.n_features_in_,
        'params': model.get_params(),
    }

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()

    return serialized_model


def deserialize_factor_analysis(model_dict):
    model = FactorAnalysis(**model_dict['params'])

    model.components_ = np.array(model_dict['components_'])
    model.loglike_ = model_dict['loglike_']
    model.noise_variance_ = np.array(model_dict['noise_variance_'])
    model.mean_ = np.array(model_dict['mean_'])
    model.n_iter_ = model_dict['n_iter_']
    model.n_features_in_ = model_dict['n_features_in_']

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

    return model


def serialize_fast_ica(model):
    serialized_model = {
        'meta': 'fast-ica',
        'components_': model.components_.tolist(),
        'mixing_': model.mixing_.tolist(),
        'whitening_': model.whitening_.tolist(),
        'mean_': model.mean_.tolist(),
        'n_iter_': model.n_iter_,
        'n_features_in_': model.n_features_in_,
        'params': model.get_params(),
    }

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()
        
    if '_whiten' in model.__dict__:
        serialized_model['_whiten'] = model._whiten
    else:
        serialized_model['whiten'] = model.whiten

    return serialized_model


def deserialize_fast_ica(model_dict):
    model = FastICA(**model_dict['params'])

    model.components_ = np.array(model_dict['components_'])
    model.mixing_ = np.array(model_dict['mixing_'])
    model.whitening_ = np.array(model_dict['whitening_'])
    model.mean_ = np.array(model_dict['mean_'])
    model.n_iter_ = model_dict['n_iter_']
    model.n_features_in_ = model_dict['n_features_in_']

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

    if '_whiten' in model_dict.keys():
        model._whiten = model_dict['_whiten']
    else:
        model.whithen = model_dict['whiten']

    return model


def serialize_incremental_pca(model):
    serialized_model = {
        'meta': 'incremental-pca',
        'components_': model.components_.tolist(),
        'explained_variance_': model.explained_variance_.tolist(),
        'explained_variance_ratio_': model.explained_variance_ratio_.tolist(),
        'singular_values_': model.singular_values_.tolist(),
        'mean_': model.mean_.tolist(),
        'var_': model.var_.tolist(),
        'noise_variance_': model.noise_variance_,
        'n_components_': model.n_components_,
        'n_samples_seen_': int(model.n_samples_seen_),
        'batch_size_': model.batch_size_,
        'n_features_in_': model.n_features_in_,
        'params': model.get_params(),
    }

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()

    return serialized_model


def deserialize_incremental_pca(model_dict):
    model = IncrementalPCA(**model_dict['params'])

    model.components_ = np.array(model_dict['components_'])
    model.explained_variance_ = np.array(model_dict['explained_variance_'])
    model.explained_variance_ratio_ = np.array(model_dict['explained_variance_ratio_'])
    model.singular_values_ = np.array(model_dict['singular_values_'])
    model.mean_ = np.array(model_dict['mean_'])
    model.var_ = np.array(model_dict['var_'])
    model.noise_variance_ = model_dict['noise_variance_']
    model.n_components_ = model_dict['n_components_']
    model.n_samples_seen_ = np.int32(model_dict['n_samples_seen_'])
    model.batch_size_ = model_dict['batch_size_']
    model.n_features_in_ = model_dict['n_features_in_']

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

    return model


def serialize_minibatch_sparse_pca(model):
    serialized_model = {
        'meta': 'minibatch-sparse-pca',
        'components_': model.components_.tolist(),
        'mean_': model.mean_.tolist(),
        'n_components_': model.n_components_,
        'n_iter_': model.n_iter_,
        'n_features_in_': model.n_features_in_,
        'params': model.get_params(),
    }

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()

    return serialized_model


def deserialize_minibatch_sparse_pca(model_dict):
    model = MiniBatchSparsePCA(**model_dict['params'])

    model.components_ = np.array(model_dict['components_'])
    model.mean_ = np.array(model_dict['mean_'])
    model.n_components_ = model_dict['n_components_']
    model.n_iter_ = np.int32(model_dict['n_iter_'])
    model.n_features_in_ = model_dict['n_features_in_']

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

    return model


def serialize_sparse_pca(model):
    serialized_model = {
        'meta': 'sparse-pca',
        'components_': model.components_.tolist(),
        'mean_': model.mean_.tolist(),
        'n_components_': model.n_components_,
        'n_iter_': model.n_iter_,
        'n_features_in_': model.n_features_in_,
        'params': model.get_params(),
    }

    if isinstance(model.error_, list):
        serialized_model['error_'] = [float(x) for x in model.error_]
    else:
        serialized_model['error_'] = model.error_.tolist()
    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()

    return serialized_model


def deserialize_sparse_pca(model_dict):
    model = SparsePCA(**model_dict['params'])

    model.components_ = np.array(model_dict['components_'])
    model.error_ = np.array(model_dict['error_'])
    model.mean_ = np.array(model_dict['mean_'])
    model.n_components_ = model_dict['n_components_']
    model.n_iter_ = np.int32(model_dict['n_iter_'])
    model.n_features_in_ = model_dict['n_features_in_']

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

    return model


def serialize_latent_dirichlet_allocation(model):
    serialized_model = {
        'meta': 'latent-dirichlet-allocation',
        'components_': model.components_.tolist(),
        'exp_dirichlet_component_': model.exp_dirichlet_component_.tolist(),
        'bound_': model.bound_.tolist(),
        'n_iter_': model.n_iter_,
        'n_batch_iter_': model.n_batch_iter_,
        'doc_topic_prior_': model.doc_topic_prior_,
        'topic_word_prior_': model.topic_word_prior_,
        'n_features_in_': model.n_features_in_,
        'random_state_': serialize_random_state(model.random_state_),
        'params': model.get_params(),
    }

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()

    return serialized_model


def deserialize_latent_dirichlet_allocation(model_dict):
    model = LatentDirichletAllocation(**model_dict['params'])

    model.components_ = np.array(model_dict['components_'])
    model.exp_dirichlet_component_ = np.array(model_dict['exp_dirichlet_component_'])
    model.bound_ = np.array(model_dict['bound_'])
    model.n_iter_ = model_dict['n_iter_']
    model.n_batch_iter_ = model_dict['n_batch_iter_']
    model.doc_topic_prior_ = model_dict['doc_topic_prior_']
    model.topic_word_prior_ = model_dict['topic_word_prior_']
    model.n_features_in_ = model_dict['n_features_in_']
    model.random_state_ = deserialize_random_state(model_dict['random_state_'])

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

    return model


def serialize_nmf(model):
    serialized_model = {
        'meta': 'nmf',
        'components_': model.components_.tolist(),
        'n_components_': model.n_components_,
        'reconstruction_err_': model.reconstruction_err_,
        'n_iter_': model.n_iter_,
        'n_features_in_': model.n_features_in_,
        'params': model.get_params(),
    }

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()

    return serialized_model


def deserialize_nmf(model_dict):
    model = NMF(**model_dict['params'])

    model.components_ = np.array(model_dict['components_'])
    model.n_components_ = model_dict['n_components_']
    model.reconstruction_err_ = model_dict['reconstruction_err_']
    model.n_iter_ = model_dict['n_iter_']
    model.n_features_in_ = model_dict['n_features_in_']

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

    return model


def serialize_minibatch_nmf(model):
    serialized_model = {
        'meta': 'minibatch-nmf',
        'components_': model.components_.tolist(),
        'n_components_': model.n_components_,
        '_n_components': model._n_components,
        'reconstruction_err_': model.reconstruction_err_,
        'n_iter_': model.n_iter_,
        '_transform_max_iter': model._transform_max_iter,
        '_beta_loss': model._beta_loss,
        '_gamma': model._gamma,
        'n_steps_': model.n_steps_,
        'n_features_in_': model.n_features_in_,
        'params': model.get_params(),
    }

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()
    if '_l1_reg_W' in model.__dict__:
        serialized_model['_l1_reg_W'] = model._l1_reg_W
    if '_l1_reg_H' in model.__dict__:
        serialized_model['_l1_reg_H'] = model._l1_reg_H
    if '_l2_reg_W' in model.__dict__:
        serialized_model['_l2_reg_W'] = model._l2_reg_W
    if '_l2_reg_H' in model.__dict__:
        serialized_model['_l2_reg_H'] = model._l2_reg_H
    if '_batch_size' in model.__dict__:
        serialized_model['_batch_size'] = model._batch_size
    if '_components_denominator' in model.__dict__:
        serialized_model['_components_denominator'] = model._components_denominator.tolist()
    if '_components_numerator' in model.__dict__:
        serialized_model['_components_numerator'] = model._components_numerator.tolist()
    if '_ewa_cost' in model.__dict__:
        serialized_model['_ewa_cost'] = model._ewa_cost
    if '_ewa_cost_min' in model.__dict__:
        serialized_model['_ewa_cost_min'] = model._ewa_cost_min
    if '_no_improvement' in model.__dict__:
        serialized_model['_no_improvement'] = model._no_improvement
    if '_rho' in model.__dict__:
        serialized_model['_rho'] = model._rho


    return serialized_model


def deserialize_minibatch_nmf(model_dict):
    model = MiniBatchNMF(**model_dict['params'])

    model.components_ = np.array(model_dict['components_'])
    model.n_components_ = model_dict['n_components_']
    model._n_components = model_dict['_n_components']
    model.reconstruction_err_ = model_dict['reconstruction_err_']
    model.n_iter_ = model_dict['n_iter_']
    model._transform_max_iter = model_dict['_transform_max_iter']
    model._beta_loss = model_dict['_beta_loss']
    model._gamma = model_dict['_gamma']
    model.n_features_in_ = model_dict['n_features_in_']
    model.n_steps_ = model_dict['n_steps_']

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])
    if '_l1_reg_W' in model_dict.keys():
        model._l1_reg_W = model_dict['_l1_reg_W']
    if '_l1_reg_H' in model_dict.keys():
        model._l1_reg_H = model_dict['_l1_reg_H']
    if '_l2_reg_W' in model_dict.keys():
        model._l2_reg_W = model_dict['_l2_reg_W']
    if '_l2_reg_H' in model_dict.keys():
        model._l2_reg_H = model_dict['_l2_reg_H']

    if '_batch_size' in model_dict.keys():
        model._batch_size = model_dict['_batch_size']
    if '_components_denominator' in model_dict.keys():
        model._components_denominator = np.array(model_dict['_components_denominator'])
    if '_components_numerator' in model_dict.keys():
         model._components_numerator = np.array(model_dict['_components_numerator'])
    if '_ewa_cost' in model_dict.keys():
        model._ewa_cost = model_dict['_ewa_cost']
    if '_ewa_cost_min' in model_dict.keys():
        model._ewa_cost_min = model_dict['_ewa_cost_min']
    if '_no_improvement' in model_dict.keys():
        model._no_improvement = model_dict['_no_improvement']
    if '_rho' in model_dict.keys():
        model._rho = model_dict['_rho']

    return model


def serialize_minibatch_dictionary_learning(model):
    serialized_model = {
        'meta': 'minibatch-dictionary-learning',
        'components_': model.components_.tolist(),
        'n_iter_': model.n_iter_,
        'n_steps_': model.n_steps_,
        'n_features_in_': model.n_features_in_,
        'params': model.get_params(),
    }

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()

    return serialized_model


def deserialize_minibatch_dictionary_learning(model_dict):
    model = MiniBatchDictionaryLearning(**model_dict['params'])

    model.components_ = np.array(model_dict['components_'])
    model.n_iter_ = model_dict['n_iter_']
    model.n_steps_ = model_dict['n_steps_']
    model.n_features_in_ = model_dict['n_features_in_']

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

    return model


def serialize_sparse_coder(model):
    serialized_model = {
        'meta': 'sparse-coder',
        'params': model.get_params(),
    }

    serialized_model['params']['dictionary'] = serialized_model['params']['dictionary'].tolist()

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()

    return serialized_model


def deserialize_sparse_coder(model_dict):

    model_dict['params']['dictionary'] = np.array(model_dict['params']['dictionary'])

    model = SparseCoder(**model_dict['params'])

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

    return model


def serialize_truncated_svd(model):
    serialized_model = {
        'meta': 'truncated-svd',
        'components_': model.components_.tolist(),
        'explained_variance_': model.explained_variance_.tolist(),
        'explained_variance_ratio_': model.explained_variance_ratio_.tolist(),
        'singular_values_': model.singular_values_.tolist(),
        'n_features_in_': model.n_features_in_,
        'params': model.get_params(),
    }

    if 'feature_names_in_' in model.__dict__:
        serialized_model['feature_names_in_'] = model.feature_names_in_.tolist()

    return serialized_model


def deserialize_truncated_svd(model_dict):
    model = TruncatedSVD(**model_dict['params'])

    model.components_ = np.array(model_dict['components_'])
    model.explained_variance_ = np.array(model_dict['explained_variance_'])
    model.explained_variance_ratio_ = np.array(model_dict['explained_variance_ratio_'])
    model.singular_values_ = np.array(model_dict['singular_values_'])
    model.n_features_in_ = model_dict['n_features_in_']

    if 'feature_names_in_' in model_dict.keys():
        model.feature_names_in_ = np.array(model_dict['feature_names_in_'][0])

    return model
