# -*- coding: utf-8 -*-

import numpy as np
from sklearn.feature_extraction import DictVectorizer

from . import _base


def serialize_dict_vectorizer(model):
    serialized_model = {
        'dtype': model.dtype.__name__,
        'separator': model.separator,
        'sparse': model.sparse,
        'sort': model.sort,
        'feature_names': model.feature_names_,
        'vocabulary': model.vocabulary_
    }

    return serialized_model


def deserialize_dict_vectorizer(model_dict):
    model = DictVectorizer()

    model.dtype = np.dtype(model_dict['dtype']).type
    model.separator = model_dict['separator']
    model.sparse = model_dict['sparse']
    model.sort = model_dict['sort']
    model.feature_names_ = model_dict['feature_names']
    model.vocabulary_ = model_dict['vocabulary']

    return model


def serialize_feature_hasher(model):
    return _base.serialize_model_generic(model)


def deserialize_feature_hasher(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_count_vectorizer(model):
    return _base.serialize_model_generic(model)


def deserialize_count_vectorizer(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_tfidf_transformer(model):
    return _base.serialize_model_generic(model)


def deserialize_tfidf_transformer(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_tfidf_vectorizer(model):
    return _base.serialize_model_generic(model)


def deserialize_tfidf_vectorizer(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_hashing_vectorizer(model):
    return _base.serialize_model_generic(model)


def deserialize_hashing_vectorizer(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_patch_extractor(model):
    return _base.serialize_model_generic(model)


def deserialize_patch_extractor(model_dict):
    return _base.deserialize_model_generic(model_dict)
