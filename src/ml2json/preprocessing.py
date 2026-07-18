# -*- coding: utf-8 -*-

from . import _base


def serialize_label_binarizer(model):
    return _base.serialize_model_generic(model, meta='label-binarizer')


def deserialize_label_binarizer(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_multilabel_binarizer(model):
    return _base.serialize_model_generic(model, meta='multilabel-binarizer')


def deserialize_multilabel_binarizer(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_minmax_scaler(model):
    return _base.serialize_model_generic(model, meta='minmax-scaler')


def deserialize_minmax_scaler(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_standard_scaler(model):
    return _base.serialize_model_generic(model, meta='standard-scaler')


def deserialize_standard_scaler(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_robust_scaler(model):
    return _base.serialize_model_generic(model, meta='robust-scaler')


def deserialize_robust_scaler(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_maxabs_scaler(model):
    return _base.serialize_model_generic(model, meta='maxabs-scaler')


def deserialize_maxabs_scaler(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_label_encoder(model):
    return _base.serialize_model_generic(model, meta='label-encoder')


def deserialize_label_encoder(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_kernel_centerer(model):
    return _base.serialize_model_generic(model, meta='kernel-centerer')


def deserialize_kernel_centerer(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_onehot_encoder(model):
    return _base.serialize_model_generic(model, meta='onehot-encoder')


def deserialize_onehot_encoder(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_ordinal_encoder(model):
    return _base.serialize_model_generic(model, meta='ordinal-encoder')


def deserialize_ordinal_encoder(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_normalizer(model):
    return _base.serialize_model_generic(model, meta='normalizer')


def deserialize_normalizer(model_dict):
    return _base.deserialize_model_generic(model_dict)
