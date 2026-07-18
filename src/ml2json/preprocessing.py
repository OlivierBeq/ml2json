# -*- coding: utf-8 -*-

from . import _base


def serialize_label_binarizer(model):
    return _base.serialize_model_generic(model)


def deserialize_label_binarizer(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_multilabel_binarizer(model):
    return _base.serialize_model_generic(model)


def deserialize_multilabel_binarizer(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_minmax_scaler(model):
    return _base.serialize_model_generic(model)


def deserialize_minmax_scaler(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_standard_scaler(model):
    return _base.serialize_model_generic(model)


def deserialize_standard_scaler(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_robust_scaler(model):
    return _base.serialize_model_generic(model)


def deserialize_robust_scaler(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_maxabs_scaler(model):
    return _base.serialize_model_generic(model)


def deserialize_maxabs_scaler(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_label_encoder(model):
    return _base.serialize_model_generic(model)


def deserialize_label_encoder(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_kernel_centerer(model):
    return _base.serialize_model_generic(model)


def deserialize_kernel_centerer(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_onehot_encoder(model):
    return _base.serialize_model_generic(model)


def deserialize_onehot_encoder(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_ordinal_encoder(model):
    return _base.serialize_model_generic(model)


def deserialize_ordinal_encoder(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_normalizer(model):
    return _base.serialize_model_generic(model)


def deserialize_normalizer(model_dict):
    return _base.deserialize_model_generic(model_dict)


# Note: FunctionTransformer is intentionally not supported - it can hold an
# arbitrary user-supplied Python callable (`func`/`inverse_func`), which
# isn't safely JSON-serializable (and reconstructing/calling arbitrary code
# from a JSON payload would be a security hazard regardless).

def serialize_binarizer(model):
    return _base.serialize_model_generic(model)


def deserialize_binarizer(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_power_transformer(model):
    return _base.serialize_model_generic(model)


def deserialize_power_transformer(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_quantile_transformer(model):
    return _base.serialize_model_generic(model)


def deserialize_quantile_transformer(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_kbins_discretizer(model):
    return _base.serialize_model_generic(model)


def deserialize_kbins_discretizer(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_polynomial_features(model):
    return _base.serialize_model_generic(model)


def deserialize_polynomial_features(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_spline_transformer(model):
    return _base.serialize_model_generic(model)


def deserialize_spline_transformer(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_target_encoder(model):
    return _base.serialize_model_generic(model)


def deserialize_target_encoder(model_dict):
    return _base.deserialize_model_generic(model_dict)
