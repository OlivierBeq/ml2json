# -*- coding: utf-8 -*-

from . import _base

# Allow additional dependencies to be optional
__optionals__ = []
try:
    from pynndescent import NNDescent, PyNNDescentTransformer
    __optionals__.extend(['NNDescent', 'PyNNDescentTransformer'])
except:
    pass


def serialize_nearest_neighbors(model):
    return _base.serialize_model_generic(model)


def deserialize_nearest_neighbors(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_kernel_density(model):
    return _base.serialize_model_generic(model)


def deserialize_kernel_density(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_kdtree(model):
    return _base.serialize_kdtree(model)


def deserialize_kdtree(model_dict):
    # ml2json's top-level dispatcher overwrites 'meta' with the library-
    # qualified tag before routing here; _base.deserialize_kdtree asserts on
    # its own internal 'kdtree' tag, so restore it for that call only.
    return _base.deserialize_kdtree({**model_dict, 'meta': 'kdtree'})


def serialize_balltree(model):
    return _base.serialize_balltree(model)


def deserialize_balltree(model_dict):
    # Same 'meta' restore dance as deserialize_kdtree above.
    return _base.deserialize_balltree({**model_dict, 'meta': 'balltree'})


def serialize_kneighbors_transformer(model):
    return _base.serialize_model_generic(model)


def deserialize_kneighbors_transformer(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_radius_neighbors_transformer(model):
    return _base.serialize_model_generic(model)


def deserialize_radius_neighbors_transformer(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_local_outlier_factor(model):
    return _base.serialize_model_generic(model)


def deserialize_local_outlier_factor(model_dict):
    return _base.deserialize_model_generic(model_dict)


def serialize_neighborhood_components_analysis(model):
    return _base.serialize_model_generic(model)


def deserialize_neighborhood_components_analysis(model_dict):
    return _base.deserialize_model_generic(model_dict)


if 'NNDescent' in __optionals__:
    def serialize_nndescent(model):
        return _base.serialize_nndescent(model)


    def deserialize_nndescent(model_dict):
        return _base.deserialize_nndescent({**model_dict, 'meta': 'nn-descent'})


if 'PyNNDescentTransformer' in __optionals__:
    # A thin TransformerMixin wrapper storing its NNDescent as the `index_`
    # attribute; the generic engine recurses into it and hits NNDescent's own
    # leaf-type handler (_base.serialize_nndescent/deserialize_nndescent)
    # automatically, so no hand-written logic is needed here.
    def serialize_pynndescent_transformer(model):
        return _base.serialize_model_generic(model)


    def deserialize_pynndescent_transformer(model_dict):
        return _base.deserialize_model_generic(model_dict)
