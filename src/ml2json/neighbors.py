# -*- coding: utf-8 -*-

from . import _base

# Allow additional dependencies to be optional
__optionals__ = []
try:
    from pynndescent import NNDescent
    __optionals__.append('NNDescent')
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


if 'NNDescent' in __optionals__:
    def serialize_nndescent(model):
        return _base.serialize_nndescent(model)


    def deserialize_nndescent(model_dict):
        return _base.deserialize_nndescent({**model_dict, 'meta': 'nn-descent'})
