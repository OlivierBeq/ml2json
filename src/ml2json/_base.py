# -*- coding: utf-8 -*-

import sys
import inspect
import importlib
import warnings

from typing import Any
from itertools import chain

import numpy as np
import scipy as sp
from joblib import Memory
from numpy.random import RandomState
import sklearn
from sklearn.utils import Bunch
from sklearn.cluster import Birch
from sklearn.cluster._birch import _CFNode, _CFSubcluster
from sklearn.tree._tree import Tree
from sklearn._loss._loss import (CyAbsoluteError, CyExponentialLoss, CyHalfBinomialLoss, CyHalfGammaLoss,
                                 CyHalfMultinomialLoss, CyHalfPoissonLoss, CyHalfSquaredError, CyHalfTweedieLoss,
                                 CyHalfTweedieLossIdentity, CyHuberLoss, CyPinballLoss)
from sklearn.linear_model._sgd_fast import (EpsilonInsensitive, Hinge, ModifiedHuber,
                                            SquaredEpsilonInsensitive, SquaredHinge)
from sklearn.linear_model._stochastic_gradient import BaseSGD

DEFAULT_REC_DEPTH = sys.getrecursionlimit()
sys.setrecursionlimit(int(2 ** 30))


def recursive_serialize(obj: object):
    # print(obj)
    # Base types
    if isinstance(obj, (int, str, float)) or obj is None:
        return obj
    # List and Tuples
    elif isinstance(obj, list):
        return ([recursive_serialize(item) for item in obj])
    elif isinstance(obj, list):
        return str(tuple(recursive_serialize(item) for item in obj))
    elif isinstance(obj, set):
        return str(set(recursive_serialize(item) for item in obj))
    # Dictionary
    elif isinstance(obj, dict):
        return {str(recursive_serialize(key)): recursive_serialize(value)
                for key, value in obj.items()}
    # Object with predefined serialization method
    elif isinstance(obj, tuple(chain(__serialize_obj_fn__.keys()))):
        for obj_type, serialize_fn in __serialize_obj_fn__.items():
            if isinstance(obj, obj_type):
                return str(serialize_fn(obj))
    # Object is a type
    if type(obj).__name__ == 'type':
        for obj_type, serialize_type in __serialize_obj_types__.items():
            if obj is obj_type:
                return str(serialize_type(obj))
    # Object with __dict__ attribute
    elif hasattr(obj, '__dict__'):
        return {'meta': type(obj).__name__.lower()} | {str(recursive_serialize(key)): recursive_serialize(value)
                                                           for key, value in obj.__dict__.items()}
    else:
        raise RuntimeError(f'Cannot find type: {type(obj)}')


def recursive_deserialize(obj: object):
    pass


def serialize_numpy_value(value: np.bool_ | np.int8 | np.uint8 | np.int16 | np.uint16 | np.int32 | np.uint32 | np.intp | np.uintp | np.float16 | np.float32 | np.float64 | np.longdouble | np.complex64 | np.complex128 | np.clongdouble):
    return {'meta': 'numpy_value', 'module': inspect.getmodule(type(value)).__name__, 'type': type(value).__name__, 'value': value.item()}


def deserialize_numpy_value(model_dict: dict[str, Any]):
    assert model_dict['meta'] == 'numpy_value'
    return getattr(importlib.import_module(model_dict['module']), model_dict['type'])(model_dict['value'])


def serialize_numpy_type(dtype: np.dtype):
    assert any(dtype is x for x in (np.bool_, np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.intp,
                                    np.uintp, np.float16, np.float32, np.float64, np.longdouble, np.complex64,
                                    np.complex128, np.clongdouble))
    return {'meta': 'numpy_type', 'module': 'numpy', 'type': dtype.__name__}


def deserialize_numpy_type(model_dict: dict[str, Any]):
    assert model_dict['meta'] == 'numpy_type'
    return getattr(importlib.import_module(model_dict['module']), model_dict['type'])(model_dict['value'])


def serialize_numpy_dtype(dtype: np.dtype):
    assert isinstance(dtype, np.dtype)
    return {'meta': 'numpy_dtype', 'module': 'numpy', 'type': serialize_numpy_type(dtype.type)}


def deserialize_numpy_dtype(model_dict: dict[str, Any]):
    assert model_dict['meta'] == 'numpy_dtype'
    return getattr(importlib.import_module(model_dict['module']), 'dtype')(model_dict['type'])


def serialize_numpy_array(array: np.ndarray):
    assert isinstance(array, np.ndarray)
    return {'meta': 'numpy_array', 'module': 'numpy', 'type': 'array', 'values': recursive_serialize(array.tolist()), 'value_type': str(array.dtype)}


def deserialize_numpy_array(model_dict: dict[str, Any]):
    assert model_dict['meta'] == 'numpy_array'
    return getattr(importlib.import_module(model_dict['module']), model_dict['type'])(model_dict['values'], dtype=getattr(importlib.import_module(model_dict['module']), model_dict['value_type']))


def serialize_random_state(random_state: RandomState):
    assert isinstance(random_state, RandomState)
    model_dict = {'meta': 'random_state', 'random_state': random_state.get_state(legacy=False)}
    model_dict['random_state']['state']['key'] = recursive_serialize(model_dict['random_state']['state']['key'])
    return model_dict


def deserialize_random_state(model_dict: dict[str, Any]):
    assert model_dict['meta'] == 'random_state'
    model_dict['random_state']['state']['key'] = recursive_deserialize(model_dict['random_state']['state']['key'])
    random_state = np.random.RandomState()
    random_state.set_state(model_dict['random_state'])
    return random_state


def serialize_csr_matrix(csr_matrix: sp.sparse.csr_matrix):
    assert sp.sparse.issparse(csr_matrix)
    serialized_csr_matrix = {
        'meta': 'scipy_csr',
        'data': recursive_serialize(csr_matrix.data),
        'indices': recursive_serialize(csr_matrix.indices),
        'indptr': recursive_serialize(csr_matrix.indptr),
        '_shape': csr_matrix._shape,
    }
    return serialized_csr_matrix


def deserialize_csr_matrix(csr_dict: dict[str, Any]):
    assert csr_dict['meta'] == 'scipy_csr'
    csr_matrix = sp.sparse.csr_matrix(tuple(csr_dict['_shape']))
    csr_matrix.data = recursive_deserialize(csr_dict['data'])
    csr_matrix.indices = recursive_deserialize(csr_dict['indices'])
    csr_matrix.indptr = recursive_deserialize(csr_dict['indptr'])
    return csr_matrix


def serialize_bunch(bunch: Bunch):
    assert isinstance(bunch, Bunch)
    serialized_bunch = {
        'meta': 'bunch',
        'items': {param: recursive_serialize(value)
                  for param, value in bunch.items()}
    }
    return serialized_bunch


def deserialize_bunch(bunch_dict: dict[str, Any]):
    assert bunch_dict['meta'] == 'bunch'
    bunch = Bunch(**{param: recursive_deserialize(value)
                     for param, value in bunch_dict['items'].items()})
    return bunch


def serialize_memory(memory: Memory):
    assert isinstance(memory, Memory)
    serialized_memory = {
        'meta': 'memory',
        'depth': memory.depth,
        '_verbose': memory._verbose,
        'mmap_mode': memory.mmap_mode,
        'timestamp': memory.timestamp,
        'bytes_limit': memory.bytes_limit,
        'backend': memory.backend,
        'compress': memory.compress,
        'backend_options': memory.backend_options,
        'location': memory.location,
    }
    return serialized_memory


def deserialize_memory(memory_dict: dict[str, Any]):
    assert memory_dict['meta'] == 'memory'
    memory = Memory(location=memory_dict['location'],
                    backend=memory_dict['backend'],
                    mmap_mode=memory_dict['mmap_mode'],
                    compress=memory_dict['compress'],
                    verbose=memory_dict['_verbose'],
                    bytes_limit=memory_dict['bytes_limit'],
                    backend_options=memory_dict['backend_options'])
    memory.depth = memory_dict['depth']
    memory.timestamp = memory_dict['timestamp']
    return memory


def serialize_cfnode(model: _CFNode):
    assert isinstance(model, _CFNode)
    # Get memory address
    mem = lambda x: hex(id(x)) if x is not None else None

    serialized_model = {
        'meta': 'cfnode',
        'threshold': model.threshold,
        'branching_factor': model.branching_factor,
        'is_leaf': model.is_leaf,
        'n_features': model.n_features,
        'subclusters_': [mem(cfsubcluster) for cfsubcluster in model.subclusters_],
        'init_centroids_': recursive_serialize(model.init_centroids_),
        'init_sq_norm_': recursive_serialize(model.init_sq_norm_),
        'squared_norm_': recursive_serialize(model.squared_norm_) if isinstance(model.squared_norm_, np.ndarray) else model.squared_norm_,
        'prev_leaf_': mem(model.prev_leaf_),
        'next_leaf_': mem(model.next_leaf_),
    }

    if hasattr(model, 'centroids_'):
        serialized_model['centroids_'] = model.centroids_.tolist()
    serialized_model['dtype'] = str(model.init_sq_norm_.dtype)

    return serialized_model


def deserialize_cfnode(model_dict: dict[str, Any]):
    assert model_dict['meta'] == 'cfnode'
    if sklearn.__version__ < '1.2.0':
        model = _CFNode(threshold=model_dict['threshold'],
                        branching_factor=model_dict['branching_factor'],
                        is_leaf=model_dict['is_leaf'],
                        n_features=model_dict['n_features'])
    else:
        model = _CFNode(threshold=model_dict['threshold'],
                        branching_factor=model_dict['branching_factor'],
                        is_leaf=model_dict['is_leaf'],
                        n_features=model_dict['n_features'],
                        dtype=np.dtype(model_dict['dtype']))

    model.init_centroids_ = np.array(model_dict['init_centroids_'])
    model.init_sq_norm_ = np.array(model_dict['init_sq_norm_'])
    model.squared_norm_ = np.array(model_dict['squared_norm_'])
    # To be modified by the Birch deserializer
    model.subclusters_ = model_dict['subclusters_']
    model.prev_leaf_ = model_dict['prev_leaf_']
    model.next_leaf_ = model_dict['next_leaf_']
    return model


def serialize_cfsubcluster(model: _CFSubcluster):
    assert isinstance(model, _CFSubcluster)
    # Get memory address
    mem = lambda x: hex(id(x)) if x is not None else None
    serialized_model = {
        'meta': 'cfsubcluster',
        'n_samples_': model.n_samples_,
        'squared_sum_': model.squared_sum_,
        'centroid_': model.centroid_.tolist(),
        'linear_sum_': model.linear_sum_.tolist(),
        'sq_norm_': model.sq_norm_,
        'child_': mem(model.child_)
    }
    return serialized_model


def deserialize_cfsubcluster(model_dict: dict[str, Any]):
    assert model_dict['meta'] == 'cfsubcluster'
    model = _CFSubcluster()
    model.n_samples_ = model_dict['n_samples_']
    model.squared_sum_ = model_dict['squared_sum_']
    model.centroid_ = np.array(model_dict['centroid_'])
    model.linear_sum_ = np.array(model_dict['linear_sum_'])
    model.sq_norm_ = model_dict['sq_norm_']
    model.child_ = model_dict['child_']
    return model


def serialize_birch(model):
    # Get memory address
    mem = lambda x: hex(id(x)) if x is not None else None

    # Define a recursive aggregator of _CFNodes and _CFSubclusters
    def get_nodes_and_subclusters(node):
        if node is None:
            return [], []
        nodes, subclusters = [(mem(node), node)], []
        for subcluster in node.subclusters_:
            subnodes, subsubclusters = get_nodes_and_subclusters(subcluster.child_)
            nodes += subnodes
            subclusters += [(mem(subcluster), subcluster)] + subsubclusters
        return nodes, subclusters

    # Obtain _CFNodes and _CFSubclusters
    nodes, subclusters = get_nodes_and_subclusters(model.root_)
    # Add the dummy_leaf to nodes
    nodes = [(mem(model.dummy_leaf_) if model.dummy_leaf_ is not None else None, model.dummy_leaf_)] + nodes
    # Serialize nodes
    nodes = {uid: serialize_cfnode(node) for uid, node in nodes}
    subclusters = {uid: serialize_cfsubcluster(subcluster) for uid, subcluster in subclusters}

    serialized_model = {
        'meta': 'birch',
        'root_': mem(model.root_),
        'dummy_leaf_': mem(model.dummy_leaf_),
        'subcluster_centers_': model.subcluster_centers_.tolist(),
        '_n_features_out': model._n_features_out,
        '_subcluster_norms': model._subcluster_norms.tolist(),
        'subcluster_labels_': model.subcluster_labels_.tolist(),
        'labels_': model.labels_.tolist(),
        'n_features_in_': model.n_features_in_,
        'params': model.get_params(),
        'nodes': nodes,
        'subclusters': subclusters
    }

    if '_deprecated_fit' in model.__dict__:
        serialized_model['_deprecated_fit'] = model._deprecated_fit
        serialized_model['_deprecated_partial_fit'] = model._deprecated_partial_fit

    return serialized_model


def deserialize_birch(model_dict):
    assert model_dict['meta'] == 'birch'
    model = Birch(**model_dict['params'])
    model.subcluster_centers_ = np.array(model_dict['subcluster_centers_'])
    model._n_features_out = model_dict['_n_features_out']
    model._subcluster_norms = np.array(model_dict['_subcluster_norms'])
    model.subcluster_labels_ = np.array(model_dict['subcluster_labels_'])
    model.labels_ = np.array(model_dict['labels_'])
    model.n_features_in_ = model_dict['n_features_in_']
    # Deserialize _CFNodes and _CFSubclusters
    nodes = {uid: deserialize_cfnode(node) for uid, node in model_dict['nodes'].items()}
    subclusters = {uid: deserialize_cfsubcluster(subcluster) for uid, subcluster in model_dict['subclusters'].items()}
    # Link prev_leaf_ and next_leaf of _CFNodes to other _CFNodes
    for node_uid in nodes.keys():
        prev_leaf_uid = nodes[node_uid].prev_leaf_
        next_leaf_uid = nodes[node_uid].next_leaf_
        if prev_leaf_uid is not None:
            nodes[node_uid].prev_leaf_ = nodes[prev_leaf_uid]
        if next_leaf_uid is not None:
            nodes[node_uid].next_leaf_ = nodes[next_leaf_uid]
    # Link child_ of _CFSubclusters to _CFNodes
    for subcluster_uid in subclusters.keys():
        subclusters[subcluster_uid].child_ = subclusters[subcluster_uid]
    # Link subclusters_ of _CFNodes to _CFSubclusters
    for node_uid in nodes.keys():
        old_uids = nodes[node_uid].subclusters_
        if old_uids is not None:
            nodes[node_uid].subclusters_ = [subclusters[old_uid] for old_uid in old_uids]
    # Link root_ and dummy_leaf_ _CFNodes
    model.dummy_leaf_ = nodes[model_dict['dummy_leaf_']]
    model.root_ = nodes[model_dict['root_']]
    if '_deprecated_fit' in model_dict:
        model._deprecated_fit = model_dict['_deprecated_fit']
        model._deprecated_partial_fit = model_dict['_deprecated_partial_fit']
    return model


def serialize_tree(tree: Tree):
    assert isinstance(tree, Tree)
    serialized_tree = tree.__getstate__()
    dtypes = [serialized_tree['nodes'].dtype[i].str for i in range(len(serialized_tree['nodes'].dtype))]
    serialized_tree['nodes'] = serialized_tree['nodes'].tolist()
    serialized_tree['values'] = serialized_tree['values'].tolist()
    return {'meta': 'tree', 'tree': serialized_tree, 'nodes_dtype': dtypes}


def deserialize_tree(tree_dict: dict[str, Any], n_features: int, n_outputs: int):
    assert tree_dict['meta'] == 'tree'
    tree_dict['tree']['nodes'] = [tuple(lst) for lst in tree_dict['tree']['nodes']]
    names = ['left_child', 'right_child', 'feature', 'threshold', 'impurity', 'n_node_samples', 'weighted_n_node_samples']
    if sklearn.__version__ >= '1.3':
        names.append('missing_go_to_left')
    tree_dict['tree']['nodes'] = np.array(tree_dict['tree']['nodes'], dtype=np.dtype({'names': names, 'formats': tree_dict['nodes_dtype']}))
    tree_dict['tree']['values'] = np.array(tree_dict['tree']['values'])
    # Dummy classes
    dummy_classes = np.array([1] * n_outputs, dtype=np.intp)
    tree = Tree(n_features, dummy_classes, n_outputs)
    tree.__setstate__(tree_dict['tree'])
    return tree


def serialize_cyloss(loss: CyAbsoluteError |CyExponentialLoss |CyHalfBinomialLoss |CyHalfGammaLoss |CyHalfMultinomialLoss |CyHalfPoissonLoss |CyHalfSquaredError |CyHalfTweedieLoss |CyHalfTweedieLossIdentity |CyHuberLoss |CyPinballLoss):
    assert isinstance(loss, (CyAbsoluteError, CyExponentialLoss, CyHalfBinomialLoss, CyHalfGammaLoss, CyHalfMultinomialLoss, CyHalfPoissonLoss, CyHalfSquaredError, CyHalfTweedieLoss, CyHalfTweedieLossIdentity, CyHuberLoss, CyPinballLoss))
    mode_dict = {'meta': 'cython_loss', 'type': type(loss).__name__}
    if isinstance(loss, CyHuberLoss):
        mode_dict['delta'] = loss.delta
    if isinstance(loss, (CyHalfTweedieLoss, CyHalfTweedieLossIdentity)):
        mode_dict['power'] = loss.power
    if isinstance(loss, CyPinballLoss):
        mode_dict['quantile'] = loss.quantile
    return mode_dict


def deserialize_cyloss(loss_dict: dict[str, Any]):
    assert loss_dict['meta'] == 'cython_loss'
    if loss_dict['meta'] == 'CyAbsoluteError':
        return CyAbsoluteError()
    if loss_dict['meta'] == 'CyExponentialLoss':
        return CyExponentialLoss()
    if loss_dict['meta'] == 'CyHalfBinomialLoss':
        return CyHalfBinomialLoss()
    if loss_dict['meta'] == 'CyHalfGammaLoss':
        return CyHalfGammaLoss()
    if loss_dict['meta'] == 'CyHalfMultinomialLoss':
        return CyHalfMultinomialLoss()
    if loss_dict['meta'] == 'CyHalfPoissonLoss':
        return CyHalfPoissonLoss()
    if loss_dict['meta'] == 'CyHalfSquaredError':
        return CyHalfSquaredError()
    if loss_dict['meta'] == 'CyHalfTweedieLoss':
        return CyHalfTweedieLoss(loss_dict['power'])
    if loss_dict['meta'] == 'CyHalfTweedieLossIdentity':
        return CyHalfTweedieLossIdentity(loss_dict['power'])
    if loss_dict['meta'] == 'CyHuberLoss':
        return CyHuberLoss(loss_dict['delta'])
    if loss_dict['meta'] == 'CyPinballLoss':
        return CyPinballLoss(loss_dict['quantile'])


def serialize_random_generator(generator: np.random.Generator):
    assert isinstance(generator, np.random.Generator)
    return {'meta': 'random_generator', 'bit_generator_type': type(generator.bit_generator).__name__, 'state': recursive_serialize(generator.bit_generator.state)}


def deserialize_random_generator(generator_dict: dict[str, Any]):
    assert generator_dict['meta'] == 'random_generator'
    rng = getattr(importlib.import_module('numpy'), generator_dict['bit_generator_type'])()
    rng.state = recursive_deserialize(generator_dict['state'])
    return rng


def serialize_sgd_loss_functions(loss: Hinge | SquaredHinge | CyHalfBinomialLoss | ModifiedHuber | EpsilonInsensitive | SquaredEpsilonInsensitive):
    assert any(loss is x for x in (Hinge, SquaredHinge, CyHalfBinomialLoss, ModifiedHuber, EpsilonInsensitive, SquaredEpsilonInsensitive))
    return {'meta': 'sgd_loss_functions', 'module': inspect.getmodule(loss).__name__, 'type': loss.__name__}


def deserialize_sgd_loss_functions(loss_dict: dict[str, Any]):
    assert loss_dict['meta'] == 'sgd_loss_functions'
    return getattr(importlib.import_module(loss_dict['module']), loss_dict['type'])


def remove_superfluous_attribute(model: Any):
    if isinstance(model, BaseSGD) and hasattr(model, '_loss_function_'):
        delattr(model, '_loss_function_')


def serialize_unfitted_model(model):
    """Serialize an unfitted model.

    :param model: unfitted model
    """
    serialized_model = {
        'meta': 'unfit_model',
        'unfitted': True,
        'type': (inspect.getmodule(model).__name__,
                 type(model).__name__),
        'params': model.get_params()
    }
    serialize_version(model, serialized_model)
    return serialized_model


def deserialize_unfitted_model(model_dict: dict[str, Any]):
    """Deserialize an unfitted model.

    :param model_dict: previously serialized unfitted model
    """
    assert model_dict['meta'] == 'unfit_model'
    check_version(model_dict)
    model = getattr(importlib.import_module(model_dict['type'][0]), model_dict['type'][1])(**model_dict['params'])
    return model


def serialize_version(model, model_dict):
    """Add version(s) of the libraries required to instantiate the model.

    :param model: model to check the dependencies of
    :param model_dict: serialized model to add the dependencies' versions to
    """
    # Obtain library used to fit the model
    module = inspect.getmodule(model)
    if module is None:
        return model_dict
    module = sys.modules[module.__name__.partition('.')[0]]
    version = module.__version__ if hasattr(module, '__version__') else ''
    model_dict['versions'] = (module.__name__, version)
    return model_dict


def check_version(model_dict):
    """Check if the versions of the installed libraries and those the model was fitted with correspond.

    :param model_dict: serialized model
    """
    if 'versions' not in model_dict:
        return
    # Obtain module used to fit the model
    module_name, version = model_dict['versions']
    # Module is installed
    installed = importlib.util.find_spec(module_name) is not None
    if not installed:
        raise ModuleNotFoundError(f'Module {module_name} could not be found. Is it installed?')
    # Check version of the installed module
    if version == '':
        return
    installed_version = importlib.import_module(module_name).__version__
    if version != installed_version:
        warnings.warn(f'Version of the current {module_name} library ({installed_version}) '
                      f'does not match the version used to fit the serialized model ({version})')


class ModelNotSupported(Exception):
    """Custom class for unsupported model types."""
    pass


__serialize_obj_fn__ = {np.ndarray: serialize_numpy_array,
                        RandomState: serialize_random_state,
                        Memory: serialize_memory,
                        sp.sparse.csr_matrix: serialize_csr_matrix,
                        Bunch: serialize_bunch,
                        Birch: serialize_birch,
                        _CFNode: serialize_cfnode,
                        _CFSubcluster: serialize_cfsubcluster,
                        (np.int8, np.int16, np.int32, np.int64,
                         np.byte, np.short, np.intc, np.int_, np.long, np.longlong,
                         np.uint8, np.uint16, np.uint32, np.uint64,
                         np.ubyte, np.ushort, np.uintc, np.uint, np.ulong, np.ulonglong,
                         np.intp, np.uintp,
                         np.float16, np.float32, np.float64, # float96's or float128's names are platform dependent
                         np.half, np.single, np.double, np.longdouble,
                         np.complex64, np.complex128, # complex192's or comple256's names are platform dependent
                         np.csingle, np.cdouble, np.clongdouble): serialize_numpy_value,
                        Tree: serialize_tree,
                        (CyAbsoluteError, CyExponentialLoss, CyHalfBinomialLoss, CyHalfGammaLoss,
                         CyHalfMultinomialLoss, CyHalfPoissonLoss, CyHalfSquaredError, CyHalfTweedieLoss,
                         CyHalfTweedieLossIdentity, CyHuberLoss, CyPinballLoss): serialize_cyloss,
                        np.random.Generator: serialize_random_generator,
                        np.dtype: serialize_numpy_dtype,
                        BaseSGD: remove_superfluous_attribute,
                        }

__serialize_obj_types__ = {(np.int8, np.int16, np.int32, np.int64,
                            np.byte, np.short, np.intc, np.int_, np.long, np.longlong,
                            np.uint8, np.uint16, np.uint32, np.uint64,
                            np.ubyte, np.ushort, np.uintc, np.uint, np.ulong, np.ulonglong,
                            np.intp, np.uintp,
                            np.float16, np.float32, np.float64, # float96's or float128's names are platform dependent
                            np.half, np.single, np.double, np.longdouble,
                            np.complex64, np.complex128, # complex192's or comple256's names are platform dependent
                            np.csingle, np.cdouble, np.clongdouble): serialize_numpy_type,
                           (Hinge, SquaredHinge, ModifiedHuber, EpsilonInsensitive, SquaredEpsilonInsensitive,
                            CyAbsoluteError, CyExponentialLoss, CyHalfBinomialLoss, CyHalfGammaLoss,
                            CyHalfMultinomialLoss, CyHalfPoissonLoss, CyHalfSquaredError, CyHalfTweedieLoss,
                            CyHalfTweedieLossIdentity, CyHuberLoss, CyPinballLoss): serialize_sgd_loss_functions,
                           }
