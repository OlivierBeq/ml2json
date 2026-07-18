# -*- coding: utf-8 -*-

"""Generic recursive (de)serialization engine.

Instead of hand-writing a serialize_X/deserialize_X function pair for every
supported model class, `serialize_model_generic`/`deserialize_model_generic`
walk a fitted object's `__dict__` recursively, encoding base types directly
and delegating to a small registry of "leaf" handlers for the few types that
aren't JSON-representable as-is (numpy arrays/scalars/dtypes, scipy sparse
matrices, RandomState, joblib Memory, sklearn's Bunch). Values that are
themselves other supported ml2json models are recursed into via the normal
`serialize_model`/`deserialize_model` dispatcher.

A few types need dedicated (de)serialize functions rather than the fully
generic `object.__new__(cls)` + `__dict__` restore path - typically because
they're Cython extension types that reject `object.__new__` outright (KDTree,
Tree) and must be constructed through their own two-step constructor/
`__setstate__` dance instead. These are still registered as uniform leaf
types (a plain `type -> (serialize_fn, deserialize_fn)` entry), just with a
dedicated reconstruction function instead of the generic one.

Birch's `_CFNode`/`_CFSubcluster` object graph (shared and circular
references) is the one case that doesn't fit even that: it's kept here as
standalone helpers for hand-written serializers that need them, exactly as
before.
"""

import ast
import functools
import importlib
import sys
import types

import numpy as np
import scipy as sp
import sklearn
from joblib import Memory
from numpy.random import RandomState
from sklearn.utils import Bunch
from sklearn.cluster._birch import _CFNode, _CFSubcluster
from sklearn.ensemble._hist_gradient_boosting.predictor import TreePredictor
from sklearn.neighbors import BallTree, KDTree
from sklearn.tree._tree import Tree
from sklearn._loss._loss import (CyAbsoluteError, CyExponentialLoss, CyHalfBinomialLoss, CyHalfGammaLoss,
                                 CyHalfMultinomialLoss, CyHalfPoissonLoss, CyHalfSquaredError, CyHalfTweedieLoss,
                                 CyHalfTweedieLossIdentity, CyHuberLoss, CyPinballLoss)
from sklearn.linear_model._sgd_fast import (EpsilonInsensitive, Hinge, ModifiedHuber,
                                            SquaredEpsilonInsensitive, SquaredHinge)
from sklearn.linear_model._stochastic_gradient import BaseSGD

from .utils.csr import serialize_csr_matrix, deserialize_csr_matrix
from .utils.bunch import serialize_bunch, deserialize_bunch
from .utils.random_state import serialize_random_state, deserialize_random_state
from .utils.memory import serialize_memory, deserialize_memory

try:
    from pynndescent import NNDescent
    _HAS_NNDESCENT = True
except ImportError:
    _HAS_NNDESCENT = False

# Deep (but bounded) recursion for large trees/pipelines. Left far below
# sys.maxsize on purpose: a genuinely unbounded limit turns a bug (e.g. an
# accidental reference cycle) into a C-level stack overflow/segfault instead
# of a catchable RecursionError.
DEFAULT_REC_DEPTH = sys.getrecursionlimit()
sys.setrecursionlimit(max(DEFAULT_REC_DEPTH, 10_000))


class ModelNotSupported(Exception):
    """Raised when a value cannot be (de)serialized by the recursive engine."""
    pass


# ---------------------------------------------------------------------------
# Leaf type handlers
# ---------------------------------------------------------------------------

def _dtype_from_str(name):
    """str(np.dtype(...)) round-trips directly for simple dtypes (e.g. 'float64'),
    but for structured/record dtypes it yields a list-of-tuples literal (e.g.
    "[('left_node', '<i8'), ...]") that np.dtype() only accepts pre-parsed."""
    try:
        return np.dtype(name)
    except TypeError:
        return np.dtype(ast.literal_eval(name))


def serialize_numpy_dtype(dtype: np.dtype):
    assert isinstance(dtype, np.dtype)
    return {'meta': 'numpy_dtype', 'name': str(dtype)}


def deserialize_numpy_dtype(model_dict):
    assert model_dict['meta'] == 'numpy_dtype'
    return _dtype_from_str(model_dict['name'])


def serialize_numpy_scalar_type(dtype_type: type):
    assert isinstance(dtype_type, type) and issubclass(dtype_type, np.generic)
    return {'meta': 'numpy_scalar_type', 'name': str(np.dtype(dtype_type))}


def deserialize_numpy_scalar_type(model_dict):
    assert model_dict['meta'] == 'numpy_scalar_type'
    return _dtype_from_str(model_dict['name']).type


def serialize_numpy_scalar(value: np.generic):
    assert isinstance(value, np.generic)
    return {'meta': 'numpy_scalar', 'dtype': str(value.dtype), 'value': value.item()}


def deserialize_numpy_scalar(model_dict):
    assert model_dict['meta'] == 'numpy_scalar'
    return _dtype_from_str(model_dict['dtype']).type(model_dict['value'])


def serialize_numpy_array(array: np.ndarray):
    assert isinstance(array, np.ndarray)
    if array.dtype == object:
        # Elements can be arbitrary Python objects (e.g. an ensemble's
        # estimators_ array of fitted sub-estimators) - .tolist() would leave
        # them as live objects instead of a JSON-safe structure.
        return {'meta': 'numpy_array', 'dtype': 'object', 'shape': list(array.shape),
                'values': [recursive_serialize(value) for value in array.ravel().tolist()]}
    return {'meta': 'numpy_array', 'values': array.tolist(), 'dtype': str(array.dtype)}


def deserialize_numpy_array(model_dict):
    assert model_dict['meta'] == 'numpy_array'
    if model_dict['dtype'] == 'object':
        flat = [recursive_deserialize(value) for value in model_dict['values']]
        array = np.empty(len(flat), dtype=object)
        for i, value in enumerate(flat):
            array[i] = value
        return array.reshape(model_dict['shape'])
    dtype = _dtype_from_str(model_dict['dtype'])
    if dtype.names:
        # Structured/record dtype: .tolist() yields a list of plain tuples,
        # which np.array() needs re-wrapped as an actual tuple per row.
        return np.array([tuple(row) for row in model_dict['values']], dtype=dtype)
    return np.array(model_dict['values'], dtype=dtype)


def serialize_random_generator(generator: np.random.Generator):
    assert isinstance(generator, np.random.Generator)
    return {'meta': 'random_generator', 'bit_generator_type': type(generator.bit_generator).__name__,
           'state': recursive_serialize(generator.bit_generator.state)}


def deserialize_random_generator(model_dict):
    assert model_dict['meta'] == 'random_generator'
    rng = getattr(importlib.import_module('numpy.random'), model_dict['bit_generator_type'])()
    rng.state = recursive_deserialize(model_dict['state'])
    return np.random.Generator(rng)


def serialize_function_reference(func):
    """A plain module-level function used as a parameter value (e.g. sklearn's
    `score_func=f_classif`). Unlike bound methods, these are importable by
    (module, qualname) alone, with no enclosing instance state to capture."""
    assert isinstance(func, (types.FunctionType, types.BuiltinFunctionType))
    return {'meta': 'function_reference', 'module': func.__module__, 'name': func.__qualname__}


def deserialize_function_reference(model_dict):
    assert model_dict['meta'] == 'function_reference'
    obj = importlib.import_module(model_dict['module'])
    for part in model_dict['name'].split('.'):
        obj = getattr(obj, part)
    return obj


def serialize_functools_partial(value: functools.partial):
    # e.g. sklearn's ColumnTransformer builds a FunctionTransformer wrapping
    # functools.partial(check_array, dtype=..., ensure_all_finite=False) for its
    # passthrough branch - reached here via HistGradientBoostingClassifier/
    # Regressor's internal categorical-feature preprocessor. Must be a dedicated
    # leaf rather than falling through to serialize_model_generic's __dict__ walk:
    # a partial's func/args/keywords are stored in read-only C-level slots, not
    # __dict__, so the generic walk would silently capture an empty state (and
    # object.__new__(functools.partial) rejects reconstruction outright anyway).
    assert isinstance(value, functools.partial)
    return {
        'meta': 'functools_partial',
        'func': recursive_serialize(value.func),
        'args': recursive_serialize(list(value.args)),
        'keywords': recursive_serialize(value.keywords),
    }


def deserialize_functools_partial(model_dict):
    assert model_dict['meta'] == 'functools_partial'
    func = recursive_deserialize(model_dict['func'])
    args = recursive_deserialize(model_dict['args'])
    keywords = recursive_deserialize(model_dict['keywords'])
    return functools.partial(func, *args, **keywords)


def serialize_class_reference(cls):
    """A bare class used as a parameter value (e.g. KDTree's dist_metric stores
    the metric *class*, not an instance; a class default like DBSCAN's default
    metric could too). Reconstructed by import path alone, same as a function
    reference - there's no instance state to capture, only the type itself."""
    assert isinstance(cls, type)
    return {'meta': 'class_reference', 'module': cls.__module__, 'name': cls.__qualname__}


def deserialize_class_reference(model_dict):
    assert model_dict['meta'] == 'class_reference'
    obj = importlib.import_module(model_dict['module'])
    for part in model_dict['name'].split('.'):
        obj = getattr(obj, part)
    return obj


def serialize_module_reference(module):
    """A bare module reference stashed on an instance (e.g. scipy's BSpline
    caches the array-namespace module it was built with, in `_xp`/
    `_xp_internal`). Like a class/function reference, importable by name
    alone - and it must be treated as a leaf rather than walked via
    `serialize_model_generic`, since a module's own __dict__ is its entire
    namespace (hundreds of unrelated functions/classes), not instance state."""
    assert isinstance(module, types.ModuleType)
    return {'meta': 'module_reference', 'name': module.__name__}


def deserialize_module_reference(model_dict):
    assert model_dict['meta'] == 'module_reference'
    return importlib.import_module(model_dict['name'])


def serialize_slice(value: slice):
    # e.g. ColumnTransformer's `output_indices_` dict (built by
    # HistGradientBoostingClassifier/Regressor's internal categorical-feature
    # preprocessor) maps each transformer name to a plain `slice` object - not
    # JSON-safe, and not a container/scalar/registered type the generic engine
    # otherwise recognizes.
    assert isinstance(value, slice)
    return {'meta': 'slice', 'start': value.start, 'stop': value.stop, 'step': value.step}


def deserialize_slice(model_dict):
    assert model_dict['meta'] == 'slice'
    return slice(model_dict['start'], model_dict['stop'], model_dict['step'])


# ---------------------------------------------------------------------------
# Generic recursive engine
#
# (The __serialize_leaf_fn__/__deserialize_leaf_fn__ registries are defined at
# the bottom of this file, since a couple of entries - the Cython loss types -
# are only defined further down; recursive_serialize/recursive_deserialize
# only look them up at call time, once the module has finished loading.)
# ---------------------------------------------------------------------------

def recursive_serialize(obj):
    """Serialize an arbitrary Python value into a JSON-safe structure."""
    if obj is None or isinstance(obj, (bool, int, str, float)):
        return obj

    # Bare type/class reference (e.g. np.float64 the class itself, used as a
    # constructor argument such as OneHotEncoder(dtype=np.float64))
    if isinstance(obj, type):
        if issubclass(obj, np.generic):
            return serialize_numpy_scalar_type(obj)
        return serialize_class_reference(obj)

    # Registered leaf types (checked before generic containers: Bunch is a
    # dict subclass and must be handled by its own serializer, not as a dict)
    for obj_type, serialize_fn in __serialize_leaf_fn__:
        if isinstance(obj, obj_type):
            return serialize_fn(obj)

    # Containers
    if isinstance(obj, (list, tuple, set)):
        return {'meta': type(obj).__name__, 'items': [recursive_serialize(item) for item in obj]}
    if isinstance(obj, dict):
        return {'meta': 'dict', 'items': [[recursive_serialize(key), recursive_serialize(value)]
                                          for key, value in obj.items()]}

    # A nested, already-supported ml2json model (e.g. a StandardScaler
    # embedded in some other object's __dict__)
    from .ml2json import serialize_model, ModelNotSupported as _MLNotSupported
    try:
        nested = serialize_model(obj)
        # A repeat visit to an object already seen elsewhere in this same
        # object graph (e.g. Birch's _CFNode.prev_leaf_/next_leaf_, which
        # point at each other) surfaces here as a bare {'meta': 'ref', ...}
        # marker from serialize_model_generic's memo - pass it through as-is
        # rather than wrapping it, since ml2json.deserialize_model has no
        # branch for 'ref' (it isn't a real model type).
        if isinstance(nested, dict) and nested.get('meta') == 'ref':
            return nested
        return {'meta': 'ml2json_model', 'model': nested}
    except _MLNotSupported:
        pass

    # Generic fallback: recurse into the object's own __dict__
    if hasattr(obj, '__dict__'):
        return serialize_model_generic(obj)

    raise ModelNotSupported(f'Cannot serialize object of type {type(obj)}: {obj!r}')


def recursive_deserialize(obj):
    """Reverse of `recursive_serialize`."""
    if obj is None or isinstance(obj, (bool, int, str, float)):
        return obj
    if not isinstance(obj, dict) or 'meta' not in obj:
        raise ModelNotSupported(f'Cannot deserialize malformed payload: {obj!r}')

    meta = obj['meta']

    if meta == 'list':
        return [recursive_deserialize(item) for item in obj['items']]
    if meta == 'tuple':
        return tuple(recursive_deserialize(item) for item in obj['items'])
    if meta == 'set':
        return {recursive_deserialize(item) for item in obj['items']}
    if meta == 'dict':
        return {recursive_deserialize(key): recursive_deserialize(value) for key, value in obj['items']}
    if meta == 'ml2json_model':
        from .ml2json import deserialize_model
        return deserialize_model(obj['model'])
    if meta == 'ref':
        return _deserialize_memo[obj['id']]
    if meta in __deserialize_leaf_fn__:
        return __deserialize_leaf_fn__[meta](obj)
    if meta.startswith('generic_object:'):
        return deserialize_model_generic(obj)

    raise ModelNotSupported(f'Cannot deserialize unknown meta tag: {meta!r}')


# Object-graph reference tracking for serialize_model_generic/deserialize_model_generic.
# Active only for the duration of one top-level call (reset in a finally block),
# so shared/circular references *within* a single object graph (e.g. Birch's
# _CFNode.prev_leaf_/next_leaf_, which point at each other both ways) resolve
# correctly instead of duplicating objects or recursing infinitely. Module-level
# rather than passed as a parameter so it stays active even when recursion
# passes back out through ml2json.serialize_model/deserialize_model's dispatch
# (e.g. a _CFNode nested inside another _CFNode's __dict__, which isn't itself
# a top-level ml2json model and so gets routed through that dispatcher's own
# fallback branch, calling back into this module without knowing about the memo).
_serialize_memo = None
_deserialize_memo = None


def serialize_model_generic(model, meta=None):
    """Serialize any object by recursively walking its __dict__.

    :param model: object to serialize
    :param meta: explicit 'meta' tag to use (kept stable for existing wire
        formats); auto-derived from the object's fully qualified type when
        omitted, e.g. for classes reached only through the fallback path
    """
    global _serialize_memo
    if not hasattr(model, '__dict__'):
        raise ModelNotSupported(f'Cannot serialize object of type {type(model)} (no __dict__): {model!r}')

    is_outermost_call = _serialize_memo is None
    if is_outermost_call:
        _serialize_memo = {}
    try:
        obj_id = id(model)
        if obj_id in _serialize_memo:
            return {'meta': 'ref', 'id': _serialize_memo[obj_id]}
        uid = str(len(_serialize_memo))
        _serialize_memo[obj_id] = uid

        if meta is None:
            meta = f'generic_object:{type(model).__module__}.{type(model).__qualname__}'
        attrs = dict(model.__dict__)
        if isinstance(model, BaseSGD) and '_loss_function_' in attrs:
            # Derived from the `loss` param already in __dict__ and re-created lazily;
            # itself a Cython object with no __dict__, so it can't be serialized directly.
            del attrs['_loss_function_']
        try:
            serialized_attrs = recursive_serialize(attrs)
        except BaseException:
            # The memo entry above was registered before this object's contents
            # were actually serialized. If that walk fails, any other reference
            # to this same object elsewhere in the graph (e.g. via a retried
            # fallback path) must not resolve to a 'ref' pointing at an id that
            # was never actually written out - that produces a dangling
            # reference and a confusing KeyError at deserialize time instead of
            # surfacing this real error.
            del _serialize_memo[obj_id]
            raise
        return {
            'meta': meta,
            'id': uid,
            'module': type(model).__module__,
            'type': type(model).__qualname__,
            'dict': serialized_attrs,
        }
    finally:
        if is_outermost_call:
            _serialize_memo = None


def deserialize_model_generic(model_dict):
    """Reconstruct an object serialized by `serialize_model_generic`.

    Reconstructs via `object.__new__(cls)` and restores the full `__dict__`
    directly rather than calling `__init__`/`set_params`: since well-behaved
    estimators only store constructor arguments as attributes in `__init__`
    (no side effects, per sklearn's API contract), everything `__init__`
    would have done is already captured in the serialized `__dict__`.
    """
    global _deserialize_memo
    is_outermost_call = _deserialize_memo is None
    if is_outermost_call:
        _deserialize_memo = {}
    try:
        cls = importlib.import_module(model_dict['module'])
        for part in model_dict['type'].split('.'):
            cls = getattr(cls, part)
        obj = object.__new__(cls)
        uid = model_dict.get('id')
        if uid is not None:
            # Registered before recursing into 'dict' so a cyclic reference
            # back to this same object (found while deserializing its own
            # attributes) resolves to this shell instead of recursing forever.
            _deserialize_memo[uid] = obj
        obj.__dict__ = recursive_deserialize(model_dict['dict'])
        return obj
    finally:
        if is_outermost_call:
            _deserialize_memo = None


# ---------------------------------------------------------------------------
# Special-case leaf types kept for future use (not wired into the uniform
# registry above: each needs extra context or graph-shaped reconstruction
# that a flat type -> function mapping can't express).
# ---------------------------------------------------------------------------

def serialize_tree(tree: Tree):
    # Fully self-contained: n_features/n_outputs/n_classes are needed to
    # reconstruct the Tree (they're constructor args, not part of
    # __getstate__), but Tree exposes them as its own readable attributes,
    # so no context from a parent model is required.
    assert isinstance(tree, Tree)
    serialized_tree = tree.__getstate__()
    dtypes = [serialized_tree['nodes'].dtype[i].str for i in range(len(serialized_tree['nodes'].dtype))]
    serialized_tree['nodes'] = serialized_tree['nodes'].tolist()
    serialized_tree['values'] = serialized_tree['values'].tolist()
    return {
        'meta': 'tree',
        'tree': serialized_tree,
        'nodes_dtype': dtypes,
        'n_features': int(tree.n_features),
        'n_outputs': int(tree.n_outputs),
        'n_classes': tree.n_classes.tolist(),
    }


def deserialize_tree(tree_dict):
    assert tree_dict['meta'] == 'tree'
    tree_dict['tree']['nodes'] = [tuple(lst) for lst in tree_dict['tree']['nodes']]
    names = ['left_child', 'right_child', 'feature', 'threshold', 'impurity', 'n_node_samples', 'weighted_n_node_samples']
    if sklearn.__version__ >= '1.3':
        names.append('missing_go_to_left')
    tree_dict['tree']['nodes'] = np.array(tree_dict['tree']['nodes'], dtype=np.dtype({'names': names, 'formats': tree_dict['nodes_dtype']}))
    tree_dict['tree']['values'] = np.array(tree_dict['tree']['values'])
    n_classes = np.array(tree_dict['n_classes'], dtype=np.intp)
    tree = Tree(tree_dict['n_features'], n_classes, tree_dict['n_outputs'])
    tree.__setstate__(tree_dict['tree'])
    return tree


def _serialize_binary_tree(tree, meta):
    # Self-contained, like Tree: __getstate__() carries everything except the
    # raw data array needed to construct the shell before __setstate__. KDTree
    # and BallTree (sklearn's two BinaryTree implementations, chosen from each
    # other at fit time based on data/metric) share this exact state shape.
    state = tree.__getstate__()
    return {
        'meta': meta,
        'data': recursive_serialize(np.asarray(tree.data)),
        'data_arr': recursive_serialize(state[0]),
        'idx_data_arr': recursive_serialize(state[1].astype(np.int64)),
        'node_data_arr': recursive_serialize(state[2]),
        'node_bounds_arr': recursive_serialize(state[3]),
        'leaf_size': state[4],
        'n_levels': state[5],
        'n_nodes': state[6],
        'n_trims': state[7],
        'n_leaves': state[8],
        'n_splits': state[9],
        'n_calls': state[10],
        'dist_metric': serialize_class_reference(type(state[11])),
        'sample_weight_arr': recursive_serialize(state[12]) if state[12] is not None else None,
    }


def _deserialize_binary_tree(model_dict, cls):
    tree = cls(recursive_deserialize(model_dict['data']))
    params = (
        recursive_deserialize(model_dict['data_arr']),
        np.array(recursive_deserialize(model_dict['idx_data_arr']), dtype=np.int64),
        recursive_deserialize(model_dict['node_data_arr']),
        recursive_deserialize(model_dict['node_bounds_arr']),
        model_dict['leaf_size'],
        model_dict['n_levels'],
        model_dict['n_nodes'],
        model_dict['n_trims'],
        model_dict['n_leaves'],
        model_dict['n_splits'],
        model_dict['n_calls'],
        deserialize_class_reference(model_dict['dist_metric'])(),
        recursive_deserialize(model_dict['sample_weight_arr']) if model_dict['sample_weight_arr'] is not None else None,
    )
    tree.__setstate__(params)
    return tree


def serialize_tree_predictor(predictor: TreePredictor):
    # HistGradientBoostingClassifier/Regressor's per-iteration trees. Unlike
    # Tree/KDTree, TreePredictor is a plain Python class (init/getstate/dict,
    # no C-level __new__ to fight) - object.__new__(cls) + a restored __dict__
    # would normally be enough. It still needs dedicated handling because two
    # of its three arrays (`binned_left_cat_bitsets`/`raw_left_cat_bitsets`,
    # holding categorical-split bitsets) are almost always empty with shape
    # (0, 8) when the model has no categorical features: serialize_numpy_array's
    # plain .tolist() round trip collapses that to `[]`, and reconstructing via
    # np.array([], dtype=...) silently yields shape (0,) instead of (0, 8) -
    # `predict()` then crashes ("Buffer has wrong number of dimensions") since
    # the Cython prediction loop indexes these bitsets as 2D. `nodes` doesn't
    # have this problem (a tree always has >=1 node) and is serialized the same
    # way as DecisionTreeClassifier's Tree.nodes (see serialize_tree above).
    assert isinstance(predictor, TreePredictor)
    nodes = predictor.nodes
    binned = predictor.binned_left_cat_bitsets
    raw = predictor.raw_left_cat_bitsets
    return {
        'meta': 'hgb_tree_predictor',
        'nodes': nodes.tolist(),
        'nodes_names': list(nodes.dtype.names),
        'nodes_dtype': [nodes.dtype[i].str for i in range(len(nodes.dtype))],
        'binned_left_cat_bitsets': binned.ravel().tolist(),
        'binned_left_cat_bitsets_shape': list(binned.shape),
        'binned_left_cat_bitsets_dtype': str(binned.dtype),
        'raw_left_cat_bitsets': raw.ravel().tolist(),
        'raw_left_cat_bitsets_shape': list(raw.shape),
        'raw_left_cat_bitsets_dtype': str(raw.dtype),
    }


def deserialize_tree_predictor(model_dict):
    assert model_dict['meta'] == 'hgb_tree_predictor'
    nodes_dtype = np.dtype({'names': model_dict['nodes_names'], 'formats': model_dict['nodes_dtype']})
    nodes = np.array([tuple(row) for row in model_dict['nodes']], dtype=nodes_dtype)
    binned = np.array(model_dict['binned_left_cat_bitsets'],
                      dtype=_dtype_from_str(model_dict['binned_left_cat_bitsets_dtype'])
                      ).reshape(model_dict['binned_left_cat_bitsets_shape'])
    raw = np.array(model_dict['raw_left_cat_bitsets'],
                   dtype=_dtype_from_str(model_dict['raw_left_cat_bitsets_dtype'])
                   ).reshape(model_dict['raw_left_cat_bitsets_shape'])
    return TreePredictor(nodes, binned, raw)


def serialize_kdtree(tree: KDTree):
    assert isinstance(tree, KDTree)
    return _serialize_binary_tree(tree, meta='kdtree')


def deserialize_kdtree(model_dict):
    assert model_dict['meta'] == 'kdtree'
    return _deserialize_binary_tree(model_dict, KDTree)


def serialize_balltree(tree: BallTree):
    assert isinstance(tree, BallTree)
    return _serialize_binary_tree(tree, meta='balltree')


def deserialize_balltree(model_dict):
    assert model_dict['meta'] == 'balltree'
    return _deserialize_binary_tree(model_dict, BallTree)


if _HAS_NNDESCENT:
    def serialize_nndescent(model):
        assert isinstance(model, NNDescent)
        state = model.__getstate__()

        # Compiled numba/Cython callables: not serializable, and __setstate__
        # regenerates them all via _set_distance_func()/_init_search_function().
        del state['_distance_func'], state['_tree_search']
        del state['_search_function'], state['_deheap_function']
        del state['_distance_correction']
        state.pop('_rerank_function', None)

        state['_input_dtype'] = np.dtype(state['_input_dtype']).name
        if '_min_distance' in state:
            state['_min_distance'] = float(state['_min_distance'])
        state['_raw_data'] = state['_raw_data'].astype(float).tolist()
        state['rng_state'] = state['rng_state'].astype(int).tolist()
        state['search_rng_state'] = state['search_rng_state'].astype(int).tolist()
        state['_search_graph'] = serialize_csr_matrix(state['_search_graph'])
        state['_visited'] = state['_visited'].astype(int).tolist()
        state['_vertex_order'] = state['_vertex_order'].astype(int).tolist()
        # Absent on a compressed index (e.g. built via PyNNDescentTransformer's
        # default compress_index=True): compress_index() deletes it to save
        # memory once the search graph/forest have been derived from it.
        if '_neighbor_graph' in state:
            state['_neighbor_graph'] = (state['_neighbor_graph'][0].tolist(),
                                        state['_neighbor_graph'][1].astype(float).tolist())
        state['_search_forest'] = ((state['_search_forest'][0][0].astype(float).tolist(),
                                    state['_search_forest'][0][1].astype(float).tolist(),
                                    state['_search_forest'][0][2].astype(int).tolist(),
                                    state['_search_forest'][0][3].astype(int).tolist(),
                                    state['_search_forest'][0][4]),)

        return {'meta': 'nn-descent', 'params': state}


    def deserialize_nndescent(model_dict):
        assert model_dict['meta'] == 'nn-descent'
        params = model_dict['params']

        params['_input_dtype'] = np.dtype(params['_input_dtype']).type
        if '_min_distance' in params:
            params['_min_distance'] = np.float32(params['_min_distance'])
        params['_raw_data'] = np.array(params['_raw_data'], dtype=np.float32)
        params['rng_state'] = np.array(params['rng_state'], dtype=np.int64)
        params['search_rng_state'] = np.array(params['search_rng_state'], dtype=np.int64)
        params['_search_graph'] = deserialize_csr_matrix(params['_search_graph'])
        params['_visited'] = np.array(params['_visited'], dtype=np.uint8)
        params['_vertex_order'] = np.array(params['_vertex_order'], dtype=np.int32)
        if '_neighbor_graph' in params:
            params['_neighbor_graph'] = (np.array(params['_neighbor_graph'][0]),
                                         np.array(params['_neighbor_graph'][1], dtype=np.float32))
        params['_search_forest'] = ((np.array(params['_search_forest'][0][0], dtype=np.float32),
                                     np.array(params['_search_forest'][0][1], dtype=np.float32),
                                     np.array(params['_search_forest'][0][2], dtype=np.int32),
                                     np.array(params['_search_forest'][0][3], dtype=np.int32),
                                     params['_search_forest'][0][4]),)

        model = NNDescent(params['_raw_data'], metric=params['metric'], metric_kwds=params['metric_kwds'])
        params['_distance_func'] = model._distance_func
        params['_distance_correction'] = model._distance_correction
        model.__setstate__(params)
        return model


def serialize_cfnode(model: _CFNode):
    assert isinstance(model, _CFNode)
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


def deserialize_cfnode(model_dict):
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
    model.init_centroids_ = recursive_deserialize(model_dict['init_centroids_'])
    model.init_sq_norm_ = recursive_deserialize(model_dict['init_sq_norm_'])
    model.squared_norm_ = recursive_deserialize(model_dict['squared_norm_']) if isinstance(model_dict['squared_norm_'], dict) else model_dict['squared_norm_']
    # To be linked up by the Birch deserializer
    model.subclusters_ = model_dict['subclusters_']
    model.prev_leaf_ = model_dict['prev_leaf_']
    model.next_leaf_ = model_dict['next_leaf_']
    return model


def serialize_cfsubcluster(model: _CFSubcluster):
    assert isinstance(model, _CFSubcluster)
    mem = lambda x: hex(id(x)) if x is not None else None
    return {
        'meta': 'cfsubcluster',
        'n_samples_': model.n_samples_,
        'squared_sum_': model.squared_sum_,
        'centroid_': model.centroid_.tolist(),
        'linear_sum_': model.linear_sum_.tolist(),
        'sq_norm_': model.sq_norm_,
        'child_': mem(model.child_),
    }


def deserialize_cfsubcluster(model_dict):
    assert model_dict['meta'] == 'cfsubcluster'
    model = _CFSubcluster()
    model.n_samples_ = model_dict['n_samples_']
    model.squared_sum_ = model_dict['squared_sum_']
    model.centroid_ = np.array(model_dict['centroid_'])
    model.linear_sum_ = np.array(model_dict['linear_sum_'])
    model.sq_norm_ = model_dict['sq_norm_']
    # To be linked up by the Birch deserializer
    model.child_ = model_dict['child_']
    return model


def serialize_cyloss(loss):
    assert isinstance(loss, (CyAbsoluteError, CyExponentialLoss, CyHalfBinomialLoss, CyHalfGammaLoss,
                             CyHalfMultinomialLoss, CyHalfPoissonLoss, CyHalfSquaredError, CyHalfTweedieLoss,
                             CyHalfTweedieLossIdentity, CyHuberLoss, CyPinballLoss))
    model_dict = {'meta': 'cython_loss', 'type': type(loss).__name__}
    if isinstance(loss, CyHuberLoss):
        model_dict['delta'] = loss.delta
    if isinstance(loss, (CyHalfTweedieLoss, CyHalfTweedieLossIdentity)):
        model_dict['power'] = loss.power
    if isinstance(loss, CyPinballLoss):
        model_dict['quantile'] = loss.quantile
    return model_dict


def deserialize_cyloss(loss_dict):
    assert loss_dict['meta'] == 'cython_loss'
    loss_type = loss_dict['type']
    if loss_type == 'CyAbsoluteError':
        return CyAbsoluteError()
    if loss_type == 'CyExponentialLoss':
        return CyExponentialLoss()
    if loss_type == 'CyHalfBinomialLoss':
        return CyHalfBinomialLoss()
    if loss_type == 'CyHalfGammaLoss':
        return CyHalfGammaLoss()
    if loss_type == 'CyHalfMultinomialLoss':
        return CyHalfMultinomialLoss()
    if loss_type == 'CyHalfPoissonLoss':
        return CyHalfPoissonLoss()
    if loss_type == 'CyHalfSquaredError':
        return CyHalfSquaredError()
    if loss_type == 'CyHalfTweedieLoss':
        return CyHalfTweedieLoss(loss_dict['power'])
    if loss_type == 'CyHalfTweedieLossIdentity':
        return CyHalfTweedieLossIdentity(loss_dict['power'])
    if loss_type == 'CyHuberLoss':
        return CyHuberLoss(loss_dict['delta'])
    if loss_type == 'CyPinballLoss':
        return CyPinballLoss(loss_dict['quantile'])
    raise ModelNotSupported(f'Unknown Cython loss type: {loss_type!r}')


def serialize_sgd_loss_functions(loss):
    assert any(loss is x for x in (Hinge, SquaredHinge, ModifiedHuber, EpsilonInsensitive, SquaredEpsilonInsensitive))
    return {'meta': 'sgd_loss_functions', 'module': loss.__module__, 'type': loss.__name__}


def deserialize_sgd_loss_functions(loss_dict):
    assert loss_dict['meta'] == 'sgd_loss_functions'
    return getattr(importlib.import_module(loss_dict['module']), loss_dict['type'])


_CYLOSS_TYPES = (CyAbsoluteError, CyExponentialLoss, CyHalfBinomialLoss, CyHalfGammaLoss, CyHalfMultinomialLoss,
                 CyHalfPoissonLoss, CyHalfSquaredError, CyHalfTweedieLoss, CyHalfTweedieLossIdentity, CyHuberLoss,
                 CyPinballLoss)

# Leaf types dispatched by isinstance, checked in order (most specific first).
# Bunch subclasses dict, so it must be checked before recursive_serialize's
# generic dict handling ever gets a chance to see it.
__serialize_leaf_fn__ = [
    (Bunch, serialize_bunch),
    (np.ndarray, serialize_numpy_array),
    (np.dtype, serialize_numpy_dtype),
    (np.generic, serialize_numpy_scalar),
    (RandomState, serialize_random_state),
    (np.random.Generator, serialize_random_generator),
    (Memory, serialize_memory),
    (sp.sparse.csr_matrix, serialize_csr_matrix),
    (_CYLOSS_TYPES, serialize_cyloss),
    ((types.FunctionType, types.BuiltinFunctionType), serialize_function_reference),
    (functools.partial, serialize_functools_partial),
    (types.ModuleType, serialize_module_reference),
    (slice, serialize_slice),
    (Tree, serialize_tree),
    (TreePredictor, serialize_tree_predictor),
    (KDTree, serialize_kdtree),
    (BallTree, serialize_balltree),
]

# Leaf types dispatched by their 'meta' tag on the way back.
__deserialize_leaf_fn__ = {
    'bunch': deserialize_bunch,
    'numpy_array': deserialize_numpy_array,
    'numpy_dtype': deserialize_numpy_dtype,
    'numpy_scalar_type': deserialize_numpy_scalar_type,
    'numpy_scalar': deserialize_numpy_scalar,
    'random_state': deserialize_random_state,
    'random_generator': deserialize_random_generator,
    'memory': deserialize_memory,
    'csr': deserialize_csr_matrix,
    'cython_loss': deserialize_cyloss,
    'tree': deserialize_tree,
    'hgb_tree_predictor': deserialize_tree_predictor,
    'function_reference': deserialize_function_reference,
    'functools_partial': deserialize_functools_partial,
    'class_reference': deserialize_class_reference,
    'module_reference': deserialize_module_reference,
    'slice': deserialize_slice,
    'kdtree': deserialize_kdtree,
    'balltree': deserialize_balltree,
}

if _HAS_NNDESCENT:
    __serialize_leaf_fn__.append((NNDescent, serialize_nndescent))
    __deserialize_leaf_fn__['nn-descent'] = deserialize_nndescent
