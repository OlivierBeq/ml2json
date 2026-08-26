# -*- coding: utf-8 -*-

"""(De)serializers for the native, non-sklearn-estimator objects exposed by the
gradient-boosting libraries: xgboost.Booster, lightgbm.Booster,
lightgbm.Dataset and catboost.Pool.

None of these are fitted sklearn estimators with a plain __dict__ - each is
its own library's C++/Cython-backed object with its own dedicated save/load
API, so none of them can go through `_base.serialize_model_generic`. Each
serializer below instead uses whichever native round-trip mechanism turned
out to be the most faithful/robust for that particular object - see the
docstring on each function for what was tried and why it was (not) chosen.
"""

import base64

import numpy as np
import scipy as sp

from . import _base

# Allow additional dependencies to be optional
__optionals__ = []
try:
    from xgboost import Booster as XGBBooster
    __optionals__.append('XGBBooster')
except:
    pass
try:
    from lightgbm import Booster as LGBMBooster, Dataset as LGBMDataset
    __optionals__.extend(['LGBMBooster', 'LGBMDataset'])
except:
    pass
try:
    from catboost import Pool as CatBoostPool
    __optionals__.append('CatBoostPool')
except:
    pass


if 'XGBBooster' in __optionals__:
    def serialize_xgboost_booster(model: XGBBooster):
        """xgboost's own binary "Universal Binary JSON" format via
        `save_raw(raw_format='ubj')`: a lossless, self-contained (trees +
        config) in-memory byte buffer - no temp file needed, unlike the
        XGBClassifier/XGBRegressor serializers elsewhere in this codebase,
        since Booster exposes save_raw()/load straight to/from bytes. Base64-
        encoded to embed as a JSON string.
        """
        raw = bytes(model.save_raw(raw_format='ubj'))
        return {'raw': base64.b64encode(raw).decode('ascii')}


    def deserialize_xgboost_booster(model_dict):
        raw = base64.b64decode(model_dict['raw'])
        return XGBBooster(model_file=bytearray(raw))


if 'LGBMBooster' in __optionals__:
    def serialize_lightgbm_booster(model: LGBMBooster):
        """lightgbm's own native text serialization - directly JSON-embeddable
        as a plain string, no binary encoding needed. Mirrors how the booster
        embedded inside LGBMClassifier/LGBMRegressor is already handled
        elsewhere in this codebase (model_to_string()/model_str=...).
        """
        return {'model_str': model.model_to_string()}


    def deserialize_lightgbm_booster(model_dict):
        return LGBMBooster(model_str=model_dict['model_str'])


    def serialize_lightgbm_dataset(dataset: LGBMDataset):
        """Reconstructed from the raw feature/label/weight/... arrays rather
        than `Dataset.save_binary()`. `save_binary()` writes LightGBM's
        internal bin-mapped representation - meant for fast reload during
        training continuation, not a faithful copy of the original values:
        empirically, reloading a save_binary() file hands back
        `get_data()` as the *file path string* instead of the array, and
        `categorical_feature` metadata comes back as `'auto'` rather than the
        original explicit indices. The raw-attribute route below is exact.

        Two real constraints of the underlying library apply here:
        - `free_raw_data` defaults to True, which discards `data`/`label`
          once the Dataset is constructed (e.g. after being used in
          `lgb.train()`); there is no way to recover them afterwards, so this
          raises a clear error rather than silently emitting a broken
          Dataset. Build/keep the Dataset with `free_raw_data=False` if it
          needs to be serializable later.
        - a Dataset built with `reference=other_dataset` (e.g. a validation
          set sharing its training set's bin mapping) can't be captured
          through the public API either, so that case is rejected too.
        """
        if dataset.data is None:
            raise ValueError(
                "Cannot serialize this lightgbm.Dataset: its raw data has already been "
                "freed (free_raw_data=True, the default, discards data/label once the "
                "Dataset is constructed/used for training). Build or keep the Dataset "
                "with free_raw_data=False if it needs to be serialized."
            )
        if dataset.reference is not None:
            raise ValueError(
                "Cannot serialize this lightgbm.Dataset: it was built with reference= "
                "another Dataset (e.g. a validation set sharing its training set's bin "
                "mapping), which lightgbm's public API does not expose a way to recover."
            )
        if not isinstance(dataset.data, (np.ndarray, sp.sparse.csr_matrix)):
            raise TypeError(
                f"Cannot serialize a lightgbm.Dataset whose data is a "
                f"{type(dataset.data).__name__}; only numpy arrays and scipy sparse "
                f"(CSR) matrices are currently supported."
            )
        return {
            'data': _base.recursive_serialize(dataset.data),
            'label': _base.recursive_serialize(dataset.label),
            'weight': _base.recursive_serialize(dataset.weight),
            'group': _base.recursive_serialize(dataset.group),
            'init_score': _base.recursive_serialize(dataset.init_score),
            'position': _base.recursive_serialize(dataset.position),
            # Not named 'params': serialize_version()/check_version() special-case
            # a top-level 'params' key as constructor kwargs (to sanitize/restore
            # any live RandomState instance within it), which dataset.params isn't.
            'dataset_params': _base.recursive_serialize(dataset.params),
            'categorical_feature': _base.recursive_serialize(dataset.categorical_feature),
            'feature_name': _base.recursive_serialize(dataset.feature_name),
            'free_raw_data': dataset.free_raw_data,
        }


    def deserialize_lightgbm_dataset(model_dict):
        return LGBMDataset(
            data=_base.recursive_deserialize(model_dict['data']),
            label=_base.recursive_deserialize(model_dict['label']),
            weight=_base.recursive_deserialize(model_dict['weight']),
            group=_base.recursive_deserialize(model_dict['group']),
            init_score=_base.recursive_deserialize(model_dict['init_score']),
            position=_base.recursive_deserialize(model_dict['position']),
            params=_base.recursive_deserialize(model_dict['dataset_params']),
            categorical_feature=_base.recursive_deserialize(model_dict['categorical_feature']),
            feature_name=_base.recursive_deserialize(model_dict['feature_name']),
            free_raw_data=model_dict['free_raw_data'],
        )


if 'CatBoostPool' in __optionals__:
    def serialize_catboost_pool(pool: CatBoostPool):
        """Reconstructed from the raw feature/label/weight/... accessors
        rather than `Pool.save()`. `Pool.save()` refuses to run unless the
        Pool has already been quantized (`CatBoostError: Pool is not
        quantized`) - that native format is CatBoost's own *binned* on-disk
        representation, meant to be reloaded as pre-quantized training input,
        not a faithful copy of the original raw feature values. The
        raw-accessor route below is exact for the case it supports.

        Real constraint of the underlying library: `Pool.get_features()`
        only works for pools with purely numeric feature columns - a pool
        with any categorical/text/embedding feature raises
        `CatBoostError: Pool has non-numeric features, get_features supports
        only numeric features`, and CatBoost's Python API exposes no public
        accessor to recover the original raw values for those columns from
        an already-constructed Pool. Detected proactively here (rather than
        letting that cryptic error surface) via the cat/text/embedding
        feature-index getters, which work regardless of column dtype.
        `group_id`/`subgroup_id`/`pairs`/`timestamp` are similarly not
        exposed by any public getter (only a hash of group_id is, via
        `get_group_id_hash()`) and so are not preserved either.
        """
        if (pool.get_cat_feature_indices() or pool.get_text_feature_indices()
                or pool.get_embedding_feature_indices()):
            raise ValueError(
                "Cannot serialize this catboost.Pool: it has categorical, text and/or "
                "embedding feature columns, and CatBoost's Python API only exposes "
                "get_features() for pools with purely numeric features - there is no "
                "public accessor to recover the original raw values for those columns "
                "from a constructed Pool."
            )
        if pool.get_group_id_hash() is not None:
            raise ValueError(
                "Cannot serialize this catboost.Pool: it has group_id (and/or "
                "group_weight/subgroup_id, which are only ever set alongside group_id) "
                "set, and CatBoost's Python API only exposes a hash of group_id "
                "(get_group_id_hash()), not the original values - there is no public "
                "accessor to recover them from a constructed Pool."
            )
        if pool.num_pairs():
            raise ValueError(
                "Cannot serialize this catboost.Pool: it has pairs set, and CatBoost's "
                "Python API exposes no public accessor to recover them from a "
                "constructed Pool."
            )
        baseline = pool.get_baseline()
        feature_names = pool.get_feature_names()
        return {
            'features': _base.recursive_serialize(pool.get_features()),
            'label': _base.recursive_serialize(pool.get_label()),
            'weight': _base.recursive_serialize(pool.get_weight()),
            'baseline': _base.recursive_serialize(baseline) if len(baseline) else None,
            'feature_names': _base.recursive_serialize(feature_names) if any(feature_names) else None,
        }


    def deserialize_catboost_pool(model_dict):
        return CatBoostPool(
            data=_base.recursive_deserialize(model_dict['features']),
            label=_base.recursive_deserialize(model_dict['label']),
            weight=_base.recursive_deserialize(model_dict['weight']),
            baseline=_base.recursive_deserialize(model_dict['baseline']),
            feature_names=_base.recursive_deserialize(model_dict['feature_names']),
        )
