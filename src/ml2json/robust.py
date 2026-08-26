# -*- coding: utf-8 -*-

from . import _base

# Allow additional dependencies to be optional
__optionals__ = []
try:
    from sklearn_extra.robust import (RobustWeightedClassifier, RobustWeightedRegressor,
                                      RobustWeightedKMeans)
    __optionals__.extend(['RobustWeightedClassifier', 'RobustWeightedRegressor', 'RobustWeightedKMeans'])
except:
    pass


if 'RobustWeightedClassifier' in __optionals__:
    def serialize_robust_weighted_classifier(model):
        return _base.serialize_model_generic(model)


    def deserialize_robust_weighted_classifier(model_dict):
        return _base.deserialize_model_generic(model_dict)


if 'RobustWeightedRegressor' in __optionals__:
    def serialize_robust_weighted_regressor(model):
        return _base.serialize_model_generic(model)


    def deserialize_robust_weighted_regressor(model_dict):
        return _base.deserialize_model_generic(model_dict)


if 'RobustWeightedKMeans' in __optionals__:
    def serialize_robust_weighted_kmeans(model):
        return _base.serialize_model_generic(model)


    def deserialize_robust_weighted_kmeans(model_dict):
        return _base.deserialize_model_generic(model_dict)
