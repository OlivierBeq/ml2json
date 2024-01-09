# -*- coding: utf-8 -*-

import inspect
import importlib

import numpy as np

from mlchemad.applicability_domains import (BoundingBoxApplicabilityDomain,
                                            ConvexHullApplicabilityDomain,
                                            PCABoundingBoxApplicabilityDomain,
                                            TopKatApplicabilityDomain,
                                            LeverageApplicabilityDomain,
                                            HotellingT2ApplicabilityDomain,
                                            KernelDensityApplicabilityDomain,
                                            IsolationForestApplicabilityDomain,
                                            CentroidDistanceApplicabilityDomain,
                                            KNNApplicabilityDomain)


def serialize_bounding_box_applicability_domain(model):
    serialized_model = {
        'meta': 'bounding-box-ad',
        'fitted_': model.fitted_,
        'compute_minmax': model.compute_minmax,
        'constant_value_min': model.constant_value_min,
        'constant_value_max': model.constant_value_max,
        'percentiles_min_max': model.percentiles_min_max,
    }
    if model.fitted_:
        serialized_model.update(
            {
            'num_points': model.num_points,
            'num_dims': model.num_dims,
            'min_': model.min_.tolist(),
            'max_': model.max_.tolist(),
            }
        )

    return serialized_model

def deserialize_bounding_box_applicability_domain(model_dict):
    model = BoundingBoxApplicabilityDomain()
    model.fitted_ = model_dict['fitted_']
    model.compute_minmax = model_dict['compute_minmax']
    model.constant_value_min = model_dict['constant_value_min']
    model.constant_value_max = model_dict['constant_value_max']
    model.percentiles_min_max = (tuple(model_dict['percentiles_min_max']) 
                                 if model_dict['percentiles_min_max'] else None)

    if model.fitted_:
        model.num_points = model_dict['num_points']
        model.num_dims = model_dict['num_dims']
        model.min_ = np.array(model_dict['min_'])
        model.max_ = np.array(model_dict['max_'])

    return model

def serialize_convex_hull_applicability_domain(model):
    serialized_model = {
        'meta': 'convex-hull-ad',
        'fitted_': model.fitted_,
    }
    if model.fitted_:
        serialized_model.update(
            {
            'num_points': model.num_points,
            'num_dims': model.num_dims,
            'points': model.points.tolist(),
            }
        )
    
    return serialized_model

def deserialize_convex_hull_applicability_domain(model_dict):
    model = ConvexHullApplicabilityDomain()
    model.fitted_ = model_dict['fitted_']
    
    if model.fitted_:
        model.num_points = model_dict['num_points']
        model.num_dims = model_dict['num_dims']
        model.points = np.array(model_dict['points'])

    return model

def serialize_pca_bounding_box_applicability_domain(model):
    # serialized_model = {
    #     'meta': 'pca-bounding-box-ad',
    # }
    
    # return serialized_model
    return NotImplementedError

def deserialize_pca_bounding_box_applicability_domain(model_dict):
    # model = PCABoundingBoxApplicabilityDomain()

    # return model
    return NotImplementedError

def serialize_topkat_applicability_domain(model):
    # serialized_model = {
    #     'meta': 'topkat-ad',
    # }
    
    # return serialized_model
    return NotImplementedError

def deserialize_topkat_applicability_domain(model_dict):
    # model = TopKatApplicabilityDomain()

    # return model
    return NotImplementedError

def serialize_leverage_applicability_domain(model):
    # serialized_model = {
    #     'meta': 'leverage-ad',
    # }
    
    # return serialized_model
    return NotImplementedError

def deserialize_leverage_applicability_domain(model_dict):
    # model = LeverageApplicabilityDomain()

    # return model
    return NotImplementedError

def serialize_hotelling_t2_applicability_domain(model):
    # serialized_model = {
    #     'meta': 'hotelling-t2-ad',
    # }
    
    # return serialized_model
    return NotImplementedError

def deserialize_hotelling_t2_applicability_domain(model_dict):
    # model = HotellingT2ApplicabilityDomain()

    # return model
    return NotImplementedError

def serialize_kernel_density_applicability_domain(model):
    # serialized_model = {
    #     'meta': 'kernel-density-ad',
    # }
    
    # return serialized_model
    return NotImplementedError

def deserialize_kernel_density_applicability_domain(model_dict):
    # model = KernelDensityApplicabilityDomain()

    # return model
    return NotImplementedError

def serialize_isolation_forest_applicability_domain(model):
    # serialized_model = {
    #     'meta': 'isolation-forest-ad',
    # }
    
    # return serialized_model
    return NotImplementedError

def deserialize_isolation_forest_applicability_domain(model_dict):
    # model = IsolationForestApplicabilityDomain()

    # return model
    return NotImplementedError

def serialize_centroid_distance_applicability_domain(model):
    # serialized_model = {
    #     'meta': 'centroid-distance-ad',
    # }
    
    # return serialized_model
    return NotImplementedError

def deserialize_centroid_distance_applicability_domain(model_dict):
    # model = CentroidDistanceApplicabilityDomain()

    # return model
    return NotImplementedError

def serialize_knn_applicability_domain(model):
    # serialized_model = {
    #     'meta': 'knn-ad',
    # }
    
    # return serialized_model
    return NotImplementedError

def deserialize_knn_applicability_domain(model_dict):
    # model = KNNApplicabilityDomain()

    # return model
    return NotImplementedError

def serialize_standardization_approach_applicability_domain(model):
    # serialized_model = {
    #     'meta': 'standardization-approach-ad',
    # }
    
    # return serialized_model
    return NotImplementedError

def deserialize_standardization_approach_applicability_domain(model_dict):
    # model = StandardizationApproachApplicabilityDomain()

    # return model
    return NotImplementedError