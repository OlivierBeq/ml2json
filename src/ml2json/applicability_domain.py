# -*- coding: utf-8 -*-

import inspect
import importlib
import ml2json

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
    serialized_model = {
        'meta': 'pca-bounding-box-ad',
        'fitted_': model.fitted_,
        'scaler': ml2json.to_dict(model.scaler),
        'min_explained_var': model.min_explained_var,
        'pca': ml2json.to_dict(model.pca),
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

def deserialize_pca_bounding_box_applicability_domain(model_dict):
    model = PCABoundingBoxApplicabilityDomain()
    model.fitted_ = model_dict['fitted_']
    model.scaler = ml2json.from_dict(model_dict['scaler'])
    model.min_explained_var = model_dict['min_explained_var']
    model.pca = ml2json.from_dict(model_dict['pca'])
    
    if model.fitted_:
        model.num_points = model_dict['num_points']
        model.num_dims = model_dict['num_dims']
        model.min_ = np.array(model_dict['min_'])
        model.max_ = np.array(model_dict['max_'])

    return model

def serialize_topkat_applicability_domain(model):
    serialized_model = {
        'meta': 'topkat-ad',
        'fitted_': model.fitted_,
    }
    if model.fitted_:
        serialized_model.update(
            {
            'num_points': model.num_points,
            'num_dims': model.num_dims,
            'X_min_' : model.X_min_.tolist(),
            'X_max_' : model.X_max_.tolist(),
            'eigen_val': model.eigen_val.tolist(),
            'eigen_vec': model.eigen_vec.tolist(),
            'OPS_min_': model.OPS_min_.tolist(),
            'OPS_max_': model.OPS_max_.tolist(),
            }
        )
    
    return serialized_model

def deserialize_topkat_applicability_domain(model_dict):
    model = TopKatApplicabilityDomain()
    model.fitted_ = model_dict['fitted_']
    
    if model.fitted_:
        model.num_points = model_dict['num_points']
        model.num_dims = model_dict['num_dims']
        model.X_min_ = np.array(model_dict['X_min_'])
        model.X_max_ = np.array(model_dict['X_max_'])
        model.eigen_val = np.array(model_dict['eigen_val'])
        model.eigen_vec = np.array(model_dict['eigen_vec'])
        model.OPS_min_ = np.array(model_dict['OPS_min_'])
        model.OPS_max_ = np.array(model_dict['OPS_max_'])

    return model

def serialize_leverage_applicability_domain(model):
    serialized_model = {
        'meta': 'leverage-ad',
        'fitted_': model.fitted_,
        'scaler': ml2json.to_dict(model.scaler),
    }
    
    if model.fitted_:
        serialized_model.update(
            {
            'num_points': model.num_points,
            'num_dims': model.num_dims,
            'var_covar': model.var_covar.tolist(),
            'threshold': model.threshold,
            }
        )
    
    return serialized_model

def deserialize_leverage_applicability_domain(model_dict):
    model = LeverageApplicabilityDomain()
    model.fitted_ = model_dict['fitted_']
    model.scaler = ml2json.from_dict(model_dict['scaler'])
    
    if model.fitted_:
        model.num_points = model_dict['num_points']
        model.num_dims = model_dict['num_dims']
        model.var_covar = np.array(model_dict['var_covar'])
        model.threshold = model_dict['threshold']

    return model

def serialize_hotelling_t2_applicability_domain(model):
    serialized_model = {
        'meta': 'hotelling-t2-ad',
        'fitted_': model.fitted_,
        'alpha': model.alpha,
    }
    
    if model.fitted_:
        serialized_model.update(
            {
            'num_points': model.num_points,
            'num_dims': model.num_dims,
            't2': model.t2.tolist(),
            }
        )
    
    return serialized_model

def deserialize_hotelling_t2_applicability_domain(model_dict):
    model = HotellingT2ApplicabilityDomain()
    model.fitted_ = model_dict['fitted_']
    model.alpha = model_dict['alpha']
    
    if model.fitted_:
        model.num_points = model_dict['num_points']
        model.num_dims = model_dict['num_dims']
        model.t2 = np.array(model_dict['t2'])

    return model

def serialize_kernel_density_applicability_domain(model):
    serialized_model = {
        'meta': 'kernel-density-ad',
        'fitted_': model.fitted_,
        'kde': ml2json.to_dict(model.kde), # TODO: add ml2json.to_dict(model.kde)
        'threshold': model.threshold,
    }
    
    if model.fitted_:
        serialized_model.update(
            {
            'num_points': model.num_points,
            'num_dims': model.num_dims,
            'cutoff': model.cutoff,
            }
        )
    
    return serialized_model

def deserialize_kernel_density_applicability_domain(model_dict):
    model = KernelDensityApplicabilityDomain()
    model.fitted_ = model_dict['fitted_']
    model.kde = ml2json.from_dict(model_dict['kde'])
    model.threshold = model_dict['threshold']
    
    if model.fitted_:
        model.num_points = model_dict['num_points']
        model.num_dims = model_dict['num_dims']
        model.cutoff = model_dict['cutoff']

    return model

def serialize_isolation_forest_applicability_domain(model):
    serialized_model = {
        'meta': 'isolation-forest-ad',
        'fitted_': model.fitted_,
        'isol': ml2json.to_dict(model.isol),
    }
    
    if model.fitted_:
        serialized_model.update(
            {
            'num_points': model.num_points,
            'num_dims': model.num_dims,
            }
        )
    
    return serialized_model

def deserialize_isolation_forest_applicability_domain(model_dict):
    model = IsolationForestApplicabilityDomain()
    model.fitted_ = model_dict['fitted_']
    model.isol = ml2json.from_dict(model_dict['isol'])
    
    if model.fitted_:
        model.num_points = model_dict['num_points']
        model.num_dims = model_dict['num_dims']

    return model

def serialize_centroid_distance_applicability_domain(model):
    serialized_model = {
        'meta': 'centroid-distance-ad',
        'fitted_': model.fitted_,
        'dist': model.dist,
        'scaler': ml2json.to_dict(model.scaler),
        'threshold': model.threshold if model.threshold is not None else None,
    }
    
    if model.fitted_:
        serialized_model.update(
            {
            'num_points': model.num_points,
            'num_dims': model.num_dims,
            'centroid': model.centroid.tolist(),
            }
        )
    
    return serialized_model

def deserialize_centroid_distance_applicability_domain(model_dict):
    model = CentroidDistanceApplicabilityDomain()
    model.fitted_ = model_dict['fitted_']
    model.dist = model_dict['dist']
    model.scaler = ml2json.from_dict(model_dict['scaler'])
    model.threshold = model_dict['threshold']
    
    if model.fitted_:
        model.num_points = model_dict['num_points']
        model.num_dims = model_dict['num_dims']
        model.centroid = np.array(model_dict['centroid'])

    return model

def serialize_knn_applicability_domain(model):
    serialized_model = {
        'meta': 'knn-ad',
        'fitted_': model.fitted_,
        'scaler': ml2json.to_dict(model.scaler),
        'dist': model.dist,
        'k': model.k,
        'alpha': model.alpha,
        'hard_threshold': model.hard_threshold,
    }
    
    if model.fitted_:
        serialized_model.update(
            {
            'num_points': model.num_points,
            'num_dims': model.num_dims,
            'X_norm': model.X_norm.tolist(),
            'kNN_dist': model.kNN_dist.tolist(),
            'threshold_': model.threshold_,
            }
        )
    
    return serialized_model

def deserialize_knn_applicability_domain(model_dict):
    model = KNNApplicabilityDomain()
    model.fitted_ = model_dict['fitted_']
    model.scaler = ml2json.from_dict(model_dict['scaler'])
    model.dist = model_dict['dist']
    model.k = model_dict['k']
    model.alpha = model_dict['alpha']
    model.hard_threshold = model_dict['hard_threshold']
    
    if model.fitted_:
        model.num_points = model_dict['num_points']
        model.num_dims = model_dict['num_dims']
        model.X_norm = np.array(model_dict['X_norm'])
        model.kNN_dist = np.array(model_dict['kNN_dist'])
        model.threshold_ = model_dict['threshold_']

    return model

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