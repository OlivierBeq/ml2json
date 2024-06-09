# -*- coding: utf-8 -*-

import importlib
import inspect

import numpy as np
import sklearn
from sklearn.cluster import (AffinityPropagation, AgglomerativeClustering,
                             Birch, DBSCAN, FeatureAgglomeration, KMeans,
                             BisectingKMeans, MiniBatchKMeans, MeanShift, OPTICS,
                             SpectralClustering, SpectralBiclustering, SpectralCoclustering)
from sklearn.cluster._birch import _CFNode, _CFSubcluster
from sklearn.cluster._bisect_k_means import _BisectingTree

# Allow additional dependencies to be optional
__optionals__ = []


try:
    from imblearn.under_sampling import (ClusterCentroids, CondensedNearestNeighbour, EditedNearestNeighbours,
                                         RepeatedEditedNearestNeighbours, AllKNN, InstanceHardnessThreshold,
                                         NearMiss, NeighbourhoodCleaningRule, OneSidedSelection,
                                         RandomUnderSampler, TomekLinks)
    from imblearn.over_sampling import (RandomOverSampler, SMOTE, SMOTEN, SMOTENC, ADASYN, BorderlineSMOTE,
                                        KMeansSMOTE, SVMSMOTE)
    from imblearn.combine import SMOTEENN, SMOTETomek
    from imblearn.ensemble import (EasyEnsembleClassifier, RUSBoostClassifier, BalancedBaggingClassifier,
                                   BalancedRandomForestClassifier)
    __optionals__.extend(['imblearn'])
except:
    pass

from .utils.random_state import serialize_random_state, deserialize_random_state
from .utils.memory import serialize_memory, deserialize_memory


if 'imblearn' in __optionals__:
    def serialize_cluster_centroids(model):
        from ml2json import serialize_model

        serialized_model = {'meta': 'cluster-centroids',
                            'params': {param: value
                                       for param, value in model.get_params().items()
                                       if not param.startswith('estimator')}
                            }

        serialized_model['params']['estimator'] = serialize_model(model.estimator)

        if 'n_features_in_' in model.__dict__:
            serialized_model['n_features_in_'] = model.n_features_in_
        if 'sampling_strategy_' in model.__dict__:
            serialized_model['sampling_strategy_'] = str(model.sampling_strategy_)
        if 'estimator_' in model.__dict__:
            serialized_model['estimator_'] = serialize_model(model.estimator_)
        if 'voting_' in model.__dict__:
            serialized_model['voting_'] = model.voting_

        return serialized_model


    def deserialize_cluster_centroids(model_dict):
        from collections import OrderedDict
        from ml2json import deserialize_model

        model_dict['params']['estimator'] = deserialize_model(model_dict['params']['estimator'])

        model = ClusterCentroids(**model_dict['params'])

        if 'n_features_in_' in model_dict.keys():
            model.n_features_in_ = model_dict['n_features_in_']
        if 'sampling_strategy_' in model_dict.keys():
            model.sampling_strategy_ = eval(model_dict['sampling_strategy_'])
        if 'estimator_' in model_dict.keys():
            model.estimator_ = deserialize_model(model_dict['estimator_'])
        if 'voting_' in model_dict.keys():
            model.voting_ = model_dict['voting_']

        return model


    def serialize_condensed_nearest_neighbours(model):
        from ml2json import serialize_model

        serialized_model = {'meta': 'condensed-nearest-neighbours',
                            'params': model.get_params()
                            }

        if not isinstance(model.n_neighbors, int) and model.n_neighbors is not None:
            serialized_model['params']['n_neighbors'] = serialize_model(model.n_neighbors)

        if 'n_features_in_' in model.__dict__:
            serialized_model['n_features_in_'] = model.n_features_in_
        if 'sampling_strategy_' in model.__dict__:
            serialized_model['sampling_strategy_'] = str(model.sampling_strategy_)
        if 'estimator_' in model.__dict__:
            serialized_model['estimator_'] = serialize_model(model.estimator_)
        if 'estimators_' in model.__dict__:
            serialized_model['estimators_'] = [serialize_model(estimator_) for estimator_ in model.estimators_]
        if 'sample_indices_' in model.__dict__:
            serialized_model['sample_indices_'] = model.sample_indices_.tolist()

        return serialized_model


    def deserialize_condensed_nearest_neighbours(model_dict):
        from collections import OrderedDict
        from ml2json import deserialize_model

        if not isinstance(model_dict['params']['n_neighbors'], int) and model_dict['params']['n_neighbors'] is not None:
            model_dict['params']['n_neighbors'] = deserialize_model(model_dict['params']['n_neighbors'])

        model = CondensedNearestNeighbour(**model_dict['params'])

        if 'n_features_in_' in model_dict.keys():
            model.n_features_in_ = model_dict['n_features_in_']
        if 'sampling_strategy_' in model_dict.keys():
            model.sampling_strategy_ = eval(model_dict['sampling_strategy_'])
        if 'estimator_' in model_dict.keys():
            model.estimator_ = deserialize_model(model_dict['estimator_'])
        if 'estimators_' in model_dict.keys():
            model.estimators_ = [deserialize_model(estimator_) for estimator_ in model_dict['estimators_']]
        if 'sample_indices_' in model_dict.keys():
            model.sample_indices_ = np.array(model_dict['sample_indices_'])

        return model


    def serialize_edited_nearest_neighbours(model):
        from ml2json import serialize_model

        serialized_model = {'meta': 'edited-nearest-neighbours',
                            'params': {param: value
                                       for param, value in model.get_params().items()
                                       if not param.startswith('n_neighbors__')}
                            }

        if not isinstance(model.n_neighbors, int):
            serialized_model['params']['n_neighbors'] = serialize_model(model.n_neighbors)

        if 'n_features_in_' in model.__dict__:
            serialized_model['n_features_in_'] = model.n_features_in_
        if 'feature_names_in_' in model.__dict__:
            serialized_model['feature_names_in_'] = model.feature_names_in_
        if 'sampling_strategy_' in model.__dict__:
            serialized_model['sampling_strategy_'] = str(model.sampling_strategy_)
        if 'nn_' in model.__dict__:
            serialized_model['nn_'] = serialize_model(model.nn_)
        if 'sample_indices_' in model.__dict__:
            serialized_model['sample_indices_'] = model.sample_indices_.tolist()

        return serialized_model


    def deserialize_edited_nearest_neighbours(model_dict):
        from collections import OrderedDict
        from ml2json import deserialize_model

        if not isinstance(model_dict['params']['n_neighbors'], int):
            model_dict['params']['n_neighbors'] = deserialize_model(model_dict['params']['n_neighbors'])

        model = EditedNearestNeighbours(**model_dict['params'])

        if 'n_features_in_' in model_dict.keys():
            model.n_features_in_ = model_dict['n_features_in_']
        if 'feature_names_in_' in model_dict.keys():
            model.feature_names_in_ = model_dict['feature_names_in_']
        if 'sampling_strategy_' in model_dict.keys():
            model.sampling_strategy_ = eval(model_dict['sampling_strategy_'])
        if 'nn_' in model_dict.keys():
            model.nn_ = deserialize_model(model_dict['nn_'])
        if 'sample_indices_' in model_dict.keys():
            model.sample_indices_ = np.array(model_dict['sample_indices_'])

        return model


    def serialize_repeated_edited_nearest_neighbours(model):
        from ml2json import serialize_model

        serialized_model = {'meta': 'repeated-edited-nearest-neighbours',
                            'params': model.get_params()
                            }

        if not isinstance(model.n_neighbors, int):
            serialized_model['params']['n_neighbors'] = serialize_model(model.n_neighbors)

        if 'n_features_in_' in model.__dict__:
            serialized_model['n_features_in_'] = model.n_features_in_
        if 'n_iter_' in model.__dict__:
            serialized_model['n_iter_'] = model.n_iter_
        if 'feature_names_in_' in model.__dict__:
            serialized_model['feature_names_in_'] = model.feature_names_in_
        if 'sampling_strategy_' in model.__dict__:
            serialized_model['sampling_strategy_'] = str(model.sampling_strategy_)
        if 'nn_' in model.__dict__:
            serialized_model['nn_'] = serialize_model(model.nn_)
        if 'enn_' in model.__dict__:
            serialized_model['enn_'] = serialize_edited_nearest_neighbours(model.enn_)
        if 'sample_indices_' in model.__dict__:
            serialized_model['sample_indices_'] = model.sample_indices_.tolist()

        return serialized_model


    def deserialize_repeated_edited_nearest_neighbours(model_dict):
        from collections import OrderedDict
        from ml2json import deserialize_model

        if not isinstance(model_dict['params']['n_neighbors'], int):
            model_dict['params']['n_neighbors'] = deserialize_model(model_dict['params']['n_neighbors'])

        model = RepeatedEditedNearestNeighbours(**model_dict['params'])

        if 'n_features_in_' in model_dict.keys():
            model.n_features_in_ = model_dict['n_features_in_']
        if 'feature_names_in_' in model_dict.keys():
            model.feature_names_in_ = model_dict['feature_names_in_']
        if 'n_iter_' in model_dict.keys():
            model.n_iter_ = model_dict['n_iter_']
        if 'sampling_strategy_' in model_dict.keys():
            model.sampling_strategy_ = eval(model_dict['sampling_strategy_'])
        if 'nn_' in model_dict.keys():
            model.nn_ = deserialize_model(model_dict['nn_'])
        if 'enn_' in model_dict.keys():
            model.enn_ = deserialize_edited_nearest_neighbours(model_dict['enn_'])
        if 'sample_indices_' in model_dict.keys():
            model.sample_indices_ = np.array(model_dict['sample_indices_'])

        return model


    def serialize_all_knn(model):
        from ml2json import serialize_model

        serialized_model = {'meta': 'all-knn',
                            'params': model.get_params()
                            }

        if not isinstance(model.n_neighbors, int):
            serialized_model['params']['n_neighbors'] = serialize_model(model.n_neighbors)

        if 'n_features_in_' in model.__dict__:
            serialized_model['n_features_in_'] = model.n_features_in_
        if 'feature_names_in_' in model.__dict__:
            serialized_model['feature_names_in_'] = model.feature_names_in_
        if 'sampling_strategy_' in model.__dict__:
            serialized_model['sampling_strategy_'] = str(model.sampling_strategy_)
        if 'nn_' in model.__dict__:
            serialized_model['nn_'] = serialize_model(model.nn_)
        if 'enn_' in model.__dict__:
            serialized_model['enn_'] = serialize_edited_nearest_neighbours(model.enn_)
        if 'sample_indices_' in model.__dict__:
            serialized_model['sample_indices_'] = model.sample_indices_.tolist()

        return serialized_model


    def deserialize_all_knn(model_dict):
        from collections import OrderedDict
        from ml2json import deserialize_model

        if not isinstance(model_dict['params']['n_neighbors'], int):
            model_dict['params']['n_neighbors'] = deserialize_model(model_dict['params']['n_neighbors'])

        model = AllKNN(**model_dict['params'])

        if 'n_features_in_' in model_dict.keys():
            model.n_features_in_ = model_dict['n_features_in_']
        if 'feature_names_in_' in model_dict.keys():
            model.feature_names_in_ = model_dict['feature_names_in_']
        if 'sampling_strategy_' in model_dict.keys():
            model.sampling_strategy_ = eval(model_dict['sampling_strategy_'])
        if 'nn_' in model_dict.keys():
            model.nn_ = deserialize_model(model_dict['nn_'])
        if 'enn_' in model_dict.keys():
            model.enn_ = deserialize_edited_nearest_neighbours(model_dict['enn_'])
        if 'sample_indices_' in model_dict.keys():
            model.sample_indices_ = np.array(model_dict['sample_indices_'])

        return model


    def serialize_instance_hardness_threshold(model):
        from ml2json import serialize_model

        serialized_model = {'meta': 'instance-hardness-threshold',
                            'params': model.get_params()
                            }

        if serialized_model['params']['estimator'] is not None:
            serialized_model['params']['estimator'] = serialize_model(model.estimator)

        if 'n_features_in_' in model.__dict__:
            serialized_model['n_features_in_'] = model.n_features_in_
        if 'feature_names_in_' in model.__dict__:
            serialized_model['feature_names_in_'] = model.feature_names_in_
        if 'sampling_strategy_' in model.__dict__:
            serialized_model['sampling_strategy_'] = str(model.sampling_strategy_)
        if 'estimator_' in model.__dict__:
            serialized_model['estimator_'] = serialize_model(model.estimator_)
        if 'sample_indices_' in model.__dict__:
            serialized_model['sample_indices_'] = model.sample_indices_.tolist()

        return serialized_model


    def deserialize_instance_hardness_threshold(model_dict):
        from collections import OrderedDict
        from ml2json import deserialize_model

        if model_dict['params']['estimator'] is not None:
            model_dict['params']['estimator'] = deserialize_model(model_dict['params']['estimator'])

        model = InstanceHardnessThreshold(**model_dict['params'])

        if 'n_features_in_' in model_dict.keys():
            model.n_features_in_ = model_dict['n_features_in_']
        if 'feature_names_in_' in model_dict.keys():
            model.feature_names_in_ = model_dict['feature_names_in_']
        if 'sampling_strategy_' in model_dict.keys():
            model.sampling_strategy_ = eval(model_dict['sampling_strategy_'])
        if 'estimator_' in model_dict.keys():
            model.estimator_ = deserialize_model(model_dict['estimator_'])
        if 'sample_indices_' in model_dict.keys():
            model.sample_indices_ = np.array(model_dict['sample_indices_'])

        return model


    def serialize_near_miss(model):
        from ml2json import serialize_model

        serialized_model = {'meta': 'near-miss',
                            'params': model.get_params()
                            }

        if not isinstance(model.n_neighbors, int):
            serialized_model['params']['n_neighbors'] = serialize_model(model.n_neighbors)
        if not isinstance(model.n_neighbors_ver3, int):
            serialized_model['params']['n_neighbors_ver3'] = serialize_model(model.n_neighbors_ver3)

        if 'n_features_in_' in model.__dict__:
            serialized_model['n_features_in_'] = model.n_features_in_
        if 'feature_names_in_' in model.__dict__:
            serialized_model['feature_names_in_'] = model.feature_names_in_
        if 'sampling_strategy_' in model.__dict__:
            serialized_model['sampling_strategy_'] = str(model.sampling_strategy_)
        if 'nn_' in model.__dict__:
            serialized_model['nn_'] = serialize_model(model.nn_)
        if 'sample_indices_' in model.__dict__:
            serialized_model['sample_indices_'] = model.sample_indices_.tolist()

        return serialized_model


    def deserialize_near_miss(model_dict):
        from collections import OrderedDict
        from ml2json import deserialize_model

        if not isinstance(model_dict['params']['n_neighbors'], int):
            model_dict['params']['n_neighbors'] = deserialize_model(model_dict['params']['n_neighbors'])
        if not isinstance(model_dict['params']['n_neighbors_ver3'], int):
            model_dict['params']['n_neighbors_ver3'] = deserialize_model(model_dict['params']['n_neighbors_ver3'])

        model = NearMiss(**model_dict['params'])

        if 'n_features_in_' in model_dict.keys():
            model.n_features_in_ = model_dict['n_features_in_']
        if 'feature_names_in_' in model_dict.keys():
            model.feature_names_in_ = model_dict['feature_names_in_']
        if 'sampling_strategy_' in model_dict.keys():
            model.sampling_strategy_ = eval(model_dict['sampling_strategy_'])
        if 'nn_' in model_dict.keys():
            model.nn_ = deserialize_model(model_dict['nn_'])
        if 'sample_indices_' in model_dict.keys():
            model.sample_indices_ = np.array(model_dict['sample_indices_'])

        return model


    def serialize_neighbourhood_cleaning_rule(model):
        from ml2json import serialize_model

        serialized_model = {'meta': 'neighbourhood-cleaning-rule',
                            'params': model.get_params()
                            }

        if model.edited_nearest_neighbours is not None:
            serialized_model['params']['edited_nearest_neighbours'] = serialize_model(model.edited_nearest_neighbours)
        if not isinstance(model.n_neighbors, int):
            serialized_model['params']['n_neighbors'] = serialize_model(model.n_neighbors)

        if 'n_features_in_' in model.__dict__:
            serialized_model['n_features_in_'] = model.n_features_in_
        if 'feature_names_in_' in model.__dict__:
            serialized_model['feature_names_in_'] = model.feature_names_in_
        if 'sampling_strategy_' in model.__dict__:
            serialized_model['sampling_strategy_'] = str(model.sampling_strategy_)
        if 'nn_' in model.__dict__:
            serialized_model['nn_'] = serialize_model(model.nn_)
        if 'edited_nearest_neighbours_' in model.__dict__:
            serialized_model['edited_nearest_neighbours_'] = serialize_edited_nearest_neighbours(model.edited_nearest_neighbours_)
        if 'sample_indices_' in model.__dict__:
            serialized_model['sample_indices_'] = model.sample_indices_.tolist()
        if 'classes_to_clean_' in model.__dict__:
            serialized_model['classes_to_clean_'] = [int(x) for x in model.classes_to_clean_]

        return serialized_model


    def deserialize_neighbourhood_cleaning_rule(model_dict):
        from collections import OrderedDict
        from ml2json import deserialize_model

        if not isinstance(model_dict['params']['n_neighbors'], int):
            model_dict['params']['n_neighbors'] = deserialize_model(model_dict['params']['n_neighbors'])
        if model_dict['params']['edited_nearest_neighbours'] is not None:
            model_dict['params']['edited_nearest_neighbours'] = deserialize_model(model_dict['params']['edited_nearest_neighbours'])

        model = NeighbourhoodCleaningRule(**model_dict['params'])

        if 'n_features_in_' in model_dict.keys():
            model.n_features_in_ = model_dict['n_features_in_']
        if 'feature_names_in_' in model_dict.keys():
            model.feature_names_in_ = model_dict['feature_names_in_']
        if 'sampling_strategy_' in model_dict.keys():
            model.sampling_strategy_ = eval(model_dict['sampling_strategy_'])
        if 'nn_' in model_dict.keys():
            model.nn_ = deserialize_model(model_dict['nn_'])
        if 'edited_nearest_neighbours_' in model_dict.keys():
            model.edited_nearest_neighbours_ = deserialize_edited_nearest_neighbours(model_dict['edited_nearest_neighbours_'])
        if 'sample_indices_' in model_dict.keys():
            model.sample_indices_ = np.array(model_dict['sample_indices_'])
        if 'classes_to_clean_' in model_dict.keys():
            model.classes_to_clean_ = model_dict['classes_to_clean_']

        return model


    def serialize_one_sided_selection(model):
        from ml2json import serialize_model

        serialized_model = {'meta': 'one-sided-selection',
                            'params': model.get_params()
                            }

        if not isinstance(model.n_neighbors, int) and model.n_neighbors is not None:
            serialized_model['params']['n_neighbors'] = serialize_model(model.n_neighbors)

        if 'n_features_in_' in model.__dict__:
            serialized_model['n_features_in_'] = model.n_features_in_
        if 'feature_names_in_' in model.__dict__:
            serialized_model['feature_names_in_'] = model.feature_names_in_
        if 'sampling_strategy_' in model.__dict__:
            serialized_model['sampling_strategy_'] = str(model.sampling_strategy_)
        if 'estimator_' in model.__dict__:
            serialized_model['estimator_'] = serialize_model(model.estimator_)
        if 'estimators_' in model.__dict__:
            serialized_model['estimators_'] = [serialize_model(estimator_) for estimator_ in model.estimators_]
        if 'sample_indices_' in model.__dict__:
            serialized_model['sample_indices_'] = model.sample_indices_.tolist()

        return serialized_model


    def deserialize_one_sided_selection(model_dict):
        from collections import OrderedDict
        from ml2json import deserialize_model

        if not isinstance(model_dict['params']['n_neighbors'], int) and model_dict['params']['n_neighbors'] is not None:
            model_dict['params']['n_neighbors'] = deserialize_model(model_dict['params']['n_neighbors'])

        model = OneSidedSelection(**model_dict['params'])

        if 'n_features_in_' in model_dict.keys():
            model.n_features_in_ = model_dict['n_features_in_']
        if 'feature_names_in_' in model_dict.keys():
            model.feature_names_in_ = model_dict['feature_names_in_']
        if 'sampling_strategy_' in model_dict.keys():
            model.sampling_strategy_ = eval(model_dict['sampling_strategy_'])
        if 'estimator_' in model_dict.keys():
            model.estimator_ = deserialize_model(model_dict['estimator_'])
        if 'estimators_' in model_dict.keys():
            model.estimators_ = [deserialize_model(estimator_) for estimator_ in model_dict['estimators_']]
        if 'sample_indices_' in model_dict.keys():
            model.sample_indices_ = np.array(model_dict['sample_indices_'])

        return model


    def serialize_random_under_sampler(model):
        from ml2json import serialize_model

        serialized_model = {'meta': 'random-under-sampler',
                            'params': model.get_params()
                            }

        if 'n_features_in_' in model.__dict__:
            serialized_model['n_features_in_'] = model.n_features_in_
        if 'feature_names_in_' in model.__dict__:
            serialized_model['feature_names_in_'] = model.feature_names_in_
        if 'sampling_strategy_' in model.__dict__:
            serialized_model['sampling_strategy_'] = str(model.sampling_strategy_)
        if 'sample_indices_' in model.__dict__:
            serialized_model['sample_indices_'] = model.sample_indices_.tolist()

        return serialized_model


    def deserialize_random_under_sampler(model_dict):
        from collections import OrderedDict

        model = RandomUnderSampler(**model_dict['params'])

        if 'n_features_in_' in model_dict.keys():
            model.n_features_in_ = model_dict['n_features_in_']
        if 'feature_names_in_' in model_dict.keys():
            model.feature_names_in_ = model_dict['feature_names_in_']
        if 'sampling_strategy_' in model_dict.keys():
            model.sampling_strategy_ = eval(model_dict['sampling_strategy_'])
        if 'sample_indices_' in model_dict.keys():
            model.sample_indices_ = np.array(model_dict['sample_indices_'])

        return model


    def serialize_tomek_links(model):
        from ml2json import serialize_model

        serialized_model = {'meta': 'tomek-links',
                            'params': model.get_params()
                            }

        if 'n_features_in_' in model.__dict__:
            serialized_model['n_features_in_'] = model.n_features_in_
        if 'feature_names_in_' in model.__dict__:
            serialized_model['feature_names_in_'] = model.feature_names_in_
        if 'sampling_strategy_' in model.__dict__:
            serialized_model['sampling_strategy_'] = str(model.sampling_strategy_)
        if 'sample_indices_' in model.__dict__:
            serialized_model['sample_indices_'] = model.sample_indices_.tolist()

        return serialized_model


    def deserialize_tomek_links(model_dict):
        from collections import OrderedDict

        model = TomekLinks(**model_dict['params'])

        if 'n_features_in_' in model_dict.keys():
            model.n_features_in_ = model_dict['n_features_in_']
        if 'feature_names_in_' in model_dict.keys():
            model.feature_names_in_ = model_dict['feature_names_in_']
        if 'sampling_strategy_' in model_dict.keys():
            model.sampling_strategy_ = eval(model_dict['sampling_strategy_'])
        if 'sample_indices_' in model_dict.keys():
            model.sample_indices_ = np.array(model_dict['sample_indices_'])

        return model
