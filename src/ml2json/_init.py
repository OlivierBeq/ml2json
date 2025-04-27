# -*- coding: utf-8 -*-

from ._base import *
from . import regression as reg

if 'CatBoostRegressor' in reg.__optionals__:
    from catboost import CatBoostRegressor, CatBoostRanker, Pool, CatBoostClassifier


__version__ = '1.0.0'


def serialize_model(model, catboost_data: Pool = None) -> dict[str, Any]:
    """Serialize a model into a dictionary.

    :param model: machine learning model to be serialized
    :param catboost_data: if `model` is a CatBoost model, the data `Pool` used to train it
    """
    # Verify model is fit
    if not is_model_fitted(model):
        return serialize_unfitted_model(model)

    # Classification
    if isinstance(model, LogisticRegression):
        model_dict = clf.serialize_logistic_regression(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, BernoulliNB):
        model_dict = clf.serialize_bernoulli_nb(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, GaussianNB):
        model_dict = clf.serialize_gaussian_nb(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, MultinomialNB):
        model_dict = clf.serialize_multinomial_nb(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, ComplementNB):
        model_dict = clf.serialize_complement_nb(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, discriminant_analysis.LinearDiscriminantAnalysis):
        model_dict = clf.serialize_lda(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, discriminant_analysis.QuadraticDiscriminantAnalysis):
        model_dict = clf.serialize_qda(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, svm.SVC):
        model_dict = clf.serialize_svm(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, Perceptron):
        model_dict = clf.serialize_perceptron(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, DecisionTreeClassifier):
        model_dict = clf.serialize_decision_tree(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, GradientBoostingClassifier):
        model_dict = clf.serialize_gradient_boosting(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, RandomForestClassifier):
        model_dict = clf.serialize_random_forest(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, MLPClassifier):
        model_dict = clf.serialize_mlp(model)
        return serialize_version(model, model_dict)
    elif 'XGBClassifier' in clf.__optionals__ and isinstance(model, XGBClassifier):
        model_dict = clf.serialize_xgboost_classifier(model)
        return serialize_version(model, model_dict)
    elif 'XGBRFClassifier' in clf.__optionals__ and isinstance(model, XGBRFClassifier):
        model_dict = clf.serialize_xgboost_rf_classifier(model)
        return serialize_version(model, model_dict)
    elif 'LGBMClassifier' in clf.__optionals__ and isinstance(model, LGBMClassifier):
        model_dict = clf.serialize_lightgbm_classifier(model)
        return serialize_version(model, model_dict)
    elif 'CatBoostClassifier' in clf.__optionals__ and isinstance(model, CatBoostClassifier):
        model_dict = clf.serialize_catboost_classifier(model, catboost_data)
        return serialize_version(model, model_dict)
    elif isinstance(model, AdaBoostClassifier):
        model_dict = clf.serialize_adaboost_classifier(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, BaggingClassifier):
        model_dict = clf.serialize_bagging_classifier(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, ExtraTreeClassifier):
        model_dict = clf.serialize_extra_tree_classifier(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, ExtraTreesClassifier):
        model_dict = clf.serialize_extratrees_classifier(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, IsolationForest):
        model_dict = clf.serialize_isolation_forest(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, RandomTreesEmbedding):
        model_dict = clf.serialize_random_trees_embedding(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, KNeighborsClassifier):
        model_dict = clf.serialize_nearest_neighbour_classifier(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, StackingClassifier):
        model_dict = clf.serialize_stacking_classifier(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, VotingClassifier):
        model_dict = clf.serialize_voting_classifier(model)
        return serialize_version(model, model_dict)

    # Regression
    elif isinstance(model, LinearRegression):
        model_dict = reg.serialize_linear_regressor(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, Lasso):
        model_dict = reg.serialize_lasso_regressor(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, ElasticNet):
        model_dict = reg.serialize_elastic_regressor(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, Ridge):
        model_dict = reg.serialize_ridge_regressor(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, SVR):
        model_dict = reg.serialize_svr(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, ExtraTreeRegressor):
        model_dict = reg.serialize_extra_tree_regressor(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, DecisionTreeRegressor):
        model_dict = reg.serialize_decision_tree_regressor(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, GradientBoostingRegressor):
        model_dict = reg.serialize_gradient_boosting_regressor(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, RandomForestRegressor):
        model_dict = reg.serialize_random_forest_regressor(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, ExtraTreesRegressor):
        model_dict = reg.serialize_extratrees_regressor(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, MLPRegressor):
        model_dict = reg.serialize_mlp_regressor(model)
        return serialize_version(model, model_dict)
    elif 'XGBRanker' in reg.__optionals__ and isinstance(model, XGBRanker):
        model_dict = reg.serialize_xgboost_ranker(model)
        return serialize_version(model, model_dict)
    elif 'XGBRegressor' in reg.__optionals__ and isinstance(model, XGBRegressor):
        model_dict = reg.serialize_xgboost_regressor(model)
        return serialize_version(model, model_dict)
    elif 'XGBRFRegressor' in reg.__optionals__ and isinstance(model, XGBRFRegressor):
        model_dict = reg.serialize_xgboost_rf_regressor(model)
        return serialize_version(model, model_dict)
    elif 'LGBMRegressor' in reg.__optionals__ and isinstance(model, LGBMRegressor):
        model_dict = reg.serialize_lightgbm_regressor(model)
        return serialize_version(model, model_dict)
    elif 'LGBMRanker' in reg.__optionals__ and isinstance(model, LGBMRanker):
        model_dict = reg.serialize_lightgbm_ranker(model)
        return serialize_version(model, model_dict)
    elif 'CatBoostRegressor' in reg.__optionals__ and isinstance(model, CatBoostRegressor):
        model_dict = reg.serialize_catboost_regressor(model, catboost_data)
        return serialize_version(model, model_dict)
    elif 'CatBoostRanker' in reg.__optionals__ and isinstance(model, CatBoostRanker):
        model_dict = reg.serialize_catboost_ranker(model, catboost_data)
        return serialize_version(model, model_dict)
    elif isinstance(model, AdaBoostRegressor):
        model_dict = reg.serialize_adaboost_regressor(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, BaggingRegressor):
        model_dict = reg.serialize_bagging_regressor(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, KNeighborsRegressor):
        model_dict = reg.serialize_nearest_neighbour_regressor(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, StackingRegressor):
        model_dict = reg.serialize_stacking_regressor(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, VotingRegressor):
        model_dict = reg.serialize_voting_regressor(model)
        return serialize_version(model, model_dict)

    # Clustering
    elif isinstance(model, FeatureAgglomeration):
        model_dict = clus.serialize_feature_agglomeration(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, AffinityPropagation):
        model_dict = clus.serialize_affinity_propagation(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, AgglomerativeClustering):
        model_dict = clus.serialize_agglomerative_clustering(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, DBSCAN):
        model_dict = clus.serialize_dbscan(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, MeanShift):
        model_dict = clus.serialize_meanshift(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, BisectingKMeans):
        model_dict = clus.serialize_bisecting_kmeans(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, MiniBatchKMeans):
        model_dict = clus.serialize_minibatch_kmeans(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, KMeans):
        model_dict = clus.serialize_kmeans(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, OPTICS):
        model_dict = clus.serialize_optics(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, SpectralClustering):
        model_dict = clus.serialize_spectral_clustering(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, SpectralBiclustering):
        model_dict = clus.serialize_spectral_biclustering(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, SpectralCoclustering):
        model_dict = clus.serialize_spectral_coclustering(model)
        return serialize_version(model, model_dict)
    elif 'KPrototypes' in clus.__optionals__ and isinstance(model, KPrototypes):
        model_dict = clus.serialize_kprototypes(model)
        return serialize_version(model, model_dict)
    elif 'KModes' in clus.__optionals__ and isinstance(model, KModes):
        model_dict = clus.serialize_kmodes(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, Birch):
        model_dict = clus.serialize_birch(model)
        return serialize_version(model, model_dict)
    elif 'HDBSCAN' in clus.__optionals__ and isinstance(model, HDBSCAN):
        model_dict = clus.serialize_hdbscan(model)
        return serialize_version(model, model_dict)

    # Cross-decomposition
    elif isinstance(model, CCA):
        model_dict = crdec.serialize_cca(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, PLSCanonical):
        model_dict = crdec.serialize_pls_canonical(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, PLSRegression):
        model_dict = crdec.serialize_pls_regression(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, PLSSVD):
        model_dict = crdec.serialize_pls_svd(model)
        return serialize_version(model, model_dict)

    # Decomposition
    elif isinstance(model, PCA):
        model_dict = dec.serialize_pca(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, KernelPCA):
        model_dict = dec.serialize_kernel_pca(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, IncrementalPCA):
        model_dict = dec.serialize_incremental_pca(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, MiniBatchSparsePCA):
        model_dict = dec.serialize_minibatch_sparse_pca(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, SparsePCA):
        model_dict = dec.serialize_sparse_pca(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, MiniBatchDictionaryLearning):
        model_dict = dec.serialize_minibatch_dictionary_learning(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, DictionaryLearning):
        model_dict = dec.serialize_dictionary_learning(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, FactorAnalysis):
        model_dict = dec.serialize_factor_analysis(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, FastICA):
        model_dict = dec.serialize_fast_ica(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, LatentDirichletAllocation):
        model_dict = dec.serialize_latent_dirichlet_allocation(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, MiniBatchNMF):
        model_dict = dec.serialize_minibatch_nmf(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, NMF):
        model_dict = dec.serialize_nmf(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, SparseCoder):
        model_dict = dec.serialize_sparse_coder(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, TruncatedSVD):
        model_dict = dec.serialize_truncated_svd(model)
        return serialize_version(model, model_dict)

    # Manifold
    elif isinstance(model, TSNE):
        model_dict = man.serialize_tsne(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, MDS):
        model_dict = man.serialize_mds(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, Isomap):
        model_dict = man.serialize_isomap(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, LocallyLinearEmbedding):
        model_dict = man.serialize_locally_linear_embedding(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, SpectralEmbedding):
        model_dict = man.serialize_spectral_embedding(model)
        return serialize_version(model, model_dict)
    elif 'UMAP' in man.__optionals__ and isinstance(model, UMAP):
        model_dict = man.serialize_umap(model)
        return serialize_version(model, model_dict)
    elif 'OpenTSNE' in man.__optionals__ and isinstance(model, (OpenTSNE, OpenTSNEsklearn)):
        model_dict = man.serialize_opentsne(model)
        return serialize_version(model, model_dict)
    elif 'OpenTSNE' in man.__optionals__ and isinstance(model, OpenTSNEEmbedding):
        model_dict = man.serialize_opentsne_embedding(model)
        return serialize_version(model, model_dict)
    elif 'OpenTSNE' in man.__optionals__ and isinstance(model, OpenPartialTSNEEmbedding):
        model_dict = man.serialize_opentsne_partial_embedding(model)
        return serialize_version(model, model_dict)

    # Neighbors
    elif isinstance(model, NearestNeighbors):
        model_dict = nei.serialize_nearest_neighbors(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, KDTree):
        model_dict = nei.serialize_kdtree(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, KernelDensity):
        model_dict = nei.serialize_kernel_density(model)
        return serialize_version(model, model_dict)
    elif 'NNDescent' in nei.__optionals__ and isinstance(model, NNDescent):
        model_dict = nei.serialize_nndescent(model)
        return serialize_version(model, model_dict)

    # Feature Extraction
    elif isinstance(model, DictVectorizer):
        model_dict = ext.serialize_dict_vectorizer(model)
        return serialize_version(model, model_dict)

    # Preprocess
    elif isinstance(model, LabelEncoder):
        model_dict = pre.serialize_label_encoder(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, LabelBinarizer):
        model_dict = pre.serialize_label_binarizer(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, MultiLabelBinarizer):
        model_dict = pre.serialize_multilabel_binarizer(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, MinMaxScaler):
        model_dict = pre.serialize_minmax_scaler(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, StandardScaler):
        model_dict = pre.serialize_standard_scaler(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, RobustScaler):
        model_dict = pre.serialize_robust_scaler(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, MaxAbsScaler):
        model_dict = pre.serialize_maxabs_scaler(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, KernelCenterer):
        model_dict = pre.serialize_kernel_centerer(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, OneHotEncoder):
        model_dict = pre.serialize_onehot_encoder(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, OrdinalEncoder):
        model_dict = pre.serialize_ordinal_encoder(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, Normalizer):
        model_dict = pre.serialize_normalizer(model)
        return serialize_version(model, model_dict)

    # Applicability Domain
    elif isinstance(model, BoundingBoxApplicabilityDomain):
        model_dict = ad.serialize_bounding_box_applicability_domain(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, ConvexHullApplicabilityDomain):
        model_dict = ad.serialize_convex_hull_applicability_domain(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, PCABoundingBoxApplicabilityDomain):
        model_dict = ad.serialize_pca_bounding_box_applicability_domain(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, TopKatApplicabilityDomain):
        model_dict = ad.serialize_topkat_applicability_domain(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, LeverageApplicabilityDomain):
        model_dict = ad.serialize_leverage_applicability_domain(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, HotellingT2ApplicabilityDomain):
        model_dict = ad.serialize_hotelling_t2_applicability_domain(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, KernelDensityApplicabilityDomain):
        model_dict = ad.serialize_kernel_density_applicability_domain(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, IsolationForestApplicabilityDomain):
        model_dict = ad.serialize_isolation_forest_applicability_domain(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, CentroidDistanceApplicabilityDomain):
        model_dict = ad.serialize_centroid_distance_applicability_domain(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, KNNApplicabilityDomain):
        model_dict = ad.serialize_knn_applicability_domain(model)
        return serialize_version(model, model_dict)
    elif isinstance(model, StandardizationApproachApplicabilityDomain):
        model_dict = ad.serialize_standardization_approach_applicability_domain(model)
        return serialize_version(model, model_dict)

    # Balancing
    elif 'imblearn' in ous.__optionals__ and isinstance(model, ClusterCentroids):
        model_dict = ous.serialize_cluster_centroids(model)
        return serialize_version(model, model_dict)
    elif 'imblearn' in ous.__optionals__ and isinstance(model, CondensedNearestNeighbour):
        model_dict = ous.serialize_condensed_nearest_neighbours(model)
        return serialize_version(model, model_dict)
    elif 'imblearn' in ous.__optionals__ and isinstance(model, EditedNearestNeighbours):
        model_dict = ous.serialize_edited_nearest_neighbours(model)
        return serialize_version(model, model_dict)
    elif 'imblearn' in ous.__optionals__ and isinstance(model, RepeatedEditedNearestNeighbours):
        model_dict = ous.serialize_repeated_edited_nearest_neighbours(model)
        return serialize_version(model, model_dict)
    elif 'imblearn' in ous.__optionals__ and isinstance(model, AllKNN):
        model_dict = ous.serialize_all_knn(model)
        return serialize_version(model, model_dict)
    elif 'imblearn' in ous.__optionals__ and isinstance(model, InstanceHardnessThreshold):
        model_dict = ous.serialize_instance_hardness_threshold(model)
        return serialize_version(model, model_dict)
    elif 'imblearn' in ous.__optionals__ and isinstance(model, NearMiss):
        model_dict = ous.serialize_near_miss(model)
        return serialize_version(model, model_dict)
    elif 'imblearn' in ous.__optionals__ and isinstance(model, NeighbourhoodCleaningRule):
        model_dict = ous.serialize_neighbourhood_cleaning_rule(model)
        return serialize_version(model, model_dict)
    elif 'imblearn' in ous.__optionals__ and isinstance(model, OneSidedSelection):
        model_dict = ous.serialize_one_sided_selection(model)
        return serialize_version(model, model_dict)
    elif 'imblearn' in ous.__optionals__ and isinstance(model, RandomUnderSampler):
        model_dict = ous.serialize_random_under_sampler(model)
        return serialize_version(model, model_dict)
    elif 'imblearn' in ous.__optionals__ and isinstance(model, TomekLinks):
        model_dict = ous.serialize_tomek_links(model)
        return serialize_version(model, model_dict)
    elif 'imblearn' in ous.__optionals__ and isinstance(model, RandomOverSampler):
        model_dict = ous.serialize_random_over_sampler(model)
        return serialize_version(model, model_dict)
    elif 'imblearn' in ous.__optionals__ and isinstance(model, SMOTENC):
        model_dict = ous.serialize_smotenc(model)
        return serialize_version(model, model_dict)
    elif 'imblearn' in ous.__optionals__ and isinstance(model, SMOTEN):
        model_dict = ous.serialize_smoten(model)
        return serialize_version(model, model_dict)
    elif 'imblearn' in ous.__optionals__ and isinstance(model, SMOTE):
        model_dict = ous.serialize_smote(model)
        return serialize_version(model, model_dict)
    elif 'imblearn' in ous.__optionals__ and isinstance(model, ADASYN):
        model_dict = ous.serialize_adasyn(model)
        return serialize_version(model, model_dict)
    elif 'imblearn' in ous.__optionals__ and isinstance(model, BorderlineSMOTE):
        model_dict = ous.serialize_borderline_smote(model)
        return serialize_version(model, model_dict)
    elif 'imblearn' in ous.__optionals__ and isinstance(model, KMeansSMOTE):
        model_dict = ous.serialize_kmeans_smote(model)
        return serialize_version(model, model_dict)
    elif 'imblearn' in ous.__optionals__ and isinstance(model, SVMSMOTE):
        model_dict = ous.serialize_svm_smote(model)
        return serialize_version(model, model_dict)
    elif 'imblearn' in ous.__optionals__ and isinstance(model, SMOTEENN):
        model_dict = ous.serialize_smote_enn(model)
        return serialize_version(model, model_dict)
    elif 'imblearn' in ous.__optionals__ and isinstance(model, SMOTETomek):
        model_dict = ous.serialize_smote_tomek(model)
        return serialize_version(model, model_dict)

    # Pipeline
    elif isinstance(model, Pipeline):
        model_dict = ppl.serialize_pipeline(model)
        return serialize_version(model, model_dict)

    # Otherwise
    else:
        raise ModelNotSupported('This model type is not currently supported. Email support@mlrequest.com to request a feature or report a bug.')


def deserialize_model(model_dict: Dict):
    """Instantiate a machine learning model from a previously serialized model.

    :param model_dict: dictionary of the previously serialized model
    """
    # Verify model is fitted
    if 'unfitted' in model_dict.keys() and model_dict['unfitted']:
        check_version(model_dict)
        return deserialize_unfitted_model(model_dict)
