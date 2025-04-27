import importlib
import inspect
import re
import random
import warnings
from collections import Counter
import unittest

import numpy as np
from scipy.stats import uniform
from sklearn.base import ClassifierMixin, RegressorMixin, TransformerMixin, ClusterMixin, BiclusterMixin, OutlierMixin, \
    DensityMixin, BaseEstimator, ClassNamePrefixFeaturesOutMixin
from sklearn.model_selection._split import BaseCrossValidator, _BaseKFold, _RepeatedSplits, BaseShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.datasets import make_regression, make_classification, fetch_20newsgroups, fetch_california_housing
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.linear_model._base import LinearClassifierMixin
from sklearn.svm import LinearSVR, LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics.pairwise import pairwise_kernels


warnings.simplefilter(action='ignore', category=FutureWarning)

from ml2json._base import recursive_serialize


class TestSklearn(unittest.TestCase):

    def setUp(self) -> None:
        self.modules = ['sklearn.cluster', 'sklearn.cross_decomposition', 'sklearn.decomposition',
                        'sklearn.discriminant_analysis', 'sklearn.ensemble', 'sklearn.feature_extraction',
                        'sklearn.feature_selection', 'sklearn.gaussian_process', 'sklearn.impute',
                        'sklearn.isotonic', 'sklearn.kernel_approximation', 'sklearn.kernel_ridge',
                        'sklearn.linear_model', 'sklearn.manifold', 'sklearn.mixture', 'sklearn.model_selection',
                        'sklearn.multiclass', 'sklearn.multioutput', 'sklearn.naive_bayes', 'sklearn.neighbors',
                        'sklearn.neural_network', 'sklearn.pipeline', 'sklearn.preprocessing',
                        'sklearn.random_projection', 'sklearn.semi_supervised', 'sklearn.svm', 'sklearn.tree']
        for i in range(len(self.modules)):
            self.modules[i] = (self.modules[i], importlib.import_module(self.modules[i]))


        self.X_reg, self.y_reg = make_regression(n_samples=500, n_features=30, n_informative=3, random_state=0, shuffle=False)
        self.y_reg = np.abs(self.y_reg)
        self.y_reg_rank = np.argsort(np.argsort(self.y_reg)).tolist()
        self.X_cls, self.y_cls = make_classification(n_samples=500, n_features=30, n_informative=3, random_state=0, shuffle=False)
        _, self.y_cls_multi = make_classification(n_classes=40, n_samples=500, n_features=30, n_informative=25, random_state=0,
                                             shuffle=False)
        self.X_dict_vctrz = [Counter(tok.lower() for tok in re.findall(r"\w+", text))
                        for text in fetch_20newsgroups(subset='train', categories=['sci.space'],
                                                       remove=('headers', 'footers', 'quotes')).data]
        self.X_kernel = pairwise_kernels(fetch_california_housing()['data'][:100], metric="linear", filter_params=True, degree=3,
                                    coef0=1)

        feature_hasher = FeatureHasher(n_features=3)
        features = []
        for i in range(0, 100):
            features.append({'a': random.randint(0, 2), 'b': random.randint(3, 5), 'c': random.randint(6, 8)})

        self.y_reg_sparse = [random.random() for i in range(0, 100)]
        self.X_reg_sparse = feature_hasher.transform(features)

    def test_model(self):
        for i in range(len(self.modules)):
            classes = inspect.getmembers(self.modules[i][1], inspect.isclass)
            for class_name, class_dec in classes:
                if inspect.isabstract(class_dec):
                    continue
                elif issubclass(class_dec, RegressorMixin) and class_dec is not RegressorMixin:
                    params = {}
                    X_reg_, y_reg_ = self.X_reg, self.y_reg
                    if class_name in ['CCA', 'PLSCanonical']:
                        params['n_components'] = 1
                    elif class_name in ['StackingRegressor', 'VotingRegressor']:
                        params['estimators'] = [('lr', RidgeCV()),
                                                ('svr', LinearSVR(random_state=42))]
                    elif class_name in ['IsotonicRegression']:
                        X_reg_ = self.X_reg[:, 0]
                    elif class_name in ['MultiTaskElasticNet', 'MultiTaskElasticNetCV', 'MultiTaskLasso', 'MultiTaskLassoCV']:
                        y_reg_ = np.vstack([y_reg_, y_reg_]).T
                    elif class_name in ['MultiOutputRegressor']:
                        params['estimator'] = RidgeCV()
                        y_reg_ = np.vstack([y_reg_, y_reg_]).T
                    elif class_name in ['RegressorChain']:
                        params['base_estimator'] = RidgeCV()
                        y_reg_ = np.vstack([y_reg_, y_reg_]).T
                    model = class_dec(**params)
                    _ = model.fit(X_reg_, y_reg_)
                elif (issubclass(class_dec, ClassifierMixin) and
                      class_dec not in [ClassifierMixin, LinearClassifierMixin] and
                      class_name not in ['_BaseDiscreteNB', '_BaseNB']):
                    params = {}
                    X_cls_, y_cls_ = self.X_cls, self.y_cls
                    if class_name in ['StackingClassifier', 'VotingClassifier']:
                        params['estimators'] = [('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
                                                ('svr', make_pipeline(StandardScaler(),
                                                                      LinearSVC(random_state=42)))]
                    elif class_name in ['FixedThresholdClassifier', 'TunedThresholdClassifierCV',
                                        'OneVsOneClassifier', 'OneVsRestClassifier',
                                        'OutputCodeClassifier', 'SelfTrainingClassifier']:
                        params['estimator'] = RandomForestClassifier(n_estimators=10, random_state=42)
                    elif class_name in ['ClassifierChain']:
                        params['base_estimator'] = LinearSVC(random_state=42)
                        y_cls_ = np.vstack([y_cls_, y_cls_]).T
                    elif class_name in ['MultiOutputClassifier']:
                        params['estimator'] = LinearSVC(random_state=42)
                        y_cls_ = np.vstack([y_cls_, y_cls_]).T
                    elif class_name in ['CategoricalNB', 'ComplementNB', 'MultinomialNB']:
                        X_cls_ = np.abs(X_cls_)
                    model = class_dec(**params)
                    _ = model.fit(X_cls_, y_cls_)
                elif (issubclass(class_dec, TransformerMixin) and
                      class_dec not in [TransformerMixin] and
                      class_name not in ['SelectorMixin', 'BaseRandomProjection']):
                    params = {}
                    X_cls_, y_cls_ = self.X_cls, self.y_cls
                    if class_name in ['PLSSVD']:
                        params['n_components'] = 1
                    elif class_name in ['DictionaryLearning', 'MiniBatchDictionaryLearning']:
                        X_cls_ = X_cls_[:100, :3]
                        y_cls_ = y_cls_[:100]
                    elif class_name in ['SparseCoder']:
                        X_cls_ = X_cls_[:100, :3]
                        y_cls_ = y_cls_[:100]
                        params['dictionary'] = np.array([[0.64073247, 0.51050534, -0.57345114],
                                                         [0.05330439, 0.60119588, -0.79732187],
                                                         [0.6753414, 0.54936544, -0.49204838],
                                                         [0.64448984, 0.59080913, -0.48536318],
                                                         [0.7365317, 0.51770697, -0.4353166],
                                                         [0.57270595, 0.50214318, -0.64796614],
                                                         [0.53450467, 0.57404664, -0.62030252],
                                                         [0.50931083, 0.5298429, -0.6781364],
                                                         [0.85999903, 0.38908464, -0.33017391],
                                                         [0.26433943, 0.60998531, -0.74702248],
                                                         [0.46826915, 0.62250362, -0.62706719],
                                                         [0.60188339, 0.56184998, -0.56750417],
                                                         [0.38213731, 0.64292139, -0.66379452],
                                                         [0.67742808, 0.48692251, -0.55135983],
                                                         [0.53849589, 0.65606482, -0.52877323]])
                    elif class_name in ['RFE', 'RFECV', 'SelectFromModel', 'SequentialFeatureSelector']:
                        params['estimator'] = RandomForestClassifier(n_estimators=5, random_state=42)
                    elif class_name in ['FeatureUnion']:
                        params['transformer_list'] = [("pca", PCA(n_components=1)),
                                                      ("svd", TruncatedSVD(n_components=2))]

                    elif class_name in ['KernelCenterer']:
                        X_cls_ = self.X_kernel
                    elif class_name in ['GaussianRandomProjection', 'SparseRandomProjection']:
                        X_cls_ = np.r_[X_cls_, X_cls_, X_cls_, X_cls_, X_cls_, X_cls_, X_cls_, X_cls_].T
                    elif class_name in ['LatentDirichletAllocation', 'MiniBatchNMF', 'NMF', 'AdditiveChi2Sampler']:
                        X_cls_ = np.abs(X_cls_)
                    elif class_name in ['DictVectorizer']:
                        X_cls_ = self.X_dict_vctrz
                    model = class_dec(**params)
                    if class_name not in ['LabelBinarizer', 'LabelEncoder', 'MultiLabelBinarizer']:
                        _ = model.fit(X_cls_, y_cls_)
                    elif class_name in ['MultiLabelBinarizer']:
                        _ = model.fit(np.c_[self.y_cls_multi, self.y_cls_multi[::-1]])
                    else:
                        _ = model.fit(y_cls_)
                elif (issubclass(class_dec, (ClusterMixin, BiclusterMixin, OutlierMixin)) and
                      class_name not in ['ClusterMixin', 'BiclusterMixin', 'OutlierMixin', 'DensityMixin']):
                    params = {}
                    X_cls_, y_cls_ = self.X_cls, self.y_cls
                    model = class_dec(**params)
                    _ = model.fit(X_cls_, y_cls_)
                elif issubclass(class_dec, (BaseEstimator)) and class_name not in ['BaseEstimator', 'BaseEnsemble']:
                    params = {}
                    X_cls_, y_cls_ = self.X_cls, self.y_cls
                    if class_name in ['GridSearchCV', 'HalvingGridSearchCV']:
                        params['estimator'] = RandomForestClassifier(n_jobs=5, random_state=42)
                        params['param_grid'] = {'n_estimators': [10, 20],
                                                'max_features': ['sqrt', 'log2'],
                                                'max_depth': [4, 5, 6, 7, 8],
                                                'criterion': ['gini', 'entropy']
                                                }
                    elif class_name in ['RandomizedSearchCV']:
                        params['estimator'] = LogisticRegression(solver='saga', tol=1e-2, max_iter=200, random_state=42)
                        params['param_distributions'] = dict(C=uniform(loc=0, scale=4), penalty=['l2', 'l1'])
                    elif class_name in ['Pipeline']:
                        params['steps'] = [('scaler', StandardScaler()), ('svc', SVC())]
                    model = class_dec(**params)
                    _ = model.fit(X_cls_, y_cls_)
                elif (issubclass(class_dec, (BaseCrossValidator, _BaseKFold, _RepeatedSplits, BaseShuffleSplit)) and
                      class_name not in ['BaseCrossValidator', '_BaseKFold', '_RepeatedSplits', 'BaseShuffleSplit']):
                    params = {}
                    y_cls_ = self.y_cls
                    if class_name in ['LeavePGroupsOut']:
                        params['n_groups'] = 2
                    elif class_name in ['LeavePOut']:
                        params['p'] = 2
                    elif class_name in ['PredefinedSplit']:
                        params['test_fold'] = y_cls_
                    model = class_dec(**params)
                elif class_name in ['ParameterGrid', 'ParameterSampler']:
                    params = {}
                    if class_name in ['ParameterGrid']:
                        params['param_grid'] = [{'kernel': ['linear']}, {'kernel': ['rbf'], 'gamma': [1, 10]}]
                    elif class_name in ['ParameterSampler']:
                        params['param_distributions'] = {'C': uniform(loc=0, scale=4)}
                        params['n_iter'] = 4
                    model = class_dec(**params)
                elif class_name.endswith('Mixin') or class_name in ['BaseEstimator', 'BaseEnsemble', '_BaseDiscreteNB', '_BaseNB', 'BaseCrossValidator', '_BaseKFold', '_RepeatedSplits', 'BaseShuffleSplit', 'HasMethods', 'Interval', 'StrOptions']:
                    continue
                elif class_name in ['LearningCurveDisplay']:
                    rf = RandomForestClassifier(random_state=0)
                    train_sizes, train_scores, test_scores = learning_curve(rf, self.X_cls, self.y_cls)
                    params = {'train_sizes': train_sizes,
                              'train_scores': train_scores,
                              'test_scores': test_scores,
                              'score_name': "Score"}
                    model = class_dec(**params)
                else:
                    raise ValueError(class_name)

                print(class_name)
                recursive_serialize(model)

    def test_random_embedding(self):
        from sklearn.ensemble import RandomTreesEmbedding
        model = RandomTreesEmbedding()
        X_cls_, y_cls_ = self.X_cls, self.y_cls
        model.fit(X_cls_, y_cls_)
        recursive_serialize(model)

    def test_logistic_regression_cv(self):
        from sklearn.linear_model import LogisticRegressionCV
        model = LogisticRegressionCV()
        X_cls_, y_cls_ = self.X_cls, self.y_cls
        model.fit(X_cls_, y_cls_)
        recursive_serialize(model)

    def test_passive_aggressive_classifier(self):
        from sklearn.linear_model import PassiveAggressiveClassifier
        model = PassiveAggressiveClassifier()
        X_cls_, y_cls_ = self.X_cls, self.y_cls
        model.fit(X_cls_, y_cls_)
        recursive_serialize(model)
