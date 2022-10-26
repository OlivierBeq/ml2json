# sklearn-json
Export scikit-learn model files to JSON for sharing or deploying predictive models with peace of mind.

# Why sklearn-json?
Other methods for exporting scikit-learn models require Pickle or Joblib (based on Pickle).
- Serializing model files with Pickle provides a simple attack vector for malicious users - they give an attacker the ability to execute arbitrary code wherever the file is deserialized. For an example see: https://www.smartfile.com/blog/python-pickle-security-problems-and-solutions/.
- Internal designs of Pickle and Joblib files make the binary files not mandatorily supported across Python versions.  

sklearn-json is a safe and transparent solution for exporting scikit-learn model files to text files both machine and human readeable.

### Safe
Export model files to 100% JSON which cannot execute code on deserialization.

### Transparent
Model files are serialized in JSON (i.e., not binary), so you have the ability to see exactly what's inside.

# Getting Started

sklearn-json makes exporting model files to JSON simple.

## Install
```
pip install https://github.com/OlivierBeq/sklearn-json/tarball/master
```

To install other all dependencies (e.g. XGBoost, HDBSCAN), use:

```
pip install -e git+https://github.com/OlivierBeq/sklearn-json.git#egg=sklearn-json[full]
```
## Example Usage

```python
import sklearn_json as skljson
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=0).fit(X, y)

skljson.to_json(model, file_name)
deserialized_model = skljson.from_json(file_name)

deserialized_model.predict(X)
```

# Features
The list of supported models is rapidly growing.
In addition of the support for scikit-learn models, sklearn-json supports the following librairies:
- scikit-learn-extra
- XGBoost
- LightGBM
- CatBoost
- Imbalanced-learn
- kmodes
- HDBSCAN
- UMAP
- PyNNDescent
- Prince

sklearn-json requires scikit-learn >= 0.21.3.

## Supported scikit-learn Models

|       Library      |                  Category                 |                        Class                        | Supported? |
|:------------------:|:-----------------------------------------:|:---------------------------------------------------:|:----------:|
| Scikit-Learn       | Clustering                                | cluster.AffinityPropagation                         |    Yes     |
| Scikit-Learn       | Clustering                                | cluster.AgglomerativeClustering                     |    Yes     |
| Scikit-Learn       | Clustering                                | cluster.Birch                                       |    Yes     |
| Scikit-Learn       | Clustering                                | cluster.DBSCAN                                      |    Yes     |
| Scikit-Learn       | Clustering                                | cluster.FeatureAgglomeration                        |    Yes     |
| Scikit-Learn       | Clustering                                | cluster.KMeans                                      |    Yes     |
| Scikit-Learn       | Clustering                                | cluster.BisectingKMeans                             |    Yes     |
| Scikit-Learn       | Clustering                                | cluster.MiniBatchKMeans                             |    Yes     |
| Scikit-Learn       | Clustering                                | cluster.MeanShift                                   |    Yes     |
| Scikit-Learn       | Clustering                                | cluster.OPTICS                                      |    Yes     |
| Scikit-Learn       | Clustering                                | cluster.SpectralClustering                          |    Yes     |
| Scikit-Learn       | Clustering                                | cluster.SpectralBiclustering                        |    Yes     |
| Scikit-Learn       | Clustering                                | cluster.SpectralCoclustering                        |    Yes     |
| Scikit-Learn       | Cross decomposition                       | cross_decomposition.CCA                             |    Yes     |
| Scikit-Learn       | Cross decomposition                       | cross_decomposition.PLSCanonical                    |    Yes     |
| Scikit-Learn       | Cross decomposition                       | cross_decomposition.PLSRegression                   |    Yes     |
| Scikit-Learn       | Cross decomposition                       | cross_decomposition.PLSSVD                          |    Yes     |
| Scikit-Learn       | Decomposition                             | decomposition.DictionaryLearning                    |    Yes     |
| Scikit-Learn       | Decomposition                             | decomposition.FactorAnalysis                        |    Yes     |
| Scikit-Learn       | Decomposition                             | decomposition.FastICA                               |    Yes     |
| Scikit-Learn       | Decomposition                             | decomposition.IncrementalPCA                        |    Yes     |
| Scikit-Learn       | Decomposition                             | decomposition.KernelPCA                             |    Yes     |
| Scikit-Learn       | Decomposition                             | decomposition.LatentDirichletAllocation             |    Yes     |
| Scikit-Learn       | Decomposition                             | decomposition.MiniBatchDictionaryLearning           |    Yes     |
| Scikit-Learn       | Decomposition                             | decomposition.MiniBatchSparsePCA                    |    Yes     |
| Scikit-Learn       | Decomposition                             | decomposition.NMF                                   |    Yes     |
| Scikit-Learn       | Decomposition                             | decomposition.MiniBatchNMF                          |    Yes     |
| Scikit-Learn       | Decomposition                             | decomposition.PCA                                   |    Yes     |
| Scikit-Learn       | Decomposition                             | decomposition.SparsePCA                             |    Yes     |
| Scikit-Learn       | Decomposition                             | decomposition.SparseCoder                           |    Yes     |
| Scikit-Learn       | Decomposition                             | decomposition.TruncatedSVD                          |    Yes     |
| Scikit-Learn       | Discriminant Analysis                     | discriminant_analysis.LinearDiscriminantAnalysis    |    Yes     |
| Scikit-Learn       | Discriminant Analysis                     | discriminant_analysis.QuadraticDiscriminantAnalysis |    Yes     |
| Scikit-Learn       | Ensemble Methods                          | ensemble.AdaBoostClassifier                         |    Yes     |
| Scikit-Learn       | Ensemble Methods                          | ensemble.AdaBoostRegressor                          |    Yes     |
| Scikit-Learn       | Ensemble Methods                          | ensemble.BaggingClassifier                          |    Yes     |
| Scikit-Learn       | Ensemble Methods                          | ensemble.BaggingRegressor                           |    Yes     |
| Scikit-Learn       | Ensemble Methods                          | ensemble.ExtraTreesClassifier                       |    Yes     |
| Scikit-Learn       | Ensemble Methods                          | ensemble.ExtraTreesRegressor                        |    Yes     |
| Scikit-Learn       | Ensemble Methods                          | ensemble.GradientBoostingClassifier                 |    Yes     |
| Scikit-Learn       | Ensemble Methods                          | ensemble.GradientBoostingRegressor                  |    Yes     |
| Scikit-Learn       | Ensemble Methods                          | ensemble.IsolationForest                            |    Yes     |
| Scikit-Learn       | Ensemble Methods                          | ensemble.RandomForestClassifier                     |    Yes     |
| Scikit-Learn       | Ensemble Methods                          | ensemble.RandomForestRegressor                      |    Yes     |
| Scikit-Learn       | Ensemble Methods                          | ensemble.RandomTreesEmbedding                       |    Yes     |
| Scikit-Learn       | Ensemble Methods                          | ensemble.StackingClassifier                         |     No     |
| Scikit-Learn       | Ensemble Methods                          | ensemble.StackingRegressor                          |     No     |
| Scikit-Learn       | Ensemble Methods                          | ensemble.VotingClassifier                           |     No     |
| Scikit-Learn       | Ensemble Methods                          | ensemble.VotingRegressor                            |     No     |
| Scikit-Learn       | Ensemble Methods                          | ensemble.HistGradientBoostingRegressor              |     No     |
| Scikit-Learn       | Ensemble Methods                          | ensemble.HistGradientBoostingClassifier             |     No     |
| Scikit-Learn       | Feature Extraction                        | feature_extraction.DictVectorizer                   |    Yes     |
| Scikit-Learn       | Feature Extraction                        | feature_extraction.FeatureHasher                    |     No     |
| Scikit-Learn       | Feature Extraction                        | feature_extraction.image.PatchExtractor             |     No     |
| Scikit-Learn       | Feature Extraction                        | feature_extraction.text.CountVectorizer             |     No     |
| Scikit-Learn       | Feature Extraction                        | feature_extraction.text.HashingVectorizer           |     No     |
| Scikit-Learn       | Feature Extraction                        | feature_extraction.text.TfidfTransformer            |     No     |
| Scikit-Learn       | Feature Extraction                        | feature_extraction.text.TfidfVectorizer             |     No     |
| Scikit-Learn       | Feature Selection                         | feature_selection.GenericUnivariateSelect           |     No     |
| Scikit-Learn       | Feature Selection                         | feature_selection.SelectPercentile                  |     No     |
| Scikit-Learn       | Feature Selection                         | feature_selection.SelectKBest                       |     No     |
| Scikit-Learn       | Feature Selection                         | feature_selection.SelectFpr                         |     No     |
| Scikit-Learn       | Feature Selection                         | feature_selection.SelectFdr                         |     No     |
| Scikit-Learn       | Feature Selection                         | feature_selection.SelectFromModel                   |     No     |
| Scikit-Learn       | Feature Selection                         | feature_selection.SelectFwe                         |     No     |
| Scikit-Learn       | Feature Selection                         | feature_selection.SequentialFeatureSelector         |     No     |
| Scikit-Learn       | Feature Selection                         | feature_selection.RFE                               |     No     |
| Scikit-Learn       | Feature Selection                         | feature_selection.RFECV                             |     No     |
| Scikit-Learn       | Feature Selection                         | feature_selection.VarianceThreshold                 |     No     |
| Scikit-Learn       | Gaussian Processes                        | gaussian_process.GaussianProcessClassifier          |     No     |
| Scikit-Learn       | Gaussian Processes                        | gaussian_process.GaussianProcessRegressor           |     No     |
| Scikit-Learn       | Impute                                    | impute.SimpleImputer                                |     No     |
| Scikit-Learn       | Impute                                    | impute.IterativeImputer                             |     No     |
| Scikit-Learn       | Impute                                    | impute.MissingIndicator                             |     No     |
| Scikit-Learn       | Impute                                    | impute.KNNImputer                                   |     No     |
| Scikit-Learn       | Isotonic regression                       | isotonic.IsotonicRegression                         |     No     |
| Scikit-Learn       | Kernel Approximation                      | kernel_approximation.AdditiveChi2Sampler            |     No     |
| Scikit-Learn       | Kernel Approximation                      | kernel_approximation.Nystroem                       |     No     |
| Scikit-Learn       | Kernel Approximation                      | kernel_approximation.PolynomialCountSketch          |     No     |
| Scikit-Learn       | Kernel Approximation                      | kernel_approximation.RBFSampler                     |     No     |
| Scikit-Learn       | Kernel Approximation                      | kernel_approximation.SkewedChi2Sampler              |     No     |
| Scikit-Learn       | Kernel Ridge Regression                   | kernel_ridge.KernelRidge                            |     No     |
| Scikit-Learn       | Linear Models                             | linear_model.LogisticRegression                     |    Yes     |
| Scikit-Learn       | Linear Models                             | linear_model.LogisticRegressionCV                   |     No     |
| Scikit-Learn       | Linear Models                             | linear_model.PassiveAggressiveClassifier            |     No     |
| Scikit-Learn       | Linear Models                             | linear_model.Perceptron                             |    Yes     |
| Scikit-Learn       | Linear Models                             | linear_model.RidgeClassifier                        |     No     |
| Scikit-Learn       | Linear Models                             | linear_model.RidgeClassifierCV                      |     No     |
| Scikit-Learn       | Linear Models                             | linear_model.SGDClassifier                          |     No     |
| Scikit-Learn       | Linear Models                             | linear_model.SGDOneClassSVM                         |     No     |
| Scikit-Learn       | Linear Models                             | linear_model.LinearRegression                       |    Yes     |
| Scikit-Learn       | Linear Models                             | linear_model.Ridge                                  |    Yes     |
| Scikit-Learn       | Linear Models                             | linear_model.RidgeCV                                |     No     |
| Scikit-Learn       | Linear Models                             | linear_model.SGDRegressor                           |     No     |
| Scikit-Learn       | Linear Models                             | linear_model.ElasticNet                             |    Yes     |
| Scikit-Learn       | Linear Models                             | linear_model.ElasticNetCV                           |     No     |
| Scikit-Learn       | Linear Models                             | linear_model.Lars                                   |     No     |
| Scikit-Learn       | Linear Models                             | linear_model.LarsCV                                 |     No     |
| Scikit-Learn       | Linear Models                             | linear_model.Lasso                                  |    Yes     |
| Scikit-Learn       | Linear Models                             | linear_model.LassoCV                                |     No     |
| Scikit-Learn       | Linear Models                             | linear_model.LassoLars                              |     No     |
| Scikit-Learn       | Linear Models                             | linear_model.LassoLarsCV                            |     No     |
| Scikit-Learn       | Linear Models                             | linear_model.LassoLarsIC                            |     No     |
| Scikit-Learn       | Linear Models                             | linear_model.OrthogonalMatchingPursuit              |     No     |
| Scikit-Learn       | Linear Models                             | linear_model.OrthogonalMatchingPursuitCV            |     No     |
| Scikit-Learn       | Linear Models                             | linear_model.ARDRegression                          |     No     |
| Scikit-Learn       | Linear Models                             | linear_model.BayesianRidge                          |     No     |
| Scikit-Learn       | Linear Models                             | linear_model.MultiTaskElasticNet                    |     No     |
| Scikit-Learn       | Linear Models                             | linear_model.MultiTaskElasticNetCV                  |     No     |
| Scikit-Learn       | Linear Models                             | linear_model.MultiTaskLasso                         |     No     |
| Scikit-Learn       | Linear Models                             | linear_model.MultiTaskLassoCV                       |     No     |
| Scikit-Learn       | Linear Models                             | linear_model.HuberRegressor                         |     No     |
| Scikit-Learn       | Linear Models                             | linear_model.QuantileRegressor                      |     No     |
| Scikit-Learn       | Linear Models                             | linear_model.RANSACRegressor                        |     No     |
| Scikit-Learn       | Linear Models                             | linear_model.TheilSenRegressor                      |     No     |
| Scikit-Learn       | Linear Models                             | linear_model.PoissonRegressor                       |     No     |
| Scikit-Learn       | Linear Models                             | linear_model.TweedieRegressor                       |     No     |
| Scikit-Learn       | Linear Models                             | linear_model.GammaRegressor                         |     No     |
| Scikit-Learn       | Linear Models                             | linear_model.PassiveAggressiveRegressor             |     No     |
| Scikit-Learn       | Manifold Learning                         | manifold.Isomap                                     |    Yes     |
| Scikit-Learn       | Manifold Learning                         | manifold.LocallyLinearEmbedding                     |    Yes     |
| Scikit-Learn       | Manifold Learning                         | manifold.MDS                                        |    Yes     |
| Scikit-Learn       | Manifold Learning                         | manifold.SpectralEmbedding                          |    Yes     |
| Scikit-Learn       | Manifold Learning                         | manifold.TSNE                                       |    Yes     |
| Scikit-Learn       | Gaussian Mixture Models                   | mixture.BayesianGaussianMixture                     |     No     |
| Scikit-Learn       | Gaussian Mixture Models                   | mixture.GaussianMixture                             |     No     |
| Scikit-Learn       | Model Selection                           | model_selection.GroupKFold                          |     No     |
| Scikit-Learn       | Model Selection                           | model_selection.GroupShuffleSplit                   |     No     |
| Scikit-Learn       | Model Selection                           | model_selection.KFold                               |     No     |
| Scikit-Learn       | Model Selection                           | model_selection.LeaveOneGroupOut                    |     No     |
| Scikit-Learn       | Model Selection                           | model_selection.LeavePGroupsOut                     |     No     |
| Scikit-Learn       | Model Selection                           | model_selection.LeaveOneOut                         |     No     |
| Scikit-Learn       | Model Selection                           | model_selection.LeavePOut                           |     No     |
| Scikit-Learn       | Model Selection                           | model_selection.PredefinedSplit                     |     No     |
| Scikit-Learn       | Model Selection                           | model_selection.RepeatedKFold                       |     No     |
| Scikit-Learn       | Model Selection                           | model_selection.RepeatedStratifiedKFold             |     No     |
| Scikit-Learn       | Model Selection                           | model_selection.ShuffleSplit                        |     No     |
| Scikit-Learn       | Model Selection                           | model_selection.StratifiedKFold                     |     No     |
| Scikit-Learn       | Model Selection                           | model_selection.StratifiedShuffleSplit              |     No     |
| Scikit-Learn       | Model Selection                           | model_selection.StratifiedGroupKFold                |     No     |
| Scikit-Learn       | Model Selection                           | model_selection.TimeSeriesSplit                     |     No     |
| Scikit-Learn       | Model Selection                           | model_selection.GridSearchCV                        |     No     |
| Scikit-Learn       | Model Selection                           | model_selection.HalvingGridSearchCV                 |     No     |
| Scikit-Learn       | Model Selection                           | model_selection.ParameterGrid                       |     No     |
| Scikit-Learn       | Model Selection                           | model_selection.ParameterSampler                    |     No     |
| Scikit-Learn       | Model Selection                           | model_selection.RandomizedSearchCV                  |     No     |
| Scikit-Learn       | Model Selection                           | model_selection.HalvingRandomSearchCV               |     No     |
| Scikit-Learn       | Multiclass classification                 | multiclass.OneVsRestClassifier                      |     No     |
| Scikit-Learn       | Multiclass classification                 | multiclass.OneVsOneClassifier                       |     No     |
| Scikit-Learn       | Multiclass classification                 | multiclass.OutputCodeClassifier                     |     No     |
| Scikit-Learn       | Multioutput regression and classification | multioutput.ClassifierChain                         |     No     |
| Scikit-Learn       | Multioutput regression and classification | multioutput.MultiOutputRegressor                    |     No     |
| Scikit-Learn       | Multioutput regression and classification | multioutput.MultiOutputClassifier                   |     No     |
| Scikit-Learn       | Multioutput regression and classification | multioutput.RegressorChain                          |     No     |
| Scikit-Learn       | Naive Bayes                               | naive_bayes.BernoulliNB                             |    Yes     |
| Scikit-Learn       | Naive Bayes                               | naive_bayes.CategoricalNB                           |     No     |
| Scikit-Learn       | Naive Bayes                               | naive_bayes.ComplementNB                            |    Yes     |
| Scikit-Learn       | Naive Bayes                               | naive_bayes.GaussianNB                              |    Yes     |
| Scikit-Learn       | Naive Bayes                               | naive_bayes.MultinomialNB                           |    Yes     |
| Scikit-Learn       | Nearest Neighbors                         | neighbors.BallTree                                  |     No     |
| Scikit-Learn       | Nearest Neighbors                         | neighbors.KDTree                                    |    Yes     |
| Scikit-Learn       | Nearest Neighbors                         | neighbors.KernelDensity                             |     No     |
| Scikit-Learn       | Nearest Neighbors                         | neighbors.KNeighborsClassifier                      |     No     |
| Scikit-Learn       | Nearest Neighbors                         | neighbors.KNeighborsRegressor                       |     No     |
| Scikit-Learn       | Nearest Neighbors                         | neighbors.KNeighborsTransformer                     |     No     |
| Scikit-Learn       | Nearest Neighbors                         | neighbors.LocalOutlierFactor                        |     No     |
| Scikit-Learn       | Nearest Neighbors                         | neighbors.RadiusNeighborsClassifier                 |     No     |
| Scikit-Learn       | Nearest Neighbors                         | neighbors.RadiusNeighborsRegressor                  |     No     |
| Scikit-Learn       | Nearest Neighbors                         | neighbors.RadiusNeighborsTransformer                |     No     |
| Scikit-Learn       | Nearest Neighbors                         | neighbors.NearestCentroid                           |     No     |
| Scikit-Learn       | Nearest Neighbors                         | neighbors.NearestNeighbors                          |    Yes     |
| Scikit-Learn       | Nearest Neighbors                         | neighbors.NeighborhoodComponentsAnalysis            |     No     |
| Scikit-Learn       | Neural network models                     | neural_network.BernoulliRBM                         |     No     |
| Scikit-Learn       | Neural network models                     | neural_network.MLPClassifier                        |    Yes     |
| Scikit-Learn       | Neural network models                     | neural_network.MLPRegressor                         |    Yes     |
| Scikit-Learn       | Pipeline                                  | pipeline.FeatureUnion                               |     No     |
| Scikit-Learn       | Pipeline                                  | pipeline.Pipeline                                   |     No     |
| Scikit-Learn       | Preprocessing and Normalization           | preprocessing.Binarizer                             |     No     |
| Scikit-Learn       | Preprocessing and Normalization           | preprocessing.FunctionTransformer                   |     No     |
| Scikit-Learn       | Preprocessing and Normalization           | preprocessing.KBinsDiscretizer                      |     No     |
| Scikit-Learn       | Preprocessing and Normalization           | preprocessing.KernelCenterer                        |    Yes     |
| Scikit-Learn       | Preprocessing and Normalization           | preprocessing.LabelBinarizer                        |    Yes     |
| Scikit-Learn       | Preprocessing and Normalization           | preprocessing.LabelEncoder                          |    Yes     |
| Scikit-Learn       | Preprocessing and Normalization           | preprocessing.MultiLabelBinarizer                   |    Yes     |
| Scikit-Learn       | Preprocessing and Normalization           | preprocessing.MaxAbsScaler                          |     No     |
| Scikit-Learn       | Preprocessing and Normalization           | preprocessing.MinMaxScaler                          |    Yes     |
| Scikit-Learn       | Preprocessing and Normalization           | preprocessing.Normalizer                            |     No     |
| Scikit-Learn       | Preprocessing and Normalization           | preprocessing.OneHotEncoder                         |    Yes     |
| Scikit-Learn       | Preprocessing and Normalization           | preprocessing.OrdinalEncoder                        |     No     |
| Scikit-Learn       | Preprocessing and Normalization           | preprocessing.PolynomialFeatures                    |     No     |
| Scikit-Learn       | Preprocessing and Normalization           | preprocessing.PowerTransformer                      |     No     |
| Scikit-Learn       | Preprocessing and Normalization           | preprocessing.QuantileTransformer                   |     No     |
| Scikit-Learn       | Preprocessing and Normalization           | preprocessing.RobustScaler                          |     No     |
| Scikit-Learn       | Preprocessing and Normalization           | preprocessing.SplineTransformer                     |     No     |
| Scikit-Learn       | Preprocessing and Normalization           | preprocessing.StandardScaler                        |    Yes     |
| Scikit-Learn       | Random projection                         | random_projection.GaussianRandomProjection          |     No     |
| Scikit-Learn       | Random projection                         | random_projection.SparseRandomProjection            |     No     |
| Scikit-Learn       | Semi-Supervised Learning                  | semi_supervised.LabelPropagation                    |     No     |
| Scikit-Learn       | Semi-Supervised Learning                  | semi_supervised.LabelSpreading                      |     No     |
| Scikit-Learn       | Semi-Supervised Learning                  | semi_supervised.SelfTrainingClassifier              |     No     |
| Scikit-Learn       | Support Vector Machines                   | svm.LinearSVC                                       |     No     |
| Scikit-Learn       | Support Vector Machines                   | svm.LinearSVR                                       |     No     |
| Scikit-Learn       | Support Vector Machines                   | svm.NuSVC                                           |     No     |
| Scikit-Learn       | Support Vector Machines                   | svm.NuSVR                                           |     No     |
| Scikit-Learn       | Support Vector Machines                   | svm.OneClassSVM                                     |     No     |
| Scikit-Learn       | Support Vector Machines                   | svm.SVC                                             |    Yes     |
| Scikit-Learn       | Support Vector Machines                   | svm.SVR                                             |    Yes     |
| Scikit-Learn       | Decision Trees                            | tree.DecisionTreeClassifier                         |    Yes     |
| Scikit-Learn       | Decision Trees                            | tree.DecisionTreeRegressor                          |    Yes     |
| Scikit-Learn       | Decision Trees                            | tree.ExtraTreeClassifier                            |    Yes     |
| Scikit-Learn       | Decision Trees                            | tree.ExtraTreeRegressor                             |    Yes     |
| Imbalanced-Learn   | Under-sampling                            | ClusterCentroids                                    |     No     |
| Imbalanced-Learn   | Under-sampling                            | CondensedNearestNeighbour                           |     No     |
| Imbalanced-Learn   | Under-sampling                            | EditedNearestNeighbours                             |     No     |
| Imbalanced-Learn   | Under-sampling                            | RepeatedEditedNearestNeighbours                     |     No     |
| Imbalanced-Learn   | Under-sampling                            | AllKNN                                              |     No     |
| Imbalanced-Learn   | Under-sampling                            | InstanceHardnessThreshold                           |     No     |
| Imbalanced-Learn   | Under-sampling                            | NearMiss                                            |     No     |
| Imbalanced-Learn   | Under-sampling                            | NeighbourhoodCleaningRule                           |     No     |
| Imbalanced-Learn   | Under-sampling                            | OneSidedSelection                                   |     No     |
| Imbalanced-Learn   | Under-sampling                            | RandomUnderSampler                                  |     No     |
| Imbalanced-Learn   | Under-sampling                            | TomekLinks                                          |     No     |
| Imbalanced-Learn   | Over-sampling                             | RandomOverSampler                                   |     No     |
| Imbalanced-Learn   | Over-sampling                             | SMOTE                                               |     No     |
| Imbalanced-Learn   | Over-sampling                             | SMOTENC                                             |     No     |
| Imbalanced-Learn   | Over-sampling                             | SMOTEN                                              |     No     |
| Imbalanced-Learn   | Over-sampling                             | ADASYN                                              |     No     |
| Imbalanced-Learn   | Over-sampling                             | BorderlineSMOTE                                     |     No     |
| Imbalanced-Learn   | Over-sampling                             | KMeansSMOTE                                         |     No     |
| Imbalanced-Learn   | Over-sampling                             | SVMSMOTE                                            |     No     |
| Imbalanced-Learn   | Combined over & under sampling            | SMOTEENN                                            |     No     |
| Imbalanced-Learn   | Combined over & under sampling            | SMOTETomek                                          |     No     |
| Imbalanced-Learn   | Ensemble Methods                          | EasyEnsembleClassifier                              |     No     |
| Imbalanced-Learn   | Ensemble Methods                          | RUSBoostClassifier                                  |     No     |
| Imbalanced-Learn   | Ensemble Methods                          | BalancedBaggingClassifier                           |     No     |
| Imbalanced-Learn   | Ensemble Methods                          | BalancedRandomForestClassifier                      |     No     |
| XGBoost            | Ensemble Methods                          | XGBRegressor                                        |    Yes     |
| XGBoost            | Ensemble Methods                          | XGBClassifier                                       |    Yes     |
| XGBoost            | Ensemble Methods                          | XGBRanker                                           |    Yes     |
| XGBoost            | Ensemble Methods                          | XGBRFRegressor                                      |    Yes     |
| XGBoost            | Ensemble Methods                          | XGBRFClassifier                                     |    Yes     |
| LightGBM           | Ensemble Methods                          | LGBMClassifier                                      |    Yes     |
| LightGBM           | Ensemble Methods                          | LGBMRegressor                                       |    Yes     |
| LightGBM           | Ensemble Methods                          | LGBMRanker                                          |    Yes     |
| CatBoost           | Ensemble Methods                          | CatBoostClassifier                                  |    Yes     |
| CatBoost           | Ensemble Methods                          | CatBoostRanker                                      |    Yes     |
| CatBoost           | Ensemble Methods                          | CatBoostRegressor                                   |    Yes     |
| kmodes             | Clustering                                | KModes                                              |    Yes     |
| kmodes             | Clustering                                | KPrototypes                                         |    Yes     |
| Scikit-Learn-extra | Clustering                                | cluster.KMedoids                                    |     No     |
| Scikit-Learn-extra | Clustering                                | cluster.CommonNNClustering                          |     No     |
| Scikit-Learn-extra | Kernel approximation                      | kernel_approximation.Fastfood                       |     No     |
| Scikit-Learn-extra | EigenPro                                  | kernel_methods.EigenProRegressor                    |     No     |
| Scikit-Learn-extra | Robust                                    | kernel_methods.EigenProClassifier                   |     No     |
| Scikit-Learn-extra | Robust                                    | robust.RobustWeightedClassifier                     |     No     |
| Scikit-Learn-extra | Robust                                    | robust.RobustWeightedRegressor                      |     No     |
| Scikit-Learn-extra | Robust                                    | robust.RobustWeightedKMeans                         |     No     |
| HDBSCAN            | Clustering                                | HDBSCAN                                             |    Yes     |
| UMAP               | Manifold Learning                         | UMAP                                                |    Yes     |
| PyNNDescent        | Nearest Neighbors                         | NNDescent                                           |    Yes     |
| Prince             | Decomposition                             | PCA                                                 |     No     |
| Prince             | Decomposition                             | CA                                                  |     No     |
| Prince             | Decomposition                             | MCA                                                 |     No     |
| Prince             | Decomposition                             | MFA                                                 |     No     |
| Prince             | Decomposition                             | FAMD                                                |     No     |
| Prince             | Decomposition                             | GPA                                                 |     No     |