# ml2json
Export scikit-learn model files to JSON for sharing or deploying predictive models with peace of mind.

This is the continuation of the work hosted at [OlivierBeq/sklearn-json](https://github.com/OlivierBeq/sklearn-json).

# Why ml2json?
Other methods for exporting scikit-learn models require Pickle or Joblib (based on Pickle).
- Serializing model files with Pickle provides a simple attack vector for malicious users - they give an attacker the ability to execute arbitrary code wherever the file is deserialized. For an example see: https://www.smartfile.com/blog/python-pickle-security-problems-and-solutions/.
- Internal designs of Pickle and Joblib files make the binary files not mandatorily supported across Python versions.  

ml2json is a safe and transparent solution for exporting scikit-learn model files to text files both machine and human readable.

### Safe
Export model files to 100% JSON which cannot execute code on deserialization.

### Transparent
Model files are serialized in JSON (i.e., not binary), so you have the ability to see exactly what's inside.

# Getting Started

ml2json makes exporting model files to JSON simple.

## Install
```
pip install ml2json
```

To install other all dependencies (e.g. XGBoost, HDBSCAN), use:

```
pip install ml2json[full]
```
## Example Usage

```python
import ml2json
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=0).fit(X, y)

ml2json.to_json(model, file_name)
deserialized_model = ml2json.from_json(file_name)

deserialized_model.predict(X)
```

# Features
The list of supported models is rapidly growing.
In addition of the support for scikit-learn models, ml2json supports the following libraries:
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
- MLChemAD
- openTSNE

ml2json requires scikit-learn >= 1.2.2, <=1.4.0.

## Supported scikit-learn Models

|       Library      |                  Category                 |                        Class                        |     Supported?     |
|:------------------:|:-----------------------------------------:|:---------------------------------------------------:|:------------------:|
| Scikit-Learn       | Clustering                                | cluster.AffinityPropagation                         | :heavy_check_mark: |
| Scikit-Learn       | Clustering                                | cluster.AgglomerativeClustering                     | :heavy_check_mark: |
| Scikit-Learn       | Clustering                                | cluster.Birch                                       | :heavy_check_mark: |
| Scikit-Learn       | Clustering                                | cluster.DBSCAN                                      | :heavy_check_mark: |
| Scikit-Learn       | Clustering                                | cluster.FeatureAgglomeration                        | :heavy_check_mark: |
| Scikit-Learn       | Clustering                                | cluster.KMeans                                      | :heavy_check_mark: |
| Scikit-Learn       | Clustering                                | cluster.BisectingKMeans                             | :heavy_check_mark: |
| Scikit-Learn       | Clustering                                | cluster.MiniBatchKMeans                             | :heavy_check_mark: |
| Scikit-Learn       | Clustering                                | cluster.MeanShift                                   | :heavy_check_mark: |
| Scikit-Learn       | Clustering                                | cluster.OPTICS                                      | :heavy_check_mark: |
| Scikit-Learn       | Clustering                                | cluster.SpectralClustering                          | :heavy_check_mark: |
| Scikit-Learn       | Clustering                                | cluster.SpectralBiclustering                        | :heavy_check_mark: |
| Scikit-Learn       | Clustering                                | cluster.SpectralCoclustering                        | :heavy_check_mark: |
| Scikit-Learn       | Cross decomposition                       | cross_decomposition.CCA                             | :heavy_check_mark: |
| Scikit-Learn       | Cross decomposition                       | cross_decomposition.PLSCanonical                    | :heavy_check_mark: |
| Scikit-Learn       | Cross decomposition                       | cross_decomposition.PLSRegression                   | :heavy_check_mark: |
| Scikit-Learn       | Cross decomposition                       | cross_decomposition.PLSSVD                          | :heavy_check_mark: |
| Scikit-Learn       | Decomposition                             | decomposition.DictionaryLearning                    | :heavy_check_mark: |
| Scikit-Learn       | Decomposition                             | decomposition.FactorAnalysis                        | :heavy_check_mark: |
| Scikit-Learn       | Decomposition                             | decomposition.FastICA                               | :heavy_check_mark: |
| Scikit-Learn       | Decomposition                             | decomposition.IncrementalPCA                        | :heavy_check_mark: |
| Scikit-Learn       | Decomposition                             | decomposition.KernelPCA                             | :heavy_check_mark: |
| Scikit-Learn       | Decomposition                             | decomposition.LatentDirichletAllocation             | :heavy_check_mark: |
| Scikit-Learn       | Decomposition                             | decomposition.MiniBatchDictionaryLearning           | :heavy_check_mark: |
| Scikit-Learn       | Decomposition                             | decomposition.MiniBatchSparsePCA                    | :heavy_check_mark: |
| Scikit-Learn       | Decomposition                             | decomposition.NMF                                   | :heavy_check_mark: |
| Scikit-Learn       | Decomposition                             | decomposition.MiniBatchNMF                          | :heavy_check_mark: |
| Scikit-Learn       | Decomposition                             | decomposition.PCA                                   | :heavy_check_mark: |
| Scikit-Learn       | Decomposition                             | decomposition.SparsePCA                             | :heavy_check_mark: |
| Scikit-Learn       | Decomposition                             | decomposition.SparseCoder                           | :heavy_check_mark: |
| Scikit-Learn       | Decomposition                             | decomposition.TruncatedSVD                          | :heavy_check_mark: |
| Scikit-Learn       | Discriminant Analysis                     | discriminant_analysis.LinearDiscriminantAnalysis    | :heavy_check_mark: |
| Scikit-Learn       | Discriminant Analysis                     | discriminant_analysis.QuadraticDiscriminantAnalysis | :heavy_check_mark: |
| Scikit-Learn       | Ensemble Methods                          | ensemble.AdaBoostClassifier                         | :heavy_check_mark: |
| Scikit-Learn       | Ensemble Methods                          | ensemble.AdaBoostRegressor                          | :heavy_check_mark: |
| Scikit-Learn       | Ensemble Methods                          | ensemble.BaggingClassifier                          | :heavy_check_mark: |
| Scikit-Learn       | Ensemble Methods                          | ensemble.BaggingRegressor                           | :heavy_check_mark: |
| Scikit-Learn       | Ensemble Methods                          | ensemble.ExtraTreesClassifier                       | :heavy_check_mark: |
| Scikit-Learn       | Ensemble Methods                          | ensemble.ExtraTreesRegressor                        | :heavy_check_mark: |
| Scikit-Learn       | Ensemble Methods                          | ensemble.GradientBoostingClassifier                 | :heavy_check_mark: |
| Scikit-Learn       | Ensemble Methods                          | ensemble.GradientBoostingRegressor                  | :heavy_check_mark: |
| Scikit-Learn       | Ensemble Methods                          | ensemble.IsolationForest                            | :heavy_check_mark: |
| Scikit-Learn       | Ensemble Methods                          | ensemble.RandomForestClassifier                     | :heavy_check_mark: |
| Scikit-Learn       | Ensemble Methods                          | ensemble.RandomForestRegressor                      | :heavy_check_mark: |
| Scikit-Learn       | Ensemble Methods                          | ensemble.RandomTreesEmbedding                       | :heavy_check_mark: |
| Scikit-Learn       | Ensemble Methods                          | ensemble.StackingClassifier                         | :heavy_check_mark: |
| Scikit-Learn       | Ensemble Methods                          | ensemble.StackingRegressor                          | :heavy_check_mark: |
| Scikit-Learn       | Ensemble Methods                          | ensemble.VotingClassifier                           | :heavy_check_mark: |
| Scikit-Learn       | Ensemble Methods                          | ensemble.VotingRegressor                            | :heavy_check_mark: |
| Scikit-Learn       | Ensemble Methods                          | ensemble.HistGradientBoostingRegressor              |        :x:         |
| Scikit-Learn       | Ensemble Methods                          | ensemble.HistGradientBoostingClassifier             |        :x:         |
| Scikit-Learn       | Feature Extraction                        | feature_extraction.DictVectorizer                   | :heavy_check_mark: |
| Scikit-Learn       | Feature Extraction                        | feature_extraction.FeatureHasher                    |        :x:         |
| Scikit-Learn       | Feature Extraction                        | feature_extraction.image.PatchExtractor             |        :x:         |
| Scikit-Learn       | Feature Extraction                        | feature_extraction.text.CountVectorizer             |        :x:         |
| Scikit-Learn       | Feature Extraction                        | feature_extraction.text.HashingVectorizer           |        :x:         |
| Scikit-Learn       | Feature Extraction                        | feature_extraction.text.TfidfTransformer            |        :x:         |
| Scikit-Learn       | Feature Extraction                        | feature_extraction.text.TfidfVectorizer             |        :x:         |
| Scikit-Learn       | Feature Selection                         | feature_selection.GenericUnivariateSelect           |        :x:         |
| Scikit-Learn       | Feature Selection                         | feature_selection.SelectPercentile                  |        :x:         |
| Scikit-Learn       | Feature Selection                         | feature_selection.SelectKBest                       |        :x:         |
| Scikit-Learn       | Feature Selection                         | feature_selection.SelectFpr                         |        :x:         |
| Scikit-Learn       | Feature Selection                         | feature_selection.SelectFdr                         |        :x:         |
| Scikit-Learn       | Feature Selection                         | feature_selection.SelectFromModel                   |        :x:         |
| Scikit-Learn       | Feature Selection                         | feature_selection.SelectFwe                         |        :x:         |
| Scikit-Learn       | Feature Selection                         | feature_selection.SequentialFeatureSelector         |        :x:         |
| Scikit-Learn       | Feature Selection                         | feature_selection.RFE                               |        :x:         |
| Scikit-Learn       | Feature Selection                         | feature_selection.RFECV                             |        :x:         |
| Scikit-Learn       | Feature Selection                         | feature_selection.VarianceThreshold                 |        :x:         |
| Scikit-Learn       | Gaussian Processes                        | gaussian_process.GaussianProcessClassifier          |        :x:         |
| Scikit-Learn       | Gaussian Processes                        | gaussian_process.GaussianProcessRegressor           |        :x:         |
| Scikit-Learn       | Impute                                    | impute.SimpleImputer                                |        :x:         |
| Scikit-Learn       | Impute                                    | impute.IterativeImputer                             |        :x:         |
| Scikit-Learn       | Impute                                    | impute.MissingIndicator                             |        :x:         |
| Scikit-Learn       | Impute                                    | impute.KNNImputer                                   |        :x:         |
| Scikit-Learn       | Isotonic regression                       | isotonic.IsotonicRegression                         |        :x:         |
| Scikit-Learn       | Kernel Approximation                      | kernel_approximation.AdditiveChi2Sampler            |        :x:         |
| Scikit-Learn       | Kernel Approximation                      | kernel_approximation.Nystroem                       |        :x:         |
| Scikit-Learn       | Kernel Approximation                      | kernel_approximation.PolynomialCountSketch          |        :x:         |
| Scikit-Learn       | Kernel Approximation                      | kernel_approximation.RBFSampler                     |        :x:         |
| Scikit-Learn       | Kernel Approximation                      | kernel_approximation.SkewedChi2Sampler              |        :x:         |
| Scikit-Learn       | Kernel Ridge Regression                   | kernel_ridge.KernelRidge                            |        :x:         |
| Scikit-Learn       | Linear Models                             | linear_model.LogisticRegression                     | :heavy_check_mark: |
| Scikit-Learn       | Linear Models                             | linear_model.LogisticRegressionCV                   |        :x:         |
| Scikit-Learn       | Linear Models                             | linear_model.PassiveAggressiveClassifier            |        :x:         |
| Scikit-Learn       | Linear Models                             | linear_model.Perceptron                             | :heavy_check_mark: |
| Scikit-Learn       | Linear Models                             | linear_model.RidgeClassifier                        |        :x:         |
| Scikit-Learn       | Linear Models                             | linear_model.RidgeClassifierCV                      |        :x:         |
| Scikit-Learn       | Linear Models                             | linear_model.SGDClassifier                          |        :x:         |
| Scikit-Learn       | Linear Models                             | linear_model.SGDOneClassSVM                         |        :x:         |
| Scikit-Learn       | Linear Models                             | linear_model.LinearRegression                       | :heavy_check_mark: |
| Scikit-Learn       | Linear Models                             | linear_model.Ridge                                  | :heavy_check_mark: |
| Scikit-Learn       | Linear Models                             | linear_model.RidgeCV                                |        :x:         |
| Scikit-Learn       | Linear Models                             | linear_model.SGDRegressor                           |        :x:         |
| Scikit-Learn       | Linear Models                             | linear_model.ElasticNet                             | :heavy_check_mark: |
| Scikit-Learn       | Linear Models                             | linear_model.ElasticNetCV                           |        :x:         |
| Scikit-Learn       | Linear Models                             | linear_model.Lars                                   |        :x:         |
| Scikit-Learn       | Linear Models                             | linear_model.LarsCV                                 |        :x:         |
| Scikit-Learn       | Linear Models                             | linear_model.Lasso                                  | :heavy_check_mark: |
| Scikit-Learn       | Linear Models                             | linear_model.LassoCV                                |        :x:         |
| Scikit-Learn       | Linear Models                             | linear_model.LassoLars                              |        :x:         |
| Scikit-Learn       | Linear Models                             | linear_model.LassoLarsCV                            |        :x:         |
| Scikit-Learn       | Linear Models                             | linear_model.LassoLarsIC                            |        :x:         |
| Scikit-Learn       | Linear Models                             | linear_model.OrthogonalMatchingPursuit              |        :x:         |
| Scikit-Learn       | Linear Models                             | linear_model.OrthogonalMatchingPursuitCV            |        :x:         |
| Scikit-Learn       | Linear Models                             | linear_model.ARDRegression                          |        :x:         |
| Scikit-Learn       | Linear Models                             | linear_model.BayesianRidge                          |        :x:         |
| Scikit-Learn       | Linear Models                             | linear_model.MultiTaskElasticNet                    |        :x:         |
| Scikit-Learn       | Linear Models                             | linear_model.MultiTaskElasticNetCV                  |        :x:         |
| Scikit-Learn       | Linear Models                             | linear_model.MultiTaskLasso                         |        :x:         |
| Scikit-Learn       | Linear Models                             | linear_model.MultiTaskLassoCV                       |        :x:         |
| Scikit-Learn       | Linear Models                             | linear_model.HuberRegressor                         |        :x:         |
| Scikit-Learn       | Linear Models                             | linear_model.QuantileRegressor                      |        :x:         |
| Scikit-Learn       | Linear Models                             | linear_model.RANSACRegressor                        |        :x:         |
| Scikit-Learn       | Linear Models                             | linear_model.TheilSenRegressor                      |        :x:         |
| Scikit-Learn       | Linear Models                             | linear_model.PoissonRegressor                       |        :x:         |
| Scikit-Learn       | Linear Models                             | linear_model.TweedieRegressor                       |        :x:         |
| Scikit-Learn       | Linear Models                             | linear_model.GammaRegressor                         |        :x:         |
| Scikit-Learn       | Linear Models                             | linear_model.PassiveAggressiveRegressor             |        :x:         |
| Scikit-Learn       | Manifold Learning                         | manifold.Isomap                                     | :heavy_check_mark: |
| Scikit-Learn       | Manifold Learning                         | manifold.LocallyLinearEmbedding                     | :heavy_check_mark: |
| Scikit-Learn       | Manifold Learning                         | manifold.MDS                                        | :heavy_check_mark: |
| Scikit-Learn       | Manifold Learning                         | manifold.SpectralEmbedding                          | :heavy_check_mark: |
| Scikit-Learn       | Manifold Learning                         | manifold.TSNE                                       | :heavy_check_mark: |
| Scikit-Learn       | Gaussian Mixture Models                   | mixture.BayesianGaussianMixture                     |        :x:         |
| Scikit-Learn       | Gaussian Mixture Models                   | mixture.GaussianMixture                             |        :x:         |
| Scikit-Learn       | Model Selection                           | model_selection.GroupKFold                          |        :x:         |
| Scikit-Learn       | Model Selection                           | model_selection.GroupShuffleSplit                   |        :x:         |
| Scikit-Learn       | Model Selection                           | model_selection.KFold                               |        :x:         |
| Scikit-Learn       | Model Selection                           | model_selection.LeaveOneGroupOut                    |        :x:         |
| Scikit-Learn       | Model Selection                           | model_selection.LeavePGroupsOut                     |        :x:         |
| Scikit-Learn       | Model Selection                           | model_selection.LeaveOneOut                         |        :x:         |
| Scikit-Learn       | Model Selection                           | model_selection.LeavePOut                           |        :x:         |
| Scikit-Learn       | Model Selection                           | model_selection.PredefinedSplit                     |        :x:         |
| Scikit-Learn       | Model Selection                           | model_selection.RepeatedKFold                       |        :x:         |
| Scikit-Learn       | Model Selection                           | model_selection.RepeatedStratifiedKFold             |        :x:         |
| Scikit-Learn       | Model Selection                           | model_selection.ShuffleSplit                        |        :x:         |
| Scikit-Learn       | Model Selection                           | model_selection.StratifiedKFold                     |        :x:         |
| Scikit-Learn       | Model Selection                           | model_selection.StratifiedShuffleSplit              |        :x:         |
| Scikit-Learn       | Model Selection                           | model_selection.StratifiedGroupKFold                |        :x:         |
| Scikit-Learn       | Model Selection                           | model_selection.TimeSeriesSplit                     |        :x:         |
| Scikit-Learn       | Model Selection                           | model_selection.GridSearchCV                        |        :x:         |
| Scikit-Learn       | Model Selection                           | model_selection.HalvingGridSearchCV                 |        :x:         |
| Scikit-Learn       | Model Selection                           | model_selection.ParameterGrid                       |        :x:         |
| Scikit-Learn       | Model Selection                           | model_selection.ParameterSampler                    |        :x:         |
| Scikit-Learn       | Model Selection                           | model_selection.RandomizedSearchCV                  |        :x:         |
| Scikit-Learn       | Model Selection                           | model_selection.HalvingRandomSearchCV               |        :x:         |
| Scikit-Learn       | Multiclass classification                 | multiclass.OneVsRestClassifier                      |        :x:         |
| Scikit-Learn       | Multiclass classification                 | multiclass.OneVsOneClassifier                       |        :x:         |
| Scikit-Learn       | Multiclass classification                 | multiclass.OutputCodeClassifier                     |        :x:         |
| Scikit-Learn       | Multioutput regression and classification | multioutput.ClassifierChain                         |        :x:         |
| Scikit-Learn       | Multioutput regression and classification | multioutput.MultiOutputRegressor                    |        :x:         |
| Scikit-Learn       | Multioutput regression and classification | multioutput.MultiOutputClassifier                   |        :x:         |
| Scikit-Learn       | Multioutput regression and classification | multioutput.RegressorChain                          |        :x:         |
| Scikit-Learn       | Naive Bayes                               | naive_bayes.BernoulliNB                             | :heavy_check_mark: |
| Scikit-Learn       | Naive Bayes                               | naive_bayes.CategoricalNB                           |        :x:         |
| Scikit-Learn       | Naive Bayes                               | naive_bayes.ComplementNB                            | :heavy_check_mark: |
| Scikit-Learn       | Naive Bayes                               | naive_bayes.GaussianNB                              | :heavy_check_mark: |
| Scikit-Learn       | Naive Bayes                               | naive_bayes.MultinomialNB                           | :heavy_check_mark: |
| Scikit-Learn       | Nearest Neighbors                         | neighbors.BallTree                                  |        :x:         |
| Scikit-Learn       | Nearest Neighbors                         | neighbors.KDTree                                    | :heavy_check_mark: |
| Scikit-Learn       | Nearest Neighbors                         | neighbors.KernelDensity                             | :heavy_check_mark: |
| Scikit-Learn       | Nearest Neighbors                         | neighbors.KNeighborsClassifier                      | :heavy_check_mark: |
| Scikit-Learn       | Nearest Neighbors                         | neighbors.KNeighborsRegressor                       | :heavy_check_mark: |
| Scikit-Learn       | Nearest Neighbors                         | neighbors.KNeighborsTransformer                     |        :x:         |
| Scikit-Learn       | Nearest Neighbors                         | neighbors.LocalOutlierFactor                        |        :x:         |
| Scikit-Learn       | Nearest Neighbors                         | neighbors.RadiusNeighborsClassifier                 |        :x:         |
| Scikit-Learn       | Nearest Neighbors                         | neighbors.RadiusNeighborsRegressor                  |        :x:         |
| Scikit-Learn       | Nearest Neighbors                         | neighbors.RadiusNeighborsTransformer                |        :x:         |
| Scikit-Learn       | Nearest Neighbors                         | neighbors.NearestCentroid                           |        :x:         |
| Scikit-Learn       | Nearest Neighbors                         | neighbors.NearestNeighbors                          | :heavy_check_mark: |
| Scikit-Learn       | Nearest Neighbors                         | neighbors.NeighborhoodComponentsAnalysis            |        :x:         |
| Scikit-Learn       | Neural network models                     | neural_network.BernoulliRBM                         |        :x:         |
| Scikit-Learn       | Neural network models                     | neural_network.MLPClassifier                        | :heavy_check_mark: |
| Scikit-Learn       | Neural network models                     | neural_network.MLPRegressor                         | :heavy_check_mark: |
| Scikit-Learn       | Pipeline                                  | pipeline.FeatureUnion                               |        :x:         |
| Scikit-Learn       | Pipeline                                  | pipeline.Pipeline                                   | :heavy_check_mark: |
| Scikit-Learn       | Preprocessing and Normalization           | preprocessing.Binarizer                             |        :x:         |
| Scikit-Learn       | Preprocessing and Normalization           | preprocessing.FunctionTransformer                   |        :x:         |
| Scikit-Learn       | Preprocessing and Normalization           | preprocessing.KBinsDiscretizer                      |        :x:         |
| Scikit-Learn       | Preprocessing and Normalization           | preprocessing.KernelCenterer                        | :heavy_check_mark: |
| Scikit-Learn       | Preprocessing and Normalization           | preprocessing.LabelBinarizer                        | :heavy_check_mark: |
| Scikit-Learn       | Preprocessing and Normalization           | preprocessing.LabelEncoder                          | :heavy_check_mark: |
| Scikit-Learn       | Preprocessing and Normalization           | preprocessing.MultiLabelBinarizer                   | :heavy_check_mark: |
| Scikit-Learn       | Preprocessing and Normalization           | preprocessing.MaxAbsScaler                          | :heavy_check_mark: |
| Scikit-Learn       | Preprocessing and Normalization           | preprocessing.MinMaxScaler                          | :heavy_check_mark: |
| Scikit-Learn       | Preprocessing and Normalization           | preprocessing.Normalizer                            | :heavy_check_mark: |
| Scikit-Learn       | Preprocessing and Normalization           | preprocessing.OneHotEncoder                         | :heavy_check_mark: |
| Scikit-Learn       | Preprocessing and Normalization           | preprocessing.OrdinalEncoder                        | :heavy_check_mark: |
| Scikit-Learn       | Preprocessing and Normalization           | preprocessing.PolynomialFeatures                    |        :x:         |
| Scikit-Learn       | Preprocessing and Normalization           | preprocessing.PowerTransformer                      |        :x:         |
| Scikit-Learn       | Preprocessing and Normalization           | preprocessing.QuantileTransformer                   |        :x:         |
| Scikit-Learn       | Preprocessing and Normalization           | preprocessing.RobustScaler                          | :heavy_check_mark: |
| Scikit-Learn       | Preprocessing and Normalization           | preprocessing.SplineTransformer                     |        :x:         |
| Scikit-Learn       | Preprocessing and Normalization           | preprocessing.StandardScaler                        | :heavy_check_mark: |
| Scikit-Learn       | Random projection                         | random_projection.GaussianRandomProjection          |        :x:         |
| Scikit-Learn       | Random projection                         | random_projection.SparseRandomProjection            |        :x:         |
| Scikit-Learn       | Semi-Supervised Learning                  | semi_supervised.LabelPropagation                    |        :x:         |
| Scikit-Learn       | Semi-Supervised Learning                  | semi_supervised.LabelSpreading                      |        :x:         |
| Scikit-Learn       | Semi-Supervised Learning                  | semi_supervised.SelfTrainingClassifier              |        :x:         |
| Scikit-Learn       | Support Vector Machines                   | svm.LinearSVC                                       |        :x:         |
| Scikit-Learn       | Support Vector Machines                   | svm.LinearSVR                                       |        :x:         |
| Scikit-Learn       | Support Vector Machines                   | svm.NuSVC                                           |        :x:         |
| Scikit-Learn       | Support Vector Machines                   | svm.NuSVR                                           |        :x:         |
| Scikit-Learn       | Support Vector Machines                   | svm.OneClassSVM                                     |        :x:         |
| Scikit-Learn       | Support Vector Machines                   | svm.SVC                                             | :heavy_check_mark: |
| Scikit-Learn       | Support Vector Machines                   | svm.SVR                                             | :heavy_check_mark: |
| Scikit-Learn       | Decision Trees                            | tree.DecisionTreeClassifier                         | :heavy_check_mark: |
| Scikit-Learn       | Decision Trees                            | tree.DecisionTreeRegressor                          | :heavy_check_mark: |
| Scikit-Learn       | Decision Trees                            | tree.ExtraTreeClassifier                            | :heavy_check_mark: |
| Scikit-Learn       | Decision Trees                            | tree.ExtraTreeRegressor                             | :heavy_check_mark: |
| Imbalanced-Learn   | Under-sampling                            | ClusterCentroids                                    | :heavy_check_mark: |
| Imbalanced-Learn   | Under-sampling                            | CondensedNearestNeighbour                           | :heavy_check_mark: |
| Imbalanced-Learn   | Under-sampling                            | EditedNearestNeighbours                             | :heavy_check_mark: |
| Imbalanced-Learn   | Under-sampling                            | RepeatedEditedNearestNeighbours                     | :heavy_check_mark: |
| Imbalanced-Learn   | Under-sampling                            | AllKNN                                              | :heavy_check_mark: |
| Imbalanced-Learn   | Under-sampling                            | InstanceHardnessThreshold                           | :heavy_check_mark: |
| Imbalanced-Learn   | Under-sampling                            | NearMiss                                            | :heavy_check_mark: |
| Imbalanced-Learn   | Under-sampling                            | NeighbourhoodCleaningRule                           | :heavy_check_mark: |
| Imbalanced-Learn   | Under-sampling                            | OneSidedSelection                                   | :heavy_check_mark: |
| Imbalanced-Learn   | Under-sampling                            | RandomUnderSampler                                  | :heavy_check_mark: |
| Imbalanced-Learn   | Under-sampling                            | TomekLinks                                          | :heavy_check_mark: |
| Imbalanced-Learn   | Over-sampling                             | RandomOverSampler                                   | :heavy_check_mark: |
| Imbalanced-Learn   | Over-sampling                             | SMOTE                                               | :heavy_check_mark: |
| Imbalanced-Learn   | Over-sampling                             | SMOTENC                                             | :heavy_check_mark: |
| Imbalanced-Learn   | Over-sampling                             | SMOTEN                                              | :heavy_check_mark: |
| Imbalanced-Learn   | Over-sampling                             | ADASYN                                              | :heavy_check_mark: |
| Imbalanced-Learn   | Over-sampling                             | BorderlineSMOTE                                     | :heavy_check_mark: |
| Imbalanced-Learn   | Over-sampling                             | KMeansSMOTE                                         | :heavy_check_mark: |
| Imbalanced-Learn   | Over-sampling                             | SVMSMOTE                                            | :heavy_check_mark: |
| Imbalanced-Learn   | Combined over & under sampling            | SMOTEENN                                            | :heavy_check_mark: |
| Imbalanced-Learn   | Combined over & under sampling            | SMOTETomek                                          | :heavy_check_mark: |
| Imbalanced-Learn   | Ensemble Methods                          | EasyEnsembleClassifier                              |        :x:         |
| Imbalanced-Learn   | Ensemble Methods                          | RUSBoostClassifier                                  |        :x:         |
| Imbalanced-Learn   | Ensemble Methods                          | BalancedBaggingClassifier                           |        :x:         |
| Imbalanced-Learn   | Ensemble Methods                          | BalancedRandomForestClassifier                      |        :x:         |
| XGBoost            | Ensemble Methods                          | XGBRegressor                                        | :heavy_check_mark: |
| XGBoost            | Ensemble Methods                          | XGBClassifier                                       | :heavy_check_mark: |
| XGBoost            | Ensemble Methods                          | XGBRanker                                           | :heavy_check_mark: |
| XGBoost            | Ensemble Methods                          | XGBRFRegressor                                      | :heavy_check_mark: |
| XGBoost            | Ensemble Methods                          | XGBRFClassifier                                     | :heavy_check_mark: |
| LightGBM           | Ensemble Methods                          | LGBMClassifier                                      | :heavy_check_mark: |
| LightGBM           | Ensemble Methods                          | LGBMRegressor                                       | :heavy_check_mark: |
| LightGBM           | Ensemble Methods                          | LGBMRanker                                          | :heavy_check_mark: |
| CatBoost           | Ensemble Methods                          | CatBoostClassifier                                  | :heavy_check_mark: |
| CatBoost           | Ensemble Methods                          | CatBoostRanker                                      | :heavy_check_mark: |
| CatBoost           | Ensemble Methods                          | CatBoostRegressor                                   | :heavy_check_mark: |
| kmodes             | Clustering                                | KModes                                              | :heavy_check_mark: |
| kmodes             | Clustering                                | KPrototypes                                         | :heavy_check_mark: |
| Scikit-Learn-extra | Clustering                                | cluster.KMedoids                                    |        :x:         |
| Scikit-Learn-extra | Clustering                                | cluster.CommonNNClustering                          |        :x:         |
| Scikit-Learn-extra | Kernel approximation                      | kernel_approximation.Fastfood                       |        :x:         |
| Scikit-Learn-extra | EigenPro                                  | kernel_methods.EigenProRegressor                    |        :x:         |
| Scikit-Learn-extra | Robust                                    | kernel_methods.EigenProClassifier                   |        :x:         |
| Scikit-Learn-extra | Robust                                    | robust.RobustWeightedClassifier                     |        :x:         |
| Scikit-Learn-extra | Robust                                    | robust.RobustWeightedRegressor                      |        :x:         |
| Scikit-Learn-extra | Robust                                    | robust.RobustWeightedKMeans                         |        :x:         |
| HDBSCAN            | Clustering                                | HDBSCAN                                             | :heavy_check_mark: |
| UMAP               | Manifold Learning                         | UMAP                                                | :heavy_check_mark: |
| PyNNDescent        | Nearest Neighbors                         | NNDescent                                           | :heavy_check_mark: |
| Prince             | Decomposition                             | PCA                                                 |        :x:         |
| Prince             | Decomposition                             | CA                                                  |        :x:         |
| Prince             | Decomposition                             | MCA                                                 |        :x:         |
| Prince             | Decomposition                             | MFA                                                 |        :x:         |
| Prince             | Decomposition                             | FAMD                                                |        :x:         |
| Prince             | Decomposition                             | GPA                                                 |        :x:         |
| MLChemAD           | Applicability Domain                      | BoundingBoxApplicabilityDomain                      | :heavy_check_mark: |
| MLChemAD           | Applicability Domain                      | ConvexHullApplicabilityDomain                       | :heavy_check_mark: |
| MLChemAD           | Applicability Domain                      | PCABoundingBoxApplicabilityDomain                   | :heavy_check_mark: |
| MLChemAD           | Applicability Domain                      | TopKatApplicabilityDomain                           | :heavy_check_mark: |
| MLChemAD           | Applicability Domain                      | LeverageApplicabilityDomain                         | :heavy_check_mark: |
| MLChemAD           | Applicability Domain                      | HotellingT2ApplicabilityDomain                      | :heavy_check_mark: |
| MLChemAD           | Applicability Domain                      | KernelDensityApplicabilityDomain                    | :heavy_check_mark: |
| MLChemAD           | Applicability Domain                      | IsolationForestApplicabilityDomain                  | :heavy_check_mark: |
| MLChemAD           | Applicability Domain                      | CentroidDistanceApplicabilityDomain                 | :heavy_check_mark: |
| MLChemAD           | Applicability Domain                      | KNNApplicabilityDomain                              | :heavy_check_mark: |
| MLChemAD           | Applicability Domain                      | StandardizationApproachApplicabilityDomain          | :heavy_check_mark: |
| openTSNE           | Manifold Learning                         | openTSNE.TSNE                                       | :heavy_check_mark: |
| openTSNE           | Manifold Learning                         | openTSNE.sklearn.TSNE                               | :heavy_check_mark: |
| openTSNE           | Manifold Learning                         | openTSNE.TSNE                                       | :heavy_check_mark: |
| openTSNE           | Manifold Learning                         | openTSNE.sklearn.TSNE                               | :heavy_check_mark: |
