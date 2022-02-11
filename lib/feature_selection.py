import warnings
from collections import OrderedDict
from typing import Dict, Union, List
import os
import pickle
import numpy as np
import pandas as pd
from multiprocessing import RawArray, Array, Process
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

##### Beginning of helper methods. #####

def _check_positive_integer(**kwargs) -> None:

    for k, v in kwargs.items():
        if ( not isinstance(v, int) ) or (v < 1):
            raise ValueError(f"{k} must be a positive integer: got {v}")

##### End of helper methods. #####


class Stepwise_Additive_Selector:
    """
    Performs stepwise additive selection to achieve dimensionality reduction.

    Parameters
    ----------
    `max_retained`
        The maximum number of features that will be identified as predictive.
    `n_splits`
        Parametrizes the internal cross validation procedure used to determine 
        the predictive agency of a feature: the number of splits in this procedure.
    `n_repeats`
        Parametrizes the internal cross validation procedure used to determine 
        the predictive agency of a feature: the number of repeats in this procedure.
    `interaction_depth`
        The maximum number of features whose interaction terms should be included when
        determining the predictive utility of the feature. Once the number of features
        identified as predictive by the selector exceed `interaction depth`, interaction
        terms are no longer included. 
    `look_forward`
        The feature selection process will terminate when `look_forward` features
        have been included since the last improvement in predictive performance.
    `eps`
        Parametrizes what it means for a feature to induce an improvement in predictive
        performance. New score is an improvement over old score if new_score / old_score > 1 + eps
    `n_processes`
        The number of processes to use. Set to the number of physical CPU cores.
    `seed`
        Used to seed the random number generator.
    """
    def __init__(self,
        max_retained : int = 20, n_splits : int = 5, n_repeats : int = 1, interaction_depth : int = 5,
        look_forward : int = 3, eps : float = 0.015, n_processes : int = 1, seed : int = 9999):
        
        # Check validity of parameters.
        _check_positive_integer(
            max_retained = max_retained, n_splits = n_splits, n_repeats = n_repeats,
            look_forward = look_forward, n_processes = n_processes, seed = seed
        )
        if (eps <= 0) or (eps >= 1):
            raise ValueError(f"eps must be between 0 and 1 exclusive: got {eps}")

        # Declare and initialize constant attributes.
        self.__max_retained = max_retained
        self.__n_splits = n_splits
        self.__n_repeats = n_repeats
        self.__interaction_depth = interaction_depth
        self.__look_forward = look_forward
        self.__eps = eps
        self.__n_processes = n_processes
        self.__seed = seed
        
        # Declare stateful attributes.
        self.__n_features = None
        self.__global_feature_mapping = None
        self.__fold_best_indices = None
        self.__best_indices_graph = None
        self.__fold_best_scores = None
        self.__best_scores_graph = []

    @property
    def fold_best_indices(self) -> Union[np.ndarray, None]:
        """
        A 1-D array containing the indices of the features that were identified as 
        predictive in the data that was fit most recently using `fit()`. These feature
        indices are local to the fold, and may not correspond with global feature
        indices if the feature names were set manually.

        Return
        ------
        `np.ndarray`
            Feature indices of predictive features.
        `None`
            If no data has been fit.
        """
        return self.__fold_best_indices

    @property
    def best_indices_graph(self) -> Union[np.ndarray, None]:
        """
        A 2-D array whose i'th row corresponds to the results obtained when using
        the `fit()` method for the i'th time.

        Each row has the same number of entries as there are features in the data.
        If the entry in a row is 0, this means that the feature is not predictive.
        If the entry in a row is > 0, this means that the feature is predictive. The
        most predictive feature is assigned the number of features in the data, and 
        the least predictive feature that was identified as predictive is assigned 1.

        This 2-D array is useful to determine whether the same features are being 
        identified as predictive across different folds of a modelling pipeline that
        is utilizing a resampling method such as cross-validation or bootstrapping.
        This 2-D array can be difficult to interpret when the data contains many
        features; in this case, the `best_indices_tree` should be preferred.

        Return
        ------
        `np.ndarray`
            Scores of predictive features across folds.
        `None`
            If no data has been fit.

        Example
        -------
        Suppose the data has 5 features. In fold 1, only feature 4 was identified as
        predictive, and in fold 2, features 2 and 3 were identified as predictive, with
        feature 3 being more predictive: 
        >>> sas.best_indices_graph
        array([[0, 0, 0, 5, 0],
               [0, 4, 5, 0, 0]])
        """
        return self.__best_indices_graph

    @property
    def best_indices_tree(self) -> Union[OrderedDict, None]:
        """
        An ordered dictionary whose i'th key corresponds to the feature name of the
        i'th most important feature, and whose associated value is a 1-D array whose
        entries indicate the importance of the i'th feature throughout all folds.
        Thus, the length of the 1-D array will be equal to the number of folds fitted.
        If a feature was not identified as predictive in any folds, it is NOT included
        in the ordered dictionary.

        With respect to the 1-D array comprising the value: an entry of 0 means no 
        importance, while an entry of 1 means most important, an entry of 2 means 2'nd 
        most important, and so on.

        This ordered dictionary is useful to determine whether the same features are being 
        identified as predictive across different folds of a modelling pipeline that
        is utilizing a resampling method such as cross-validation or bootstrapping.
        This attribute should be preferred over `best_indices_graph` when the data
        contains many features.

        Return
        ------
        `OrderedDict`
            Key: feature name. \n
            Value: 1-D array of importance entries.
        `None`
            If no data has been fit.

        Example
        -------
        We will assume the same circumstances as in the example provided in 
        `best_indices_graph`:
        >>> sas.best_indices_graph
        array([[0, 0, 0, 5, 0],
               [0, 4, 5, 0, 0]])
        >>> sas.best_indices_tree
        OrderedDict([(3, array([1, 0], dtype=int32)),
                     (2, array([0, 1], dtype=int32)),
                     (1, array([0, 2], dtype=int32))])
        """
        if self.best_indices_graph is None:
            return None

        best_indices_graph = self.best_indices_graph.copy(order = 'C')
        importance_mask = best_indices_graph.sum(axis = 0)
        argsort_importance = importance_mask.argsort()[::-1]

        best_indices_graph[best_indices_graph == 0] = self.__n_features + 1
        tree = OrderedDict()
        for i in range( (importance_mask > 0).sum() ):
            ind = argsort_importance[i]
            tree.update( { self.__global_feature_names[ind] : (self.__n_features + 1) - best_indices_graph[:, ind] } )

        return tree

    @property
    def fold_best_scores(self) -> Union[np.ndarray, None]:
        """
        A 1-D array containing the scores of the features that were identified as 
        predictive in the data that was fit most recently using `fit()`. 

        Return
        ------
        `np.ndarray`
            Scores of predictive features.
        `None`
            If no data has been fit.
        """
        return self.__fold_best_scores

    @property
    def best_scores_graph(self) -> Union[List[List[float]], None]:
        """
        A list of lists where the i'th inner list contains the `fold_best_scores` 
        for the i'th fold.

        Return
        ------
        `List`
            List of lists of fold_best_scores.
        `None`
            If no data has been fit.
        """
        if len(self.__best_scores_graph) > 0:
            return self.__best_scores_graph
        else:
            return None

    @property
    def feature_mapping(self) -> Union[Dict, None]:
        """
        A dictionary where each key is a global feature name and its associated
        value is the index assigned to that feature. Hence, the number of entries
        is equal to the number of columns in `best_indices_graph`.

        Return
        ------
        `Dict`
            Key: global feature name. \n
            Value: index assigned to feature name ( 0 <= Value < ncol(best_indices_graph) ).
        `None`
            If no data has been fit.

        Example
        -------
        >>> feature_names = np.array(["feat1", "feat2"])
        >>> sas.set_feature_names( feature_names )
        >>> sas.feature_mapping
        {'feat1': 0, 'feat2': 1}
        """
        return self.__global_feature_mapping

    def set_feature_names(self, feature_names : np.ndarray) -> None:
        """
        Manually set the feature names that the selector should be aware of. If 
        the selector encounters a feature name in data to be fit that is not
        contained in `feature_names`, an error will be thrown.

        This must be done when the first fold of data that the selector sees does 
        not contain all of the features that are present in the modelling pipeline.

        Parameters
        ----------
        `feature_names`
            Feature names that the selector should be aware of.

        Raises
        ----------
        AttributeError:
            1) If `fit()` has already been called. 
        
        TypeError:
            1) If `feature_names` is not a numpy array.

        Example
        -------
        >>> feature_names = np.array(["feat1", "feat2"])
        >>> sas.set_feature_names( feature_names )
        >>> sas.feature_mapping
        {'feat1': 0, 'feat2': 1}
        """
        if self.__best_indices_graph is not None:
            raise AttributeError("Column names cannot be set if data has already been fitted. Reset()?")
        if not isinstance(feature_names, np.ndarray):
            raise TypeError(f"feature_names is of type {type(feature_names)}. Expected {np.ndarray}.")

        self.__n_features = len(feature_names)
        self.__global_feature_names = feature_names
        self.__global_feature_mapping = dict( zip( feature_names, range(self.__n_features) ) )
        self.__fold_to_global_feat_map = dict( zip( range(self.__n_features), range(self.__n_features) ) )

    def fit(self,
        X : pd.DataFrame, y : Union[pd.Series, np.ndarray],
        nominal_indices : set = None) -> None:
        """
        Perform stepwise additive selection on `X` using `y` as labels.

        Parameters
        ----------
        `X`
            High-dimensional data to be fitted (excluding labels).
        `y`
            The corresponding labels to `X`.
        `nominal indices`
            Column indices of X that correspond to nominal features. If X doesn't
            contain any nominal features, leave set to None.

        Raises
        ----------
        AttributeError:
            1) A column name in `X` is not in `feature_mapping`.

        TypeError:
            1) If `X` is not a pandas DataFrame. \n
            2) If `y` is not either a pandas Series or 1-D numpy array.

        ValueError:
            1) If the number of rows in `X` != the length of `y`.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X is of type {type(X)}. Expected {pd.DataFrame}.")

        if self.__global_feature_mapping is None:
            self.set_feature_names( X.columns.to_numpy() ) 
        else:
            self.__fold_to_global_feat_map = dict()
            for fold_col_ind, col_name in enumerate( X.columns.to_numpy() ):
                global_col_ind = self.__global_feature_mapping.get(col_name)
                if global_col_ind is None:
                    raise ValueError(f"{col_name} in X is not in the current feature mapping. Reset()?")
                self.__fold_to_global_feat_map.update( {fold_col_ind : global_col_ind} )

        X = X.to_numpy()

        if X.ndim != 2:
                raise ValueError(f"X has {X.ndim} dimensions. Expected 2.")

        if isinstance(y, pd.Series):
            y = y.to_numpy()
        elif not isinstance(y, np.ndarray):
            raise TypeError(f"y is of type {type(y)}. Expected {np.ndarray} or {pd.Series}.")

        if y.ndim != 1:
            raise ValueError(f"y has {y.ndim} dimensions. Expected 1.")
        if len(y) != X.shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows but y is length {len(y)}.")

        if self.__n_features < self.__max_retained:
            self.__max_retained = self.__n_features
            warnings.warn(f"max_retained has been reduced to {self.__n_features}.", RuntimeWarning)

        def worker_func(X, y, start, stop, best_indices_gl, nominal_indices,
            n_splits, n_repeats, seed, best_scores_per_feat):

            ohe = OneHotEncoder(
                drop = "if_binary",
                sparse = False
            )

            poly_model = PolynomialFeatures(
                degree = 2,
                interaction_only = True,
                include_bias = False,
                order = 'C'
            )

            rskf = RepeatedStratifiedKFold(
                n_splits = n_splits,
                n_repeats = n_repeats,
                random_state = seed
            )

            lr = LogisticRegression(
                penalty = "none",
                fit_intercept = True,
                tol = 1e-4,
                C = 100000,
                random_state = 9999,
                solver = "newton-cg",
                max_iter = 200,
                n_jobs = 1
            )

            local_scores = np.empty(shape = (n_splits*n_repeats,) )
            n_features = stop - start
            process_scores = np.zeros(shape = (n_features,) )
            indices_to_evaluate = set(best_indices_gl)

            for ind, feat in enumerate( range(start, stop, 1) ):

                if feat in indices_to_evaluate:
                    continue
                else:
                    indices_to_evaluate.add(feat)

                if nominal_indices is None:
                    final_data = X[:, best_indices_gl + [feat] ]
                else:
                    local_nominal_indices = list( indices_to_evaluate.intersection(nominal_indices) )
                    other_indices = list( indices_to_evaluate.difference(nominal_indices) )
                    
                    encoded_data = original_data = None
                    if len(local_nominal_indices) > 0:
                        encoded_data = ohe.fit_transform( X[:, local_nominal_indices ] )
                    if len(other_indices) > 0:
                        original_data = X[:, other_indices]
    
                    if (encoded_data is not None) and (original_data is not None):
                        final_data = np.hstack( (original_data, encoded_data) )
                    elif encoded_data is not None:
                        final_data = encoded_data
                    elif original_data is not None:
                        final_data = original_data
                    else:
                        raise ValueError("Encoded data and original data are both NoneType.")

                og_ncol = len(indices_to_evaluate)
                if (og_ncol > 1) and (og_ncol <= self.__interaction_depth): 
                    final_data = poly_model.fit_transform(final_data)
        
                fold = 0

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    
                    for tr, val in rskf.split(final_data, y):

                        fd_tr = final_data[tr] ; fd_val = final_data[val]
                        y_tr = y[tr] ; y_val = y[val]

                        lr.fit(fd_tr, y_tr)
                        local_scores[fold] = roc_auc_score(y_val, lr.predict(fd_val))
                        fold += 1

                process_scores[ind] = local_scores.mean()

            for ind, feat in enumerate( range(start, stop, 1) ):
                best_scores_per_feat[feat] = process_scores[ind]


        X_shape = X.shape
        n_features = X_shape[1]
        sh_X = RawArray("d", X.ravel())
        sh_X_np = np.frombuffer(sh_X, dtype = np.float64)
        sh_X_np = sh_X_np.reshape(X_shape)

        sh_y = RawArray("i", y)
        sh_y_np = np.frombuffer(sh_y, dtype = np.int32)

        best_scores_per_feat = Array("d", n_features, lock = True)

        best_indices_for_fold = []
        best_scores_for_fold = []
        highest_score = .5
        no_improvement = 0
        n_features_per_process = n_features // self.__n_processes
        n_remaining_features = n_features % self.__n_processes

        for _ in range(self.__max_retained):

            start = 0
            processes = []
            for i in range(self.__n_processes):

                stop = start + n_features_per_process
                if i < n_remaining_features:
                    stop += 1

                p = Process(
                    target = worker_func,
                    args = (sh_X_np, sh_y_np, start, stop, best_indices_for_fold, nominal_indices,
                        self.__n_splits, self.__n_repeats, self.__seed, best_scores_per_feat)
                    )
                p.start()
                processes.append(p)

                start = stop

            for i in range(self.__n_processes):
                processes[i].join()

            bspf_np = np.frombuffer(best_scores_per_feat.get_obj(), dtype = np.float64)
            best_score_per_iter_ind = bspf_np.argmax()
            best_score_per_iter = bspf_np[best_score_per_iter_ind]
            best_indices_for_fold.append( best_score_per_iter_ind )
            best_scores_for_fold.append( best_score_per_iter )

            # Consider new best score an improvement if it is >= (100*eps)% higher than highest score.
            if (best_score_per_iter / highest_score) > (1 + self.__eps):
                highest_score = best_score_per_iter
                no_improvement = 0
            else:
                no_improvement += 1
            
            if no_improvement == self.__look_forward:
                break
                
        if no_improvement > 0:
            # Remove the last 'no_improvement' elements.
            best_indices_for_fold = best_indices_for_fold[: -no_improvement]
            best_scores_for_fold = best_scores_for_fold[: -no_improvement]

        self.__fold_best_indices = np.array( best_indices_for_fold )
        self.__fold_best_scores = np.array( best_scores_for_fold )
        
        if self.__best_indices_graph is None:
            self.__best_indices_graph = np.zeros(shape = (1, self.__n_features), dtype = np.int32 )
        else:
            self.__best_indices_graph = np.vstack( (self.__best_indices_graph, np.zeros(shape = (1, self.__n_features), dtype = np.int32 )) )
        
        for i, ind in enumerate(best_indices_for_fold):
            # Map local ind to the global column space.
            self.__best_indices_graph[-1, self.__fold_to_global_feat_map[ind] ] = self.__n_features - i
        
        self.__best_scores_graph.append( best_scores_for_fold )

        print(f"{len(best_indices_for_fold)} informative features were found.")
    
    def pickle_best_indices_tree(self, file_path : str = "auto") -> None:
        """
        Serialize `best_indices_tree` into the pickle file format and store it as `file_path`.

        Parameters
        ----------
        `file_path`: default = "auto"
            If "auto", the pickle file is saved in the curent working directory as 
            `best_indices_tree.pkl`. Else, the absolute path of the pickle file
            will be `file_path`.

        Raises
        ------
        AttributeError:
            1) If `fit()` has not been called yet (i.e. `best_indices_tree` is None) 
        """
        if self.best_indices_tree is None:
            raise AttributeError("Best indices tree is None.")

        fp = os.path.join( os.getcwd(), "best_indices_tree.pkl" ) if file_path == "auto" else file_path

        with open(fp, "wb") as pickle_out:
            pickle.dump(self.best_indices_tree, pickle_out)

    def reset(self) -> None:
        """
        Reset all stateful attributes of the selector instance to their initial 
        values. This is useful if you want to reuse the selector instance on different 
        data.

        Example
        -------
        >>> data1, labels1 = ...
        >>> data2, labels2 = ...
        >>> sas.fit(data1, labels1)
        >>> sas.reset()
        >>> sas.fit(data2, labels2)
        """
        self.__n_features = None
        self.__global_feature_mapping = None
        self.__fold_best_indices = None
        self.__best_indices_graph = None
        self.__fold_best_scores = None
        self.__best_scores_graph = []
