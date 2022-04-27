# To enable forward type references.
from __future__ import annotations
import os
import subprocess
import warnings
from typing import Tuple, Union, Generator
import pandas as pd
from pandas.core.indexing import IndexingError
import numpy as np
from numba import njit, prange
import pyarrow.feather as feather
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

### Worker functions start. ###

def _type_checker(t : type, **kwargs : any):
    for k, v in kwargs.items():
        if not isinstance(v, t):
            raise TypeError(f"{k} is of type {type(v)}. Expected {t}.")

def _na_checker(**kwargs : pd.DataFrame):
    for k, v in kwargs.items():
        if v.isna().any().any():
            raise ValueError(f"Missing value(s) found in {k}.")

def _unpack_slice(slc : slice) -> Tuple:
    return slc.start, slc.stop, slc.step

@njit
def _owner_is_consecutive(owner : np.ndarray, n_timepoints : int) -> bool:

    if n_timepoints == 1:
        return True

    for i in range(0, len(owner), n_timepoints):
        owner_id = owner[i]
        for j in range(1, n_timepoints):
            if owner_id != owner[i + j]:
                return False

    return True
                
def _custom_write_feather(df : pd.DataFrame, dir : str, fn : str) -> None:
    feather.write_feather(
        # Have to reset index as arrow doesn't support index serialization.
        df = df.reset_index(),
        dest = os.path.join(dir, fn),
        compression = "lz4"
        )

### Worker functions end. ###

class __EsetPrepBase:
    """
    Abstract base class for subclasses EsetPrepFull and EsetPrepSplit. 
    """

    def __init__(self, time, n_timepoints, owner, label):
        
        if self.__class__.__name__ == "__EsetPrepBase":
            raise AttributeError(f"eset_prep_base cannot be instantiated.")

        self._time = time
        self._n_timepoints = n_timepoints
        self._owner = owner
        self._label = label

        self.__de_genes_fn = None

    def __repr__(self) -> str:
        
        if isinstance(self, EsetPrepFull):
            mes = "Object: EsetPrepFull \n"
            mes += f"Samples: {self.n_samples} \n"
            mes += f"Probes: {self.n_probes} \n"
            mes += f"Timepoints: {self.time_names} \n"
            mes += f"Owner: {self.owner} \n"
            mes += f"Label: {self.label} \n"
            mes += f"Clinical data: {'None' if self.clin_data is None else 'Yes'} \n"
            mes += f"Feature data: {'None' if self.feature_data is None else 'Yes'} \n"
            mes += "--------------------------------------\n"
            return mes

        if isinstance(self, EsetPrepSplit):
            mes = "Object: EsetPrepSplit \n"
            mes += f"Training samples: {self.n_tr_samples} \n"
            mes += f"Testing samples: {self.n_te_samples} \n"
            mes += f"Probes: {self.n_probes} \n"
            mes += f"Timepoints: {self.time_names} \n"
            mes += f"Owner: {self.owner} \n"
            mes += f"Label: {self.label} \n"
            mes += f"Clinical data: {'None' if self.clin_data_tr is None else 'Yes'} \n"
            mes += f"Feature data: {'None' if self.feature_data is None else 'Yes'} \n"
            mes += "--------------------------------------\n"
            return mes

    def __getitem__(self, keys : Union[slice, np.ndarray]) -> Union[EsetPrepFull, EsetPrepSplit]:
        
        if isinstance(keys, slice):
            
            start, stop, step = _unpack_slice(keys)
            start = start if start is not None else 0
            stop = stop if stop is not None else self.n_probes
            step = step if step is not None else 1
            feature_data = self.feature_data.iloc[start:stop:step] if self.feature_data is not None else None

            if isinstance(self, EsetPrepFull):
                return EsetPrepFull(
                    self.assay_data.iloc[start:stop:step],
                    self.pheno_data, self.clin_data, feature_data,
                    self.owner, self.time, self.label,
                    sort = False, validate = False
                    )

            if isinstance(self, EsetPrepSplit):
                return EsetPrepSplit(
                    self.assay_data_tr.iloc[start:stop:step],
                    self.assay_data_te.iloc[start:stop:step],
                    self.pheno_data_tr, self.pheno_data_te, 
                    self.clin_data_tr, self.clin_data_te,
                    feature_data,
                    self.owner, self.time, self.label,
                    sort = False, validate = False
                    )

        if isinstance(keys, np.ndarray):

            if np.issubdtype(keys.dtype, np.integer):
                feature_data = self.feature_data.iloc[keys] if self.feature_data is not None else None
                if isinstance(self, EsetPrepFull):
                    return EsetPrepFull(
                        self.assay_data.iloc[keys], self.pheno_data, self.clin_data,
                        feature_data, self.owner, self.time, self.label,
                        sort = False, validate = False
                        )
                if isinstance(self, EsetPrepSplit):
                    return EsetPrepSplit(
                        self.assay_data_tr.iloc[keys], self.assay_data_te.iloc[keys],
                        self.pheno_data_tr, self.pheno_data_te,
                        self.clin_data_tr, self.clin_data_te,
                        feature_data,
                        self.owner, self.time, self.label,
                        sort = False, validate = False
                        )
            else:
                feature_data = self.feature_data.loc[keys] if self.feature_data is not None else None
                if isinstance(self, EsetPrepFull):
                    return EsetPrepFull(
                        self.assay_data.loc[keys], self.pheno_data, self.clin_data,
                        feature_data, self.owner, self.time, self.label,
                        sort = False, validate = False
                        )
                if isinstance(self, EsetPrepSplit):
                    return EsetPrepSplit(
                        self.assay_data_tr.loc[keys], self.assay_data_te.loc[keys],
                        self.pheno_data_tr, self.pheno_data_te,
                        self.clin_data_tr, self.clin_data_te,
                        feature_data,
                        self.owner, self.time, self.label,
                        sort = False, validate = False
                        )

        raise IndexingError("This method of indexing is not supported. Use slice or ndarray.")

    @property
    def feature_data(self) -> Union[pd.DataFrame, None]:
        """
        The feature data that corresponds to the assay data. 
        
        Returns
        -------
        `pd.DataFrame`
            Feature data.
        `None`
            No feature data given.
        """
        return self._feature_data

    @feature_data.deleter
    def feature_data(self) -> None:
        self._feature_data = None

    @property
    def time(self) -> Union[str, None]:
        return self._time

    @property
    def n_timepoints(self) -> int:
        return self._n_timepoints

    @property 
    def time_names(self) -> Union[np.ndarray, None]:
        """
        The unique time categories. All assays belong to one of these categories. Each owner
        must have an assay acquired at each of the unique time categories.

        Returns
        -------
        `np.ndarray`
            Names of unique time categories.
        `None`
            No time given.
        """
        if self.time is None:
            return None
        elif isinstance(self, EsetPrepFull):
            return self.times.cat.categories.to_numpy()
        elif isinstance(self, EsetPrepSplit):
            return self.tr_times.cat.categories.to_numpy()
        
    @property
    def owner(self) -> str:
        return self._owner

    @property
    def label(self) -> str:
        return self._label

    @property
    def label_names(self) -> np.ndarray:
        """
        The unique label categories. All assays belong to one of these categories. All of 
        an owner's assays must belong to the same label category.

        Returns
        -------
        `np.ndarray`
            Names of unique time categories.
        """
        if isinstance(self, EsetPrepFull):
            return self.labels.cat.categories.to_numpy()
        elif isinstance(self, EsetPrepSplit):
            return self.tr_labels.cat.categories.to_numpy()

    @label_names.setter
    def label_names(self, names : Union[list, dict]):

        if isinstance(self, EsetPrepFull):
            if isinstance(names, list):
                self._EsetPrepFull__pheno_data[self.label].cat.reorder_categories(names, inplace = True)
            elif isinstance(names, dict):
                self._EsetPrepFull__pheno_data[self.label].cat.rename_categories(names, inplace = True)
            else:
                raise TypeError(f"'names' is of type {type(names)}. Expecting list or dict.") 

        elif isinstance(self, EsetPrepSplit):
            if isinstance(names, list):
                self._EsetPrepSplit__pheno_data_tr[self.label].cat.reorder_categories(names, inplace = True)
                self._EsetPrepSplit__pheno_data_te[self.label].cat.reorder_categories(names, inplace = True)
            elif isinstance(names, dict):
                self._EsetPrepSplit__pheno_data_tr[self.label].cat.rename_categories(names, inplace = True)
                self._EsetPrepSplit__pheno_data_te[self.label].cat.rename_categories(names, inplace = True)
            else:
                raise TypeError(f"'names' is of type {type(names)}. Expecting list or dict.") 

    @property
    def n_labels(self) -> int:
        """
        The number of unique label categories.

        Returns
        -------
        `int`
            Number of labels.
        """
        return len(self.label_names)

    @property
    def n_probes(self) -> int:
        """
        The number of features extracted from the assay. This is equivalent to the number 
        of rows in the assay_data.

        Returns
        -------
        `int`
            Number of probes.
        """
        if isinstance(self, EsetPrepFull):
            return self.assay_data.shape[0]
        elif isinstance(self, EsetPrepSplit):
            return self.assay_data_tr.shape[0] 

    @property
    def probes(self) -> np.ndarray:
        """
        The names of the features extracted from the assay. This is equivalent to the row
        names of the assay_data.

        Returns
        -------
        `np.ndarray`
            Names of probes.
        """
        if isinstance(self, EsetPrepFull):
            return self.assay_data.index.to_numpy()
        elif isinstance(self, EsetPrepSplit):
            return self.assay_data_tr.index.to_numpy()

    def cquery(self, query : str) -> Union[EsetPrepFull, EsetPrepSplit]:

        if isinstance(self, EsetPrepFull):   
            if self.clin_data is None:
                raise AttributeError("EsetPrepFull object has no clinical data.") 
            clin_data = self.clin_data.query(query)
            bool_index = np.isin( self.owners.to_numpy(), clin_data.index.to_numpy() )
            pheno_data = self.pheno_data.loc[bool_index]
            assay_data = self.assay_data.loc[:, pheno_data.index.to_numpy()]

            return EsetPrepFull(
                assay_data, pheno_data, clin_data, self.feature_data,
                self.owner, self.time, self.label,
                sort = False, validate = False
                )

        if isinstance(self, EsetPrepSplit):   
            if self.clin_data_tr is None:
                raise AttributeError("EsetPrepSplit object has no clinical data.") 
            clin_data_tr = self.clin_data_tr.query(query)
            clin_data_te = self.clin_data_te.query(query)
            bool_index_tr = np.isin( self.tr_owners.to_numpy(), clin_data_tr.index.to_numpy() )
            bool_index_te = np.isin( self.te_owners.to_numpy(), clin_data_te.index.to_numpy() )
            pheno_data_tr = self.pheno_data_tr.loc[bool_index_tr]
            pheno_data_te = self.pheno_data_te.loc[bool_index_te]
            assay_data_tr = self.assay_data_tr.loc[:, pheno_data_tr.index.to_numpy()]
            assay_data_te = self.assay_data_te.loc[:, pheno_data_te.index.to_numpy()]
            
            return EsetPrepSplit(
                assay_data_tr, assay_data_te,
                pheno_data_tr, pheno_data_te,
                clin_data_tr, clin_data_te,
                self.feature_data,
                self.owner, self.time, self.label,
                sort = False, validate = False 
                )

    def pquery(self, query : str) -> Union[EsetPrepFull, EsetPrepSplit]:

        if isinstance(self, EsetPrepFull):    
            pheno_data = self.pheno_data.query(query)
            
            times_counts = pheno_data[self.time].cat.remove_unused_categories().value_counts()
            if (times_counts != times_counts[0]).any():
                raise ValueError(f"Inappropriate query resulting in unbalanced times: \n {times_counts}")

            assay_data = self.assay_data.loc[:, pheno_data.index.to_numpy()]
            if self.clin_data is not None:
                n_tps = len(times_counts)
                clin_data = self.clin_data.loc[ pheno_data[self.owner].to_numpy()[::n_tps] ]
            else:
                clin_data = None

            return EsetPrepFull(
                assay_data, pheno_data, clin_data, self.feature_data,
                self.owner, self.time, self.label,
                sort = False, validate = False
                )

        if isinstance(self, EsetPrepSplit):
            pheno_data_tr = self.pheno_data_tr.query(query)
            pheno_data_te = self.pheno_data_te.query(query)
            assay_data_tr = self.assay_data_tr.loc[:, pheno_data_tr.index.to_numpy()]
            assay_data_te = self.assay_data_te.loc[:, pheno_data_te.index.to_numpy()]
            if self.clin_data_tr is not None:
                clin_data_tr = self.clin_data_tr.loc[ pheno_data_tr[self.owner].to_numpy() ]
                clin_data_te = self.clin_data_te.loc[ pheno_data_te[self.owner].to_numpy() ]
            else:
                clin_data_tr = clin_data_te = None

            return EsetPrepSplit(
                assay_data_tr, assay_data_te,
                pheno_data_tr, pheno_data_te,
                clin_data_tr, clin_data_te,
                self.feature_data,
                self.owner, self.time, self.label,
                sort = False, validate = False 
                )

    def screen_probes(self, sd_low : float, sd_high : float, threshold : Union[str,float] = "auto") \
        -> Union[EsetPrepFull, EsetPrepSplit]:
        """
        Remove probes with intensity values less than the threshold in a
        greater number of samples than the smallest experimental group.
        Remove probes that are either invariant (sd < sd_low) or excessively variant
        (sd > sd_high).
        """

        # Numba worker function.
        @njit(nogil = True, parallel = True, fastmath = True, cache = True)
        def probes_to_keep(mat : np.ndarray, smallest : int, threshold : float, \
            sd_low : float, sd_high : float) -> np.ndarray:
            """
            Each row in mat is a probe.
            Returns a boolean mask indicating which probes to keep.
            """
            n_probes = mat.shape[0]
            index = np.empty(n_probes, dtype = np.bool8)
            for i in prange(n_probes):
                probe_vec = mat[i, :]
                has_signal = np.sum((probe_vec > threshold)) > smallest
                sd = np.std(probe_vec)
                variant = (sd > sd_low) and (sd < sd_high)
                index[i] = has_signal and variant

            return index

        if threshold == "auto":
            # Get the median intensity for each probe.
            if isinstance(self, EsetPrepFull):
                medians = np.median(self.assay_data.to_numpy(), axis = 1)
            elif isinstance(self, EsetPrepSplit):
                medians = np.median(self.assay_data_tr.to_numpy(), axis = 1)
            # Acquire the threshold.
            threshold = np.percentile(medians, 40)

        if isinstance(self, EsetPrepFull):
            # Get the number of arrays in the smallest experimental group.
            smallest = self.labels.value_counts().min() // self.n_timepoints 
            # Get boolean index of which probes to keep.
            index = probes_to_keep(self.assay_data.to_numpy(), smallest, threshold, sd_low, sd_high)

        if isinstance(self, EsetPrepSplit):
            smallest = self.tr_labels.value_counts().min() // self.n_timepoints
            index = probes_to_keep(self.assay_data_tr.to_numpy(), smallest, threshold, sd_low, sd_high)
            
        return self[index]

    def de_to_feather(self, dir : str, merge_clinical : bool = False, include_feat : bool = True) -> None:
        """
        Store data for DE analysis in `dir` as feather files (uses Apache Arrow). \n
        File names stored in dir are: \n
            `assay_data_de` \n
            `pheno_data_de` \n
            `feature_data_de` \n
        """

        if not os.path.isdir(dir):
            raise NotADirectoryError(f"{dir} is not an existing directory.")

        if isinstance(self, EsetPrepFull):
            _custom_write_feather(self.assay_data, dir, "assay_data_de")
            if merge_clinical:
                if self.clin_data is not None:
                    pheno_clin_merged = pd.merge(
                        left = self.pheno_data,
                        right = self.clin_data,
                        how = "left",
                        left_on = self.owner,
                        right_index = True
                    )
                    _custom_write_feather(pheno_clin_merged, dir, "pheno_data_de")
                else:
                    _custom_write_feather(self.pheno_data, dir, "pheno_data_de")
                    warnings.warn("clin_data is None: no clinical data was merged into pheno_data.", RuntimeWarning)
            else:
                _custom_write_feather(self.pheno_data, dir, "pheno_data_de")

        if isinstance(self, EsetPrepSplit):
            _custom_write_feather(self.assay_data_tr, dir, "assay_data_de")
            if merge_clinical:
                if self.clin_data_tr is not None:
                    pheno_clin_merged = pd.merge(
                        left = self.pheno_data_tr,
                        right = self.clin_data_tr,
                        how = "left",
                        left_on = self.owner,
                        right_index = True
                    )
                    _custom_write_feather(pheno_clin_merged, dir, "pheno_data_de")
                else:
                    _custom_write_feather(self.pheno_data_tr, dir, "pheno_data_de")
                    warnings.warn("clin_data_tr is None: no clinical data was merged into pheno_data_tr.", RuntimeWarning)
            else:
                _custom_write_feather(self.pheno_data_tr, dir, "pheno_data_de")

        if include_feat:
            if self.feature_data is not None:
                _custom_write_feather(self.feature_data, dir, "feature_data_de")
            else:
                warnings.warn("feature_data is None: no feature_data was deposited.", RuntimeWarning)

    def perform_dea_r(self, R_path : str, de_genes_fp : str, verbose : bool = False) -> None:
        """
        Performs a DE analysis in R using `R_path` as the R script sent to R. \n
        The R script must store a .txt file of the DE genes in de_dir.
        """

        # Generate process to run DE analysis.
        R_process = subprocess.Popen(
            ["Rscript", R_path],
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            universal_newlines = True
        )

        # Run DE analysis in R.
        # communicate() is blocking on the R process.
        R_stdout, R_stderr = R_process.communicate()

        if verbose:
            print("From R stdout: ") ;  print(R_stdout)
            print("")
            print("From R stderr: ") ; print(R_stderr)

        # Ensure R script completed before exiting.
        self.__de_genes_fn = de_genes_fp
        if not os.path.isfile(self.__de_genes_fn):
            print("From R stdout: ") ;  print(R_stdout)
            print("")
            print("From R stderr: ") ; print(R_stderr)
            raise FileNotFoundError("From Python: list of DE genes is missing.")
        
    def remove_nde(self, sep : str = ";", **kwargs) -> Union[EsetPrepFull, EsetPrepSplit]:
        """
        Removes genes in object data that are not DE.
        """

        if "de_genes_index" in kwargs:
            de_genes = kwargs["de_genes_index"]
        else:
            if "de_genes_fn" in kwargs:
                self.__de_genes_fn = kwargs["de_genes_fn"]
            elif self.__de_genes_fn == None:
                raise AttributeError("No 'de_genes_fn' provided in kwargs.")

            with open(self.__de_genes_fn, mode = 'r') as file:
                de_genes = file.read()
            de_genes = np.array(de_genes.split(sep))
            
        return self[de_genes] 

    def intersect(self, eset_p : Union[EsetPrepFull, EsetPrepSplit]) \
        -> Union[Tuple[EsetPrepFull, EsetPrepFull], Tuple[EsetPrepSplit, EsetPrepSplit]]: 
        
        if not isinstance(self, type(eset_p)):
            raise TypeError(f"{type(eset_p)} is not of type self: {type(self)}.")

        intersect_genes = np.intersect1d(self.probes, eset_p.probes, assume_unique = True)

        return self[intersect_genes], eset_p[intersect_genes] 

    def scramble(self, inplace : bool = False, seed : int = 9999) -> Union[EsetPrepFull, EsetPrepSplit, None]:

        rng = np.random.default_rng(seed)

        if isinstance(self, EsetPrepFull):

            new_labels_c = self.labels.to_numpy()[:: self.n_timepoints ]
            rng.shuffle(new_labels_c)
            new_labels = new_labels_c.repeat(self.n_timepoints)

            if inplace:
                self.labels = new_labels
                return
            else:
                tmp_pheno_data = self.pheno_data.copy(deep = True)
                tmp_pheno_data[self.label] = pd.Categorical(new_labels)
                return EsetPrepFull(
                    self.assay_data, tmp_pheno_data, self.clin_data, self.feature_data,
                    self.owner, self.time, self.label,
                    sort = False, validate = False
                    )

        if isinstance(self, EsetPrepSplit):

            new_tr_labels_c = self.tr_labels.to_numpy()[:: self.n_timepoints ]
            new_te_labels_c = self.te_labels.to_numpy()[:: self.n_timepoints ]
            rng.shuffle(new_tr_labels_c)
            rng.shuffle(new_te_labels_c)
            new_tr_labels = new_tr_labels_c.repeat(self.n_timepoints)
            new_te_labels = new_te_labels_c.repeat(self.n_timepoints)

            if inplace:
                self.tr_labels = new_tr_labels
                self.te_labels = new_te_labels
            else:
                tmp_pheno_data_tr = self.pheno_data_tr.copy(deep = True)
                tmp_pheno_data_te = self.pheno_data_te.copy(deep = True)
                tmp_pheno_data_tr[self.label] = pd.Categorical(new_tr_labels)
                tmp_pheno_data_te[self.label] = pd.Categorical(new_te_labels)
                return EsetPrepSplit(
                    self.assay_data_tr, self.assay_data_te,
                    tmp_pheno_data_tr, tmp_pheno_data_te,
                    self.clin_data_tr, self.clin_data_te,
                    self.feature_data,
                    self.owner, self.time, self.label,
                    sort = False, validate = False 
                    )

    def extract_model_feature_names(self, include_clinical : bool = True) -> pd.Index:

        if isinstance(self, EsetPrepFull):
            og_assay_feat = self.assay_data.index
            tmp_clin_data = self.clin_data
        elif isinstance(self, EsetPrepSplit):
            og_assay_feat = self.assay_data_tr.index
            tmp_clin_data = self.clin_data_tr

        if self.n_timepoints == 1:
            if include_clinical:
                if tmp_clin_data is not None:
                    return tmp_clin_data.columns.append(og_assay_feat)
                else:
                    warnings.warn("clin_data is None: no clinical data features were included.", RuntimeWarning)
            
            return og_assay_feat

        model_feat = og_assay_feat + "_T1"
        
        for tp in range(2, self.n_timepoints + 1):
            model_feat = model_feat.append(og_assay_feat + f"_T{tp}")
            model_feat = model_feat.append(og_assay_feat + f"_T{tp}-T{tp-1}")

        if include_clinical:
            if tmp_clin_data is not None:
                return tmp_clin_data.columns.append(model_feat)
            else:
                warnings.warn("clin_data is None: no clinical data features were included.", RuntimeWarning)
        
        return model_feat

    def validate(self) -> None:

        if self.feature_data is not None: 
            _type_checker(pd.DataFrame, feature_data = self.feature_data)
            _na_checker(feature_data = self.feature_data)

        if isinstance(self, EsetPrepFull):
            
            if self.clin_data is not None: 
                _type_checker(pd.DataFrame, clin_data = self.clin_data)
                _na_checker(clin_data = self.clin_data)

            _type_checker(pd.DataFrame, assay_data = self.assay_data, pheno_data = self.pheno_data)
            _na_checker(assay_data = self.assay_data, pheno_data = self.pheno_data)

            if (self.assay_data.columns.to_numpy() != self.pheno_data.index.to_numpy()).any():
                raise ValueError("assay_data column names != pheno_data row names.")
            if len( np.unique(self.probes) ) != self.n_probes:
                raise ValueError("Probes are not unique.")
            if self.feature_data is not None:
                if (self.probes != self.feature_data.index.to_numpy()).any():
                    raise ValueError("assay_data row names != feature_data row names.")

            if (self.owners.value_counts() != self.n_timepoints).any():
                raise ValueError(f"Expected {self.n_timepoints} samples per {self.owner}.")
            if not _owner_is_consecutive(self.owners.to_numpy(), self.n_timepoints):
                raise ValueError(f"All {self.owner} are not consecutive.")
            if self.clin_data is not None:
                if ( self.owners.to_numpy()[::self.n_timepoints] != self.clin_data.index.to_numpy() ).any():
                    raise ValueError("Owners in pheno_data != clin_data index.")

            return

        if isinstance(self, EsetPrepSplit):
            
            if self.clin_data_tr is not None: 
                _type_checker(pd.DataFrame, clin_data_tr = self.clin_data_tr, clin_data_te = self.clin_data_te)
                _na_checker(clin_data_tr = self.clin_data_tr, clin_data_te = self.clin_data_te)

            _type_checker(pd.DataFrame,
                assay_data_tr = self.assay_data_tr, assay_data_te = self.assay_data_te,
                pheno_data_tr = self.pheno_data_tr, pheno_data_te = self.pheno_data_te,
                )
            _na_checker(
                assay_data_tr = self.assay_data_tr, assay_data_te = self.assay_data_te,
                pheno_data_tr = self.pheno_data_tr, pheno_data_te = self.pheno_data_te
                )
        
            n_probes_tr, n_samples_tr = self.assay_data_tr.shape
            n_probes_te, n_samples_te = self.assay_data_te.shape
            if n_probes_tr != n_probes_te:
                raise ValueError(f"{n_probes_tr} train probes != {n_probes_te} test probes.")
            if self.pheno_data_tr.shape[0] != n_samples_tr:
                raise ValueError(f"pheno_data_tr has {self.pheno_data_tr.shape[0]} samples. Expected {n_samples_tr}.")
            if self.pheno_data_te.shape[0] != n_samples_te:
                raise ValueError(f"pheno_data_te has {self.pheno_data_te.shape[0]} samples. Expected {n_samples_te}.")
            if self.feature_data is not None:    
                if self.feature_data.shape[0] != n_probes_tr:
                    raise ValueError(f"feature_data has {self.feature_data.shape[0]} probes. Expected {n_probes_tr}.")

            if len( np.unique(self.probes) ) != self.n_probes:
                raise ValueError("Probes are not unique.")
            if (self.assay_data_tr.index.to_numpy() != self.assay_data_te.index.to_numpy()).any():
                raise ValueError("assay_data_tr row names != assay_data_te row names.")
            if (self.pheno_data_tr.columns.to_numpy() != self.pheno_data_te.columns.to_numpy()).any():
                raise ValueError("pheno_data_tr column names != pheno_data_te column names.")

            if (self.assay_data_tr.columns.to_numpy() != self.pheno_data_tr.index.to_numpy()).any():
                raise ValueError("assay_data_tr column names != pheno_data_tr row names.")
            if (self.assay_data_te.columns.to_numpy() != self.pheno_data_te.index.to_numpy()).any():
                raise ValueError("assay_data_te column names != pheno_data_te row names.")
            if self.feature_data is not None:
                if (self.assay_data_tr.index.to_numpy() != self.feature_data.index.to_numpy()).any():
                    raise ValueError("assay_data_tr row names != feature_data row names.")
                if (self.assay_data_te.index.to_numpy() != self.feature_data.index.to_numpy()).any():
                    raise ValueError("assay_data_te row names != feature_data row names.")

            tr_time_names = self.tr_times.cat.categories.to_numpy()
            te_time_names = self.te_times.cat.categories.to_numpy()
            if (tr_time_names != te_time_names).any():
                raise ValueError(f"tr_time_names {tr_time_names} != te_time_names {te_time_names}.")
            if (self.tr_owners.value_counts() != self.n_timepoints).any():
                raise ValueError(f"Expected {self.n_timepoints} samples per {self.owner} in training data.")
            if (self.te_owners.value_counts() != self.n_timepoints).any():
                raise ValueError(f"Expected {self.n_timepoints} samples per {self.owner} in test data.")
            if not _owner_is_consecutive(self.tr_owners.to_numpy(), self.n_timepoints):
                raise ValueError(f"All {self.owner} are not consecutive in training data.")
            if not _owner_is_consecutive(self.te_owners.to_numpy(), self.n_timepoints):
                raise ValueError(f"All {self.owner} are not consecutive in test data.")
            if (self.clin_data_tr is not None) and (self.clin_data_te is None):
                raise TypeError("Expected test clin_data.")
            if (self.clin_data_tr is None) and (self.clin_data_te is not None):
                raise TypeError("Expected no test clin_data.")
            if self.clin_data_tr is not None:
                if ( self.tr_owners.to_numpy()[::self.n_timepoints] != self.clin_data_tr.index.to_numpy() ).any():
                    raise ValueError("Owners in pheno_data_tr != clin_data_tr index.")
                if ( self.te_owners.to_numpy()[::self.n_timepoints] != self.clin_data_te.index.to_numpy() ).any():
                    raise ValueError("Owners in pheno_data_te != clin_data_te index.")

    def _expand_indices(self, tr_c : np.ndarray, te_c : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Expand indices to accomodate n_timepoint samples per observation.
        """

        n_tps = self.n_timepoints
        if n_tps == 1:
            return tr_c, te_c

        tr_e = np.empty( len(tr_c)*n_tps, dtype = np.int32 )
        te_e = np.empty( len(te_c)*n_tps, dtype = np.int32 )
        for i in range(n_tps):
            # Fill train indices for timepoint i.
            tr_e[i::n_tps] = tr_c*n_tps + i
            # Fill test indices for timepoint i.
            te_e[i::n_tps] = te_c*n_tps + i 

        return tr_e, te_e

    def _assay_ttd(self, scope : Union[str, None] = None) -> pd.DataFrame:

        n_tps = self.n_timepoints
        owner = self.owner

        f_col_names = self.extract_model_feature_names( include_clinical = False )

        if isinstance(self, EsetPrepFull):
            f_row_names = self.pheno_data.iloc[::n_tps][owner].to_numpy()
            assay = self.assay_data.to_numpy().transpose()
        elif isinstance(self, EsetPrepSplit):
            if scope == "train":
                f_row_names = self.pheno_data_tr.iloc[::n_tps][owner].to_numpy()
                assay = self.assay_data_tr.to_numpy().transpose()
            elif scope == "test":
                f_row_names = self.pheno_data_te.iloc[::n_tps][owner].to_numpy()
                assay = self.assay_data_te.to_numpy().transpose()
            else:
                raise ValueError(f"Scope {scope} is invalid. Expected 'train' or 'test'.")

        if n_tps == 1:
            return pd.DataFrame(
                data = assay,
                index = f_row_names,
                columns = f_col_names,
                )
        else:
            n_probes = assay.shape[1] ; n_blocks = 2*n_tps - 1
            new_data = np.empty(shape = (len(f_row_names), n_probes*n_blocks  ), dtype = np.float32)
            prev_tp_data = assay[::n_tps] 
            new_data[:, :n_probes] = prev_tp_data ; block = 1
            for tp in range(2, n_tps+1):
                # Acquire this timepoint's probe data.
                current_tp_data = assay[ (tp-1)::n_tps ]
                # Increment block.
                block += 1
                # Assign actual probe data.
                new_data[:, n_probes*(block-1) : n_probes*block ] = current_tp_data
                # Increment block.
                block += 1
                # Assign differenced probe data.
                new_data[:, n_probes*(block-1) : n_probes*block ] = current_tp_data - prev_tp_data
                # Exchange previous data with current data.
                prev_tp_data = current_tp_data

            return pd.DataFrame(
                data = new_data,
                index = f_row_names,
                columns = f_col_names,
                )

    def _pheno_f(self, scope : Union[str, None] = None) -> pd.Series:

        if isinstance(self, EsetPrepFull):
            pheno = self.pheno_data[ ::self.n_timepoints ].set_index(self.owner, drop = True)
        elif isinstance(self, EsetPrepSplit):
            if scope == "train":
                pheno = self.pheno_data_tr[ ::self.n_timepoints ].set_index(self.owner, drop = True)
            elif scope == "test":
                pheno = self.pheno_data_te[ ::self.n_timepoints ].set_index(self.owner, drop = True)
            else:
                raise ValueError(f"Scope {scope} is invalid. Expected 'train' or 'test'.")

        new_categories = list( range(self.n_labels) )
        return pheno[ self.label ].cat.rename_categories(new_categories)
            
        
class EsetPrepFull(__EsetPrepBase):

    def __init__(self,
        assay_data : pd.DataFrame, pheno_data : pd.DataFrame,
        clin_data : Union[pd.DataFrame, None], feature_data : Union[pd.DataFrame, None],
        owner : str, time : Union[str, None], label : str,
        sort : bool = True,
        validate : bool = True
        ) -> None:
        """
        Instantiate an expressionet prep full object.

        Parameters
        ----------
        `assay_data`
            The assay data collected from an experiment. The row names are the names of
            the feaures that are yielded from the assay, and the column names are the 
            assay's identifiers. Hence, 1 column comprises the results from 1 assay.
        `pheno_data`
            The phenotype data that corresponds to the assay_data. The row names are the
            the assay's identifiers, and therefore must match exactly the column names of
            assay_data. The column names are expected to be descriptors of the phenotypes
            of the assayed subjects that are NOT appropriate to be included in data for
            modelling.
        `clin_data`
            The clinical data that corresponds to the pheno_data. If there isn't any 
            corresponding clinical data, set clin_data to None. The row names are the 
            "owners" of the assays, and therefore must match exactly the entries in the
            owner column of pheno_data. The column names are expected to be clinical 
            attributes of the assayed subjects that ARE appropriate to be included in data
            for modelling.
        `feature_data`
            The feature data that corresponds to the assay data. If there isn't any
            corresponding feature data, set feature_data to None. The row names are the 
            names of the feaures that are yielded from the assay, and therefore must match 
            exactly the row names of assay_data. The column names are expected to be 
            descriptors of the features that were yielded from the assay.
        `owner`
            Denotes the name of the column in pheno_data that specifies which subject
            an array in array_data belongs to. The unique entries in this column must
            match exactly the row names of clin_data. Note that if the object has n
            timepoints, there should be n entries of each owner in the column, describing
            which n assays belong to any given "owner".
        `time`
            Denotes the name of the column in pheno_data that specifies which timepoint
            an array was acquired in. If all arrays were acquired at the same timepoint 
            (or time isn't recorded / isn't important), set time to None.
        `label`
            Denotes the name of the column in pheno_data that specifies the outcome of the
            subject from which the assayed sample was taken. For a supervised learning 
            task, this column would be the column of labels (specifying which class an 
            assay belonged to).
        `sort`
            Should the assay_data, pheno_data, and clin_data (if provided) be sorted to 
            ensure it meets class specifications. Should only be set to False if you 
            really know what you are doing.
        `validate`
            Should the above parameters be checked for correctness and validity. Though 
            this check does incur some overhead, it should invariably be set to True 
            until a pipeline has been confirmed to be error-free.
        """
        if validate:
            if owner not in pheno_data.columns:
                raise ValueError(f"Owner '{owner}' not in pheno_data columns.")
            if time is not None:
                if time not in pheno_data.columns:
                    raise ValueError(f"Time '{time}' not in pheno_data columns.")  
            if label not in pheno_data.columns:
                raise ValueError(f"Label '{label}' not in pheno_data columns.")

        n_tps = 1 if time is None else len( pheno_data[time].unique() )
        super().__init__(time, n_tps, owner, label)

        if (time is not None) and sort:
            self.__pheno_data = (
                pheno_data
                .astype({time:"category", label:"category"})
                .sort_values(by = [owner, time])
                )
            self.__assay_data = assay_data.loc[:, self.__pheno_data.index.to_numpy()]
            if clin_data is not None:
                self.__clin_data = clin_data.loc[ self.__pheno_data[owner].to_numpy()[::n_tps] ] 
            else:
                self.__clin_data = None
        else:
            if time is not None:
                self.__pheno_data = pheno_data.astype({time:"category", label:"category"})
            else:
                self.__pheno_data = pheno_data.astype({label:"category"})
            self.__assay_data = assay_data.astype(np.float32)
            self.__clin_data = clin_data
        
        self._feature_data = feature_data

        if validate:
            self.validate()
        
    @property
    def assay_data(self) -> pd.DataFrame:
        """
        The assay data collected from an experiment.
        
        Returns
        -------
        `pd.DataFrame`
            Assay data.
        """
        return self.__assay_data

    @property
    def pheno_data(self) -> pd.DataFrame:
        """
        The phenotype data that corresponds to the assay_data.
        
        Returns
        -------
        `pd.DataFrame`
            Phenotype data.
        """
        return self.__pheno_data

    @property
    def clin_data(self) -> Union[pd.DataFrame, None]:
        """
        The clinical data that corresponds to the pheno_data. 
        
        Returns
        -------
        `pd.DataFrame`
            Clinical data.
        `None`
            No clinical data given.
        """
        return self.__clin_data

    @clin_data.deleter
    def clin_data(self) -> None:
        self.__clin_data = None

    @property
    def n_samples(self) -> int:
        """
        The number of assays present in assay_data.

        Returns
        -------
        `int`
            Number of samples.
        """
        return self.__assay_data.shape[1]

    @property
    def n_observations(self) -> int:
        """
        The number of unique subjects that have been assayed. This is commonly referred to
        as the number of biological replicates.

        In this case, n_observations = n_samples / n_timepoints.

        Returns
        -------
        `int`
            Number of biological replicates.
        """
        return self.n_samples // self.n_timepoints 

    @property
    def owners(self) -> pd.Series:
        """
        The column of owners in pheno_data.

        Returns
        -------
        `pd.Series`
            Owners.
        """
        return self.__pheno_data.loc[:, self.owner]

    @property
    def times(self) -> Union[pd.Series, None]:
        """
        The column of times in pheno_data.

        Returns
        -------
        `pd.Series`
            Times.
        `None`
            No time given.
        """
        if self.time is None: return None
        return self.__pheno_data.loc[:, self.time].cat.remove_unused_categories()

    @property
    def labels(self) -> pd.Series:
        """
        The column of labels in pheno_data.

        Returns
        -------
        `pd.Series`
            Labels.
        """
        return self.__pheno_data.loc[:, self.label].cat.remove_unused_categories()

    @labels.setter
    def labels(self, new_labels : np.ndarray):
        self.__pheno_data[self.label] = pd.Categorical(new_labels)

    def split(self, n_splits : int, test_size : float, seed : int = 9999) -> Generator[EsetPrepSplit]:
        """
        Splits EsetPrepFull object into EsetPrepSplit objects for preprocessing and modelling.

        Parameters
        ----------
        `n_splits`
            The number of times the EsetPrepFull object should be shuffled and split into 
            an EsetPrepSplit object. As such, the number of yielded EsetPrepSplit objects 
            equals n_splits.
        `test_size`
            The proportion of observations that should be considered as test data for the 
            EsetPrepSplit objects.
        `seed`
            Integer used to seed the random number generator.

        Returns
        -------
        `Generator[EsetPrepSplit]`
            Generator yielding EsetPrepSplit objects.
        """
        sss = StratifiedShuffleSplit(
            n_splits = n_splits,
            test_size = test_size,
            random_state = seed
            )

        n_tps = self.n_timepoints
        # Collapse data to 1 row per patient for splitting.
        collapsed_lab = self.labels.to_numpy()[::n_tps]

        for tr_c, te_c in sss.split(np.zeros( len(collapsed_lab) ), collapsed_lab):  
                
            if self.clin_data is not None:
                # Use compact indices for clinical data (always only 1 row per owner).
                clin_data_tr = self.clin_data.iloc[tr_c] 
                clin_data_te = self.clin_data.iloc[te_c]
            else:
                clin_data_tr = clin_data_te = None

            # Generate expanded indices for assay and pheno data.
            tr_e, te_e = self._expand_indices(tr_c, te_c)
            
            yield EsetPrepSplit(
                self.assay_data.iloc[:, tr_e], self.assay_data.iloc[:, te_e],
                self.pheno_data.iloc[tr_e], self.pheno_data.iloc[te_e],
                clin_data_tr, clin_data_te,
                self.feature_data,
                self.owner, self.time, self.label,
                sort = False, validate = False
                )


class EsetPrepSplit(__EsetPrepBase):

    def __init__(self,
        assay_data_tr : pd.DataFrame, assay_data_te : pd.DataFrame,
        pheno_data_tr : pd.DataFrame, pheno_data_te : pd.DataFrame,
        clin_data_tr : Union[pd.DataFrame, None], clin_data_te : Union[pd.DataFrame, None], 
        feature_data : Union[pd.DataFrame, None],
        owner : str, time : Union[str, None], label : str,
        sort : bool = True,
        validate : bool = True,
        **kwargs
        ):
        """
        Instantiate an expressionet prep split object.

        Parameters
        ----------
        `assay_data_tr`
            The training partition of the assay data collected from an experiment. 
            The row names are the names of the feaures that are yielded from the assay, 
            and the column names are the assay's identifiers. Hence, 1 column comprises
            the results from 1 assay.
        `assay_data_te`
            The test partition of the assay data collected from an experiment. 
            The row names are the names of the feaures that are yielded from the assay, 
            and the column names are the assay's identifiers. The row names must exactly
            match those of assay_data_tr.
        `pheno_data_tr`
            The phenotype data that corresponds to assay_data_tr. The row names are the
            the assay's identifiers, and therefore must match exactly the column names of
            assay_data_tr. The column names are expected to be descriptors of the phenotypes
            of the assayed subjects that are NOT appropriate to be included in data for
            modelling.
        `pheno_data_te`
            The phenotype data that corresponds to assay_data_te. The row names are the
            the assay's identifiers, and therefore must match exactly the column names of
            assay_data_te. The column names must exactly match those of pheno_data_tr.
        `clin_data_tr`
            The clinical data that corresponds to pheno_data_tr. If there isn't any 
            corresponding clinical data, set clin_data_tr to None. The row names are the 
            "owners" of the assays, and therefore must match exactly the entries in the
            owner column of pheno_data_tr. The column names are expected to be clinical 
            attributes of the assayed subjects that ARE appropriate to be included in data
            for modelling.
        `clin_data_te`
            The clinical data that corresponds to pheno_data_te. If there isn't any 
            corresponding clinical data, set clin_data_te to None. The row names are the 
            "owners" of the assays, and therefore must match exactly the entries in the
            owner column of pheno_data_te. The column names must exactly match those of 
            clin_data_tr.
        `feature_data`
            The feature data that corresponds to the assay data. If there isn't any
            corresponding feature data, set feature_data to None. The row names are the 
            names of the feaures that are yielded from the assay, and therefore must match 
            exactly the row names of assay_data_(tr/te). The column names are expected to be 
            descriptors of the features that were yielded from the assay. 
        `owner`
            Denotes the name of the column in pheno_data_(tr/te) that specifies which subject
            an array in array_data_(tr/te) belongs to. The unique entries in this column must
            match exactly the row names of clin_data_(tr/te). Note that if the object has n
            timepoints, there should be n entries of each owner in the column, describing
            which n assays belong to any given "owner".
        `time`
            Denotes the name of the column in pheno_data_(tr/te) that specifies which timepoint
            an array was acquired in. If all arrays were acquired at the same timepoint 
            (or time isn't recorded / isn't important), set time to None.
        `label`
            Denotes the name of the column in pheno_data_(tr/te) that specifies the outcome of the
            subject from which the assayed sample was taken. For a supervised learning 
            task, this column would be the column of labels (specifying which class an 
            assay belonged to).
        `sort`
            Should the assay_data_(tr/te), pheno_data_(tr/te), and clin_data_(tr/te) (if provided)
            be sorted to ensure it meets class specifications. Should only be set to False if you 
            really know what you are doing.
        `validate`
            Should the above parameters be checked for correctness and validity. Though 
            this check does incur some overhead, it should invariably be set to True 
            until a pipeline has been confirmed to be error-free.
        """
        if validate:
            if owner not in pheno_data_tr.columns:
                raise ValueError(f"Owner '{owner}' not in pheno_data_tr columns.")
            if owner not in pheno_data_te.columns:
                raise ValueError(f"Owner '{owner}' not in pheno_data_te columns.")
            if time not in pheno_data_tr.columns:
                raise ValueError(f"Time '{time}' not in pheno_data_tr columns.")
            if time not in pheno_data_te.columns:
                raise ValueError(f"Time '{time}' not in pheno_data_te columns.")
            if label not in pheno_data_tr.columns:
                raise ValueError(f"Label '{label}' not in pheno_data_tr columns.")
            if label not in pheno_data_te.columns:
                raise ValueError(f"Label '{label}' not in pheno_data_te columns.")

        n_tps = 1 if time is None else len( pheno_data_tr[time].unique() )
        super().__init__(time, n_tps, owner, label)

        if (time is not None) and sort:
            self.__pheno_data_tr = (
                pheno_data_tr
                .astype({time:"category", label:"category"})
                .sort_values(by = [owner, time])
                )
            self.__pheno_data_te = (
                pheno_data_te
                .astype({time:"category", label:"category"})
                .sort_values(by = [owner, time])
                )
            self.__assay_data_tr = assay_data_tr.loc[:, self.__pheno_data_tr.index.to_numpy()]
            self.__assay_data_te = assay_data_te.loc[:, self.__pheno_data_te.index.to_numpy()]
            if clin_data_tr is not None:
                self.__clin_data_tr = clin_data_tr.loc[ self.__pheno_data_tr[owner].to_numpy()[::n_tps] ] 
                self.__clin_data_te = clin_data_te.loc[ self.__pheno_data_te[owner].to_numpy()[::n_tps] ] 
            else:
                self.__clin_data_tr = None
                self.__clin_data_te = None
        else:
            if time is not None:
                self.__pheno_data_tr = pheno_data_tr.astype({time:"category", label:"category"})
                self.__pheno_data_te = pheno_data_te.astype({time:"category", label:"category"})
            else:
                self.__pheno_data_tr = pheno_data_tr.astype({label:"category"})
                self.__pheno_data_te = pheno_data_te.astype({label:"category"})
            self.__assay_data_tr = assay_data_tr.astype(np.float32)
            self.__assay_data_te = assay_data_te.astype(np.float32)
            self.__clin_data_tr = clin_data_tr
            self.__clin_data_te = clin_data_te
        
        self._feature_data = feature_data

        if "scalers" in kwargs:
            self.__scalers = kwargs["scalers"]
        
    @property
    def assay_data_tr(self) -> pd.DataFrame:
        """
        The training partition of the assay data collected from an experiment.
        
        Returns
        -------
        `pd.DataFrame`
            Training assay data.
        """
        return self.__assay_data_tr

    @property
    def assay_data_te(self) -> pd.DataFrame:
        """
        The test partition of the assay data collected from an experiment.
        
        Returns
        -------
        `pd.DataFrame`
            Test assay data.
        """
        return self.__assay_data_te

    @property
    def pheno_data_tr(self) -> pd.DataFrame:
        """
        The phenotype data that corresponds to assay_data_tr.
        
        Returns
        -------
        `pd.DataFrame`
            Training phenotype data.
        """
        return self.__pheno_data_tr

    @property
    def pheno_data_te(self) -> pd.DataFrame:
        """
        The phenotype data that corresponds to assay_data_te.
        
        Returns
        -------
        `pd.DataFrame`
            Test phenotype data.
        """
        return self.__pheno_data_te

    @property
    def clin_data_tr(self) -> Union[pd.DataFrame, None]:
        """
        The clinical data that corresponds to pheno_data_tr. 
        
        Returns
        -------
        `pd.DataFrame`
            Training clinical data.
        `None`
            No training clinical data given.
        """
        return self.__clin_data_tr

    @clin_data_tr.deleter
    def clin_data_tr(self) -> None:
        self.__clin_data_tr = None
        self.__clin_data_te = None

    @property
    def clin_data_te(self) -> Union[pd.DataFrame, None]:
        """
        The clinical data that corresponds to pheno_data_te. 
        
        Returns
        -------
        `pd.DataFrame`
            Test clinical data.
        `None`
            No test clinical data given.
        """
        return self.__clin_data_te

    @clin_data_te.deleter
    def clin_data_te(self) -> None:
        self.__clin_data_tr = None
        self.__clin_data_te = None

    @property
    def n_tr_samples(self) -> int:
        """
        The number of assays present in assay_data_tr.

        Returns
        -------
        `int`
            Number of training samples.
        """
        return self.__assay_data_tr.shape[1]

    @property
    def n_te_samples(self) -> int:
        """
        The number of assays present in assay_data_te.

        Returns
        -------
        `int`
            Number of test samples.
        """
        return self.__assay_data_te.shape[1]

    @property
    def n_tr_observations(self) -> int:                                                  
        """
        The number of unique subjects that have been assayed in the training partition.
        This is commonly referred to as the number of biological replicates (in the 
        training partition).

        In this case, n_tr_observations = n_tr_samples / n_timepoints.

        Returns
        -------
        `int`
            Number of biological replicates in the training partition.
        """
        return self.n_tr_samples // self.n_timepoints 

    @property
    def n_te_observations(self) -> int:
        """
        The number of unique subjects that have been assayed in the test partition.
        This is commonly referred to as the number of biological replicates (in the 
        test partition).

        In this case, n_te_observations = n_te_samples / n_timepoints.

        Returns
        -------
        `int`
            Number of biological replicates in the test partition.
        """
        return self.n_te_samples // self.n_timepoints 

    @property
    def tr_owners(self) -> pd.Series:
        """
        The column of owners in pheno_data_tr.

        Returns
        -------
        `pd.Series`
            Training owners.
        """
        return self.__pheno_data_tr.loc[:, self.owner]

    @property
    def te_owners(self) -> pd.Series:
        """
        The column of owners in pheno_data_te.

        Returns
        -------
        `pd.Series`
            Test owners.
        """
        return self.__pheno_data_te.loc[:, self.owner]

    @property
    def tr_times(self) -> Union[pd.Series, None]:
        """
        The column of times in pheno_data_tr.

        Returns
        -------
        `pd.Series`
            Training times.
        `None`
            No time given.
        """
        if self.time is None: return None
        return self.__pheno_data_tr.loc[:, self.time].cat.remove_unused_categories()

    @property
    def te_times(self) -> Union[pd.Series, None]:
        """
        The column of times in pheno_data_te.

        Returns
        -------
        `pd.Series`
            Test times.
        `None`
            No time given.
        """
        if self.time is None: return None
        return self.__pheno_data_te.loc[:, self.time].cat.remove_unused_categories()

    @property
    def tr_labels(self) -> pd.Series:
        """
        The column of labels in pheno_data_tr.

        Returns
        -------
        `pd.Series`
            Training labels.
        """
        return self.__pheno_data_tr.loc[:, self.label].cat.remove_unused_categories()

    @tr_labels.setter
    def tr_labels(self, new_tr_labels : np.ndarray):
        self.__pheno_data_tr[self.label] = pd.Categorical(new_tr_labels)

    @property
    def te_labels(self) -> pd.Series:
        """
        The column of labels in pheno_data_te.

        Returns
        -------
        `pd.Series`
            Test labels.
        """
        return self.__pheno_data_te.loc[:, self.label].cat.remove_unused_categories()

    @te_labels.setter
    def te_labels(self, new_te_labels : np.ndarray):
        self.__pheno_data_te[self.label] = pd.Categorical(new_te_labels)

    @property
    def scalers(self) -> Union[dict, None]:
        """
        The resulting scalers fitted to the assay_data following the use of the `scale()` method.
        If the scale() method has not been invoked, this property will be None.

        Returns
        -------
        `dict`
            Key: cohort name. \n
            Value: sklearn StandardScaler that was fitted to the data belonging to key.
        `None`
            scale() has not been invoked.
        """
        if hasattr(self, "_EsetPrepSplit__scalers"):
            return self.__scalers
        else:
            return None

    def swap(self, n_tr_swap : int, n_te_swap : int, shuffle : bool = True, seed : int = 9999) -> EsetPrepSplit:
        """
        Swap observations between the training and test partitions.

        Parameters
        ----------
        `n_tr_swap`
            The number of observations to move from the training partition to the test partition.
        `n_te_swap`
            The number of observations to move from the test partition to the training partition.
        `shuffle`
            Prior to selecting observations to be swapped, should they be shuffled?
        `seed`
            Integer used to seed the random number generator.

        Returns
        -------
        `EsetPrepSplit`
            A modified EsetPrepSplit instance with the same observations but different training 
            and test partition sizes.

        Raises
        ------
        ValueError:
            1) If `n_tr_swap` is either < 0 or > n_tr_observations \n
            2) If `n_te_swap` is either < 0 or > n_te_observations
        """
        if (n_tr_swap > self.n_tr_observations) or (n_tr_swap < 0):
            raise ValueError(f"Cannot swap {n_tr_swap} of {self.n_tr_observations} training observations.")
        if (n_te_swap > self.n_te_observations) or (n_te_swap < 0):
            raise ValueError(f"Cannot swap {n_te_swap} of {self.n_te_observations} test observations.")

        rng = np.random.default_rng(seed)

        tr_ind_c = rng.choice(
            a = self.n_tr_observations,
            size = n_tr_swap,
            replace = False,
            shuffle = shuffle
            )
        te_ind_c = rng.choice(
            a = self.n_te_observations,
            size = n_te_swap,
            replace = False,
            shuffle = shuffle
            )

        # Generate expanded indices for assay and pheno data.
        tr_ind_e, te_ind_e = self._expand_indices(tr_ind_c, te_ind_c)

        assay_tr_swap = self.assay_data_tr.iloc[:, tr_ind_e] 
        assay_te_swap = self.assay_data_te.iloc[:, te_ind_e] 
        pheno_tr_swap = self.pheno_data_tr.iloc[tr_ind_e]
        pheno_te_swap = self.pheno_data_te.iloc[te_ind_e]
        if self.clin_data_tr is not None:
            # Use compact indices for clinical data (always only 1 row per owner).
            clin_tr_swap = self.clin_data_tr.iloc[tr_ind_c] 
            clin_te_swap = self.clin_data_te.iloc[te_ind_c]
        else:
            clin_tr_swap = clin_te_swap = None
        
        assay_tr_keep = self.assay_data_tr.drop( columns = assay_tr_swap.columns.to_numpy() )
        assay_te_keep = self.assay_data_te.drop( columns = assay_te_swap.columns.to_numpy() )
        pheno_tr_keep = self.pheno_data_tr.drop( index = pheno_tr_swap.index.to_numpy() )
        pheno_te_keep = self.pheno_data_te.drop( index = pheno_te_swap.index.to_numpy() )
        if clin_tr_swap is not None:
            clin_tr_keep = self.clin_data_tr.drop( index = clin_tr_swap.index.to_numpy() )
            clin_te_keep = self.clin_data_te.drop( index = clin_te_swap.index.to_numpy() )
        else:
            clin_tr_keep = clin_te_keep = None

        return EsetPrepSplit(
            assay_data_tr = pd.concat( (assay_tr_keep, assay_te_swap), axis = 1 ),
            assay_data_te = pd.concat( (assay_te_keep, assay_tr_swap), axis = 1 ),
            pheno_data_tr = pd.concat( (pheno_tr_keep, pheno_te_swap), axis = 0 ),
            pheno_data_te = pd.concat( (pheno_te_keep, pheno_tr_swap), axis = 0 ),
            clin_data_tr = pd.concat( (clin_tr_keep, clin_te_swap), axis = 0 ) if clin_tr_keep is not None else None, 
            clin_data_te = pd.concat( (clin_te_keep, clin_tr_swap), axis = 0 ) if clin_te_keep is not None else None,
            feature_data = self.feature_data,
            owner = self.owner, time = self.time, label = self.label,
            sort = False, validate = False
            )

    def scale(self, cohort_label : str | None, with_mean : bool, with_std : bool) -> EsetPrepSplit:

        if cohort_label is None:

            sc = StandardScaler(with_mean = with_mean, with_std = with_std)
            scalers = {}

            scaled_assay_tr = pd.DataFrame(
                data = ( sc.fit_transform(self.assay_data_tr.to_numpy().transpose()) ).transpose(),
                index = self.assay_data_tr.index,
                columns = self.assay_data_tr.columns
            )
            scalers["train"] = sc
            
            scaled_assay_te = pd.DataFrame(
                data = ( sc.fit_transform(self.assay_data_te.to_numpy().transpose()) ).transpose(),
                index = self.assay_data_te.index,
                columns = self.assay_data_te.columns
            )
            scalers["test"] = sc

            return EsetPrepSplit(
                scaled_assay_tr, scaled_assay_te,
                self.pheno_data_tr, self.pheno_data_te,
                self.clin_data_tr, self.clin_data_te,
                self.feature_data, 
                self.owner, self.time, self.label,
                sort = False, validate = False,
                scalers = scalers
                )

        if cohort_label not in self.pheno_data_tr.columns:
            raise ValueError(f"{cohort_label} is not in pheno_data.")

        sc = StandardScaler(with_mean = with_mean, with_std = with_std)
        scalers = {}

        cohort_names = self.pheno_data_tr.loc[:, cohort_label].unique()
        scaled_assay_tr_agg = scaled_assay_te_agg = None

        for cohort_name in cohort_names:

            temp_eset_ps = self.pquery(f"{cohort_label} == '{cohort_name}'")
            assay_tr = temp_eset_ps.assay_data_tr
            assay_te = temp_eset_ps.assay_data_te

            scaled_assay_tr = pd.DataFrame(
                data = ( sc.fit_transform(assay_tr.to_numpy().transpose()) ).transpose(),
                index = assay_tr.index,
                columns = assay_tr.columns
            )
            scaled_assay_te = pd.DataFrame(
                data = ( sc.transform(assay_te.to_numpy().transpose()) ).transpose(),
                index = assay_te.index,
                columns = assay_te.columns
            )
            scalers[cohort_name] = sc

            if scaled_assay_tr_agg is None:
                scaled_assay_tr_agg = scaled_assay_tr
                scaled_assay_te_agg = scaled_assay_te
            else:
                scaled_assay_tr_agg = pd.concat( (scaled_assay_tr_agg, scaled_assay_tr), axis = 1 )
                scaled_assay_te_agg = pd.concat( (scaled_assay_te_agg, scaled_assay_te), axis = 1 )

        return EsetPrepSplit(
            scaled_assay_tr_agg, scaled_assay_te_agg,
            self.pheno_data_tr, self.pheno_data_te,
            self.clin_data_tr, self.clin_data_te,
            self.feature_data, 
            self.owner, self.time, self.label,
            sort = False, validate = False,
            scalers = scalers
            )

    @classmethod
    def from_eset_prep_fulls(cls, eset_pf_tr : EsetPrepFull, eset_pf_te : EsetPrepFull) -> EsetPrepSplit:
        """
        Create an EsetPrepSplit instance from two EsetPrepFull instances.

        Parameters
        ---------
        `eset_pf_tr`
            An EsetPrepFull instance whose data will comprise the training partition of the newly
            created EsetPrepSplit instance.
        `eset_pf_te`
            An EsetPrepFull instance whose data will comprise the test partition of the newly
            created EsetPrepSplit instance.

        Returns
        -------
        `EsetPrepSplit`
            An EsetPrepSplit instance whose training partition is comprised of the data from 
            eset_pf_tr and whose test partition is comprised of the data from eset_pf_te.

        Raises
        ------
        UNDER CONSTRUCTION
        """
        if eset_pf_tr.time != eset_pf_te.time:
            raise ValueError(f"Train time '{eset_pf_tr.time}' != test time '{eset_pf_te.time}'.")
        train_tps = eset_pf_tr.n_timepoints ; test_tps = eset_pf_te.n_timepoints
        if train_tps != test_tps:
            raise ValueError(f"eset_pf_te has {test_tps} timepoints. Expected {train_tps}.")
        if eset_pf_tr.owner != eset_pf_te.owner:
            raise ValueError(f"Train owner '{eset_pf_tr.owner}' != test owner '{eset_pf_te.owner}'.")
        if eset_pf_tr.label != eset_pf_te.label:
            raise ValueError(f"Train label '{eset_pf_tr.label}' != test label '{eset_pf_te.label}'.")

        if eset_pf_tr.n_labels != eset_pf_te.n_labels:
            raise ValueError(f"There are {eset_pf_te.n_labels} test labels. Expected {eset_pf_tr.n_labels}.")
        train_lab_names = eset_pf_tr.label_names ; test_lab_names = eset_pf_te.label_names
        if (train_lab_names != test_lab_names).any():
            if len( np.intersect1d(train_lab_names, test_lab_names, assume_unique = True) ) == eset_pf_tr.n_labels:
                eset_pf_te.label_names = list(train_lab_names)
            else:
                raise ValueError(f"Test labels: {test_lab_names}. Expected {train_lab_names}.")

        if (eset_pf_tr.clin_data is None) and (eset_pf_te.clin_data is not None):
            raise ValueError("eset_pf_tr has no clinical data but eset_pf_te does.")
        elif (eset_pf_tr.clin_data is not None) and (eset_pf_te.clin_data is None):
            raise ValueError("eset_pf_te has no clinical data but eset_pf_tr does.")
        if eset_pf_tr.clin_data is not None:
            if ( eset_pf_tr.clin_data.columns.to_numpy() != eset_pf_te.clin_data.columns.to_numpy() ).any():
                raise ValueError("eset_pf_tr clin_data column names != eset_pf_te clin_data column names.")

        if (eset_pf_tr.feature_data is None) and (eset_pf_te.feature_data is not None):
            raise ValueError("eset_pf_tr has no feature data but eset_pf_te does.")
        elif (eset_pf_tr.feature_data is not None) and (eset_pf_te.feature_data is None):
            raise ValueError("eset_pf_te has no feature data but eset_pf_tr does.")
        if eset_pf_tr.feature_data is not None:
            if ( eset_pf_tr.feature_data.columns.to_numpy() != eset_pf_te.feature_data.columns.to_numpy() ).any():
                raise ValueError("eset_pf_tr feature_data column names != eset_pf_te feature_data column names.")

        return cls(
            eset_pf_tr.assay_data, eset_pf_te.assay_data, 
            eset_pf_tr.pheno_data, eset_pf_te.pheno_data, 
            eset_pf_tr.clin_data, eset_pf_te.clin_data,
            eset_pf_tr.feature_data,
            eset_pf_tr.owner, eset_pf_tr.time, eset_pf_tr.label,
            sort = False, validate = False
            )


class __EsetModelBase:
    """
    Abstract base class for derived classes EsetModelFull and EsetModelSplit. 
    """

    def __init__(self):
        
        if self.__class__.__name__ == "__EsetModelBase":
            raise AttributeError(f"__EsetModelBase cannot be instantiated.")

    def __repr__(self) -> str:
        
        if isinstance(self, EsetModelFull):
            mes = "Object: EsetModelFull \n"
            mes += f"Observations: {self.n_observations} \n"
            mes += f"Features: {self.n_features} \n"
            mes += "-------------------------------------- \n"
            mes += f"Labels | Counts: \n {self.y.value_counts()} \n"
            mes += "--------------------------------------\n"
            return mes

        if isinstance(self, EsetModelSplit):
            mes = "Object: EsetModelSplit \n"
            mes += f"Training Observations: {self.n_tr_observations} \n"
            mes += f"Test Observations: {self.n_te_observations} \n"
            mes += f"Features: {self.n_features} \n"
            mes += "-------------------------------------- \n"
            mes += "TRAINING SUMMARY \n"
            mes += "-------------------------------------- \n"
            mes += f"Labels | Counts: \n {self.y_tr.value_counts()} \n"
            mes += "-------------------------------------- \n"
            mes += "TEST SUMMARY \n"
            mes += "-------------------------------------- \n"
            mes += f"Labels | Counts: \n {self.y_te.value_counts()} \n"
            mes += "--------------------------------------\n"
            return mes

    def __getitem__(self, keys : Union[slice, np.ndarray]) -> Union[EsetModelFull, EsetModelSplit]:
        
        if isinstance(keys, slice):
            
            start, stop, step = _unpack_slice(keys)
            start = start if start is not None else 0
            stop = stop if stop is not None else self.n_features
            step = step if step is not None else 1

            if isinstance(self, EsetModelFull):
                return EsetModelFull(
                    self.X.iloc[start:stop:step],
                    self.y.iloc[start:stop:step]
                    )

            if isinstance(self, EsetModelSplit):
                return EsetModelSplit(
                    self.X_tr.iloc[start:stop:step],
                    self.X_te,
                    self.y_tr.iloc[start:stop:step], 
                    self.y_te
                    )
        
        if isinstance(keys, np.ndarray):

            if np.issubdtype(keys.dtype, np.integer):
                
                if isinstance(self, EsetModelFull):
                    return EsetModelFull(self.X.iloc[keys], self.y.iloc[keys])

                if isinstance(self, EsetModelSplit):
                    return EsetModelSplit(
                        self.X_tr.iloc[keys],
                        self.X_te,
                        self.y_tr.iloc[keys], 
                        self.y_te
                        )

            else:
                
                if isinstance(self, EsetModelFull):
                    return EsetModelFull(self.X.loc[keys], self.y.loc[keys])

                if isinstance(self, EsetModelSplit):
                    return EsetModelSplit(
                        self.X_tr.loc[keys],
                        self.X_te,
                        self.y_tr.loc[keys], 
                        self.y_te
                        )

        if isinstance(keys, tuple):
            if len(keys) == 2:

                row_keys = keys[0] ; col_keys = keys[1]
                if isinstance(row_keys, slice):
                    start, stop, step = _unpack_slice(row_keys)
                    if (start is None) and (stop is None) and (step is None):
                        pass
                    else:
                        return self[start:stop:step][:, col_keys]
                else:
                    return self[row_keys][:, col_keys]

                if isinstance(col_keys, slice):
            
                    start, stop, step = _unpack_slice(col_keys)
                    start = start if start is not None else 0
                    stop = stop if stop is not None else self.n_features
                    step = step if step is not None else 1

                    if isinstance(self, EsetModelFull):
                        return EsetModelFull(
                            self.X.iloc[:, start:stop:step],
                            self.y
                            )

                    if isinstance(self, EsetModelSplit):
                        return EsetModelSplit(
                            self.X_tr.iloc[:, start:stop:step],
                            self.X_te.iloc[:, start:stop:step],
                            self.y_tr, self.y_te
                            )

                if isinstance(col_keys, np.ndarray):

                    if np.issubdtype(col_keys.dtype, np.integer):

                        if isinstance(self, EsetModelFull):
                            return EsetModelFull(self.X.iloc[:, col_keys], self.y)

                        if isinstance(self, EsetModelSplit):
                            return EsetModelSplit(
                                self.X_tr.iloc[:, col_keys],
                                self.X_te.iloc[:, col_keys],
                                self.y_tr, self.y_te
                                )

                    else:

                        if isinstance(self, EsetModelFull):
                            return EsetModelFull(self.X.loc[:, col_keys], self.y)

                        if isinstance(self, EsetModelSplit):
                            return EsetModelSplit(
                                self.X_tr.loc[:, col_keys],
                                self.X_te.loc[:, col_keys],
                                self.y_tr, self.y_te
                                )

        raise IndexingError("This method of indexing is not supported. Use slice and/or ndarray.")

    def to_feather(self, dir : str) -> None:
        """
        Store modelling data in the feather format.
        
        Parameters
        ----------
        UNDER CONSTRUCTION
        """

        if isinstance(self, EsetModelFull):
            _custom_write_feather(
                df = pd.concat(
                        (self.__y, self.__X),
                        axis = 1,
                        join = "inner"
                    ),
                dir = dir,
                fn = "eset_model_full_data"
                )
            return

        if isinstance(self, EsetModelSplit):
            _custom_write_feather(
                df = pd.concat(
                        (self.__y_tr, self.__X_tr),
                        axis = 1,
                        join = "inner"
                    ),
                dir = dir,
                fn = "eset_model_split_train"
                )
            _custom_write_feather(
                df = pd.concat(
                        (self.__y_te, self.__X_te),
                        axis = 1,
                        join = "inner"
                    ),
                dir = dir,
                fn = "eset_model_split_test"
                )

    def validate(self) -> None:
        # UNDER CONSTRUCTION.
        if isinstance(self, EsetModelFull):
            pass

        if isinstance(self, EsetModelSplit):
            pass


class EsetModelFull(__EsetModelBase):
    
    def __init__(self, X : pd.DataFrame, y : pd.Series):
        """
        Instantiate an expressionset model full object.

        Parameters
        ----------
        `X`
            Data to be used for modelling.
        `y`
            Corresponding labels to X.
        """
        super().__init__()
        self.__X = X
        self.__y = y

    @property
    def X(self) -> pd.DataFrame:
        """
        Data to be modelled.
        
        Returns
        -------
        `pd.DataFrame`
        """
        return self.__X

    @property
    def y(self) -> pd.Series:
        """
        Labels corresponding to X.
        
        Returns
        -------
        `pd.Series`
        """
        return self.__y

    @property
    def n_observations(self) -> int:
        """
        The number of rows in X.

        Returns
        -------
        `int`
            Number of biological replicates.
        """
        return self.__X.shape[0]

    @property
    def n_features(self) -> int:
        """
        The number of columns in X.

        Returns
        -------
        `int`
        """
        return self.__X.shape[1]

    @classmethod
    def from_eset_prep_full(cls, eset_pf : EsetPrepFull) -> EsetModelFull:
        """
        Create an EsetModelFull instance from an EsetPrepFull instance.

        If the EsetPrepFull instance has clinical data, it will be merged into the 
        modelling data (`X`) of the new EsetModelFull instance.

        If the EsetPrepFull instance has more than one timepoint, new engineered
        features will be included in the modelling data (`X`) of the new EsetModelFull 
        instance. These features will be the difference between the values of a probe
        at two adjacent timepoints.

        Parameters
        ---------
        `eset_pf`
            An EsetPrepFull instance whose data will be used to make the new EsetModelFull 
            instance.

        Returns
        -------
        `EsetModelFull`
        """
        X = eset_pf._assay_ttd()
        y = eset_pf._pheno_f()

        if eset_pf.clin_data is None:
            return cls(X, y)
        else:
            X = (
                pd.merge(
                    left = eset_pf.clin_data,
                    right = X,
                    how = "inner",
                    left_index = True,
                    right_index = True,
                    sort = False,
                    )
                )
            return cls(X, y)  


class EsetModelSplit(__EsetModelBase):

    def __init__(self,
        X_tr : pd.DataFrame, X_te : pd.DataFrame,
        y_tr : pd.Series, y_te : pd.Series,
        ):

        super().__init__()
        self.__X_tr = X_tr ; self.__X_te = X_te
        self.__y_tr = y_tr ; self.__y_te = y_te

    @property
    def X_tr(self) -> pd.DataFrame:
        return self.__X_tr

    @property
    def X_te(self) -> pd.DataFrame:
        return self.__X_te

    @property
    def y_tr(self) -> pd.Series:
        return self.__y_tr

    @property
    def y_te(self) -> pd.Series:
        return self.__y_te

    @property
    def n_tr_observations(self) -> int:
        return self.__X_tr.shape[0]

    @property
    def n_te_observations(self) -> int:
        return self.__X_te.shape[0]
        
    @property
    def n_features(self) -> int:
        return self.__X_tr.shape[1]

    @classmethod
    def from_eset_prep_split(cls, eset_ps : EsetPrepSplit) -> EsetModelSplit:
        """
        Converts data from prep layout to model layout:
        """

        X_tr = eset_ps._assay_ttd( scope = "train" )
        X_te = eset_ps._assay_ttd( scope = "test" )
        y_tr = eset_ps._pheno_f( scope = "train" )
        y_te = eset_ps._pheno_f( scope = "test" )

        if eset_ps.clin_data_tr is None:
            return cls(X_tr, X_te, y_tr, y_te)
        else:
            X_tr = (
                pd.merge(
                    left = eset_ps.clin_data_tr,
                    right = X_tr,
                    how = "inner",
                    left_index = True,
                    right_index = True,
                    sort = False,
                    )
                )
            X_te = (
                pd.merge(
                    left = eset_ps.clin_data_te,
                    right = X_te,
                    how = "inner",
                    left_index = True,
                    right_index = True,
                    sort = False,
                    )
                )
            return cls(X_tr, X_te, y_tr, y_te)

    def swap(self, n_tr_swap : int, n_te_swap : int, shuffle : bool = True, seed : int = 9999):
        
        if (n_tr_swap > self.n_tr_observations) or (n_tr_swap < 0):
            raise ValueError(f"Cannot swap {n_tr_swap} of {self.n_tr_observations} training observations.")
        if (n_te_swap > self.n_te_observations) or (n_te_swap < 0):
            raise ValueError(f"Cannot swap {n_te_swap} of {self.n_te_observations} test observations.")

        rng = np.random.default_rng(seed)

        tr_ind = rng.choice(
            a = self.n_tr_observations,
            size = n_tr_swap,
            replace = False,
            shuffle = shuffle
            )
        te_ind = rng.choice(
            a = self.n_te_observations,
            size = n_te_swap,
            replace = False,
            shuffle = shuffle
            )

        X_tr_swap = self.X_tr.iloc[tr_ind] ; X_te_swap = self.X_te.iloc[te_ind]
        y_tr_swap = self.y_tr.iloc[tr_ind] ; y_te_swap = self.y_te.iloc[te_ind] 
        X_tr_keep = self.X_tr.drop( index = X_tr_swap.index.to_numpy() )
        X_te_keep = self.X_te.drop( index = X_te_swap.index.to_numpy() )
        y_tr_keep = self.y_tr.drop( y_tr_swap.index.to_numpy() )
        y_te_keep = self.y_te.drop( y_te_swap.index.to_numpy() )
        return EsetModelSplit(
            X_tr = pd.concat( (X_tr_keep, X_te_swap), axis = 0 ),
            X_te = pd.concat( (X_te_keep, X_tr_swap), axis = 0 ),
            y_tr = pd.concat( (y_tr_keep, y_te_swap), axis = 0 ),
            y_te = pd.concat( (y_te_keep, y_tr_swap), axis = 0 )
        )
    



# TO DO
# Add validate() method to eset models. 
