import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix

class binary_classification_report:

    def __init__(self):
        self.__data = None
        self.__model_names = []
        self.__metrics = ["auroc", "sensitivity", "specificity", "ppv", "npv", "fpr", "fnr"]

    @property
    def full_report(self) -> pd.DataFrame:
        
        if self.__data is None:
            return None

        return pd.DataFrame(
            data = self.__data,
            columns = self.__metrics,
            index = pd.CategoricalIndex(self.__model_names)
        ).rename_axis("model")

    @property
    def aggregated_report(self) -> pd.DataFrame:

        full_report = self.full_report

        if full_report is None:
            return None

        sum_report = full_report.groupby("model", sort = False).mean()
        sum_report.insert(
            loc = 0,
            column = "replicates",
            value = full_report.index.value_counts(sort = False) 
        )

        return sum_report

    def record(self, model_name : str, y_true : np.ndarray, y_pred : np.ndarray) -> None:

        scores = np.empty(shape = (7,), dtype = np.float32)

        scores[0] = roc_auc_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        # Sensitivity.
        scores[1] = tp / (tp + fn)
        # Specificity.
        scores[2] = tn / (tn + fp)
        # PPV.
        scores[3] = tp / (tp + fp)
        # NPV.
        scores[4] = tn / (tn + fn)
        # FPR.
        scores[5] = fp / (fp + tn)
        # FNR.
        scores[6] = fn / (fn + tp)

        self.__model_names.append(model_name)
        if self.__data is None:
            self.__data = scores.reshape(1, 7)
        else:
            self.__data = np.vstack( (self.__data, scores) )

    def to_csv(self, path : str = "auto") -> None:

        path = os.path.join(os.getcwd(), "report.csv") if path == "auto" else path
        self.full_report.to_csv(path)