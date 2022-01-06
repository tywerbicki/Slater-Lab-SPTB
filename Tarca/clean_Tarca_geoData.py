import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pyarrow.feather as feather

CWD = os.getcwd()
GEO_DATA_PATH = os.path.join(CWD, "Raw_Data")
if not os.path.isdir(GEO_DATA_PATH):
    raise FileNotFoundError("GEO data directory not found.")
DEPOSIT_PATH = os.path.join(CWD, "Processed_Data")
if not os.path.isdir(DEPOSIT_PATH):
    os.mkdir(DEPOSIT_PATH)

# Phenotype data clenaing.
pheno_data = pd.read_csv(
    os.path.join(GEO_DATA_PATH, "pheno_data_r.csv"),
    header = 0,
    index_col = 0
)

pheno_data.columns = (pheno_data.columns
    .str.strip()
    .str.replace(" ", "_")
    .str.replace(":", "_")
)

pheno_columns_to_grab = [
    "individual_ch1",
    "gadelivery_ch1",
    "gestational_age_ch1",
    "group_ch1"
]
types = {
    pheno_columns_to_grab[0] : "int16",
    pheno_columns_to_grab[1] : "float32",
    pheno_columns_to_grab[2] : "float32",
    pheno_columns_to_grab[3] : "category"
}
new_names = {
    pheno_columns_to_grab[0] : "patient_id",
    pheno_columns_to_grab[1] : "time_delivery",
    pheno_columns_to_grab[2] : "gest_age",
    pheno_columns_to_grab[3] : "birth_outcome"
}

# Heng collection timepoints.
T1 = (16, 24) ; T2 = (26, 34)
tp_bins = pd.IntervalIndex.from_tuples([T1, T2], closed = "both")

# Check if patient has an array at both timepoints.
def has_both_times(df):
    vec = df["gest_age"].values
    t1 = sum( (vec >= T1[0]) & (vec <= T1[1]) ) > 0
    t2 = sum( (vec >= T2[0]) & (vec <= T2[1]) ) > 0
    return t1 and t2

# Keep the array for each patient that is closest to the
# middle of each timepoint.
def keep_middle(df):

    tp = (
        pd.cut(df["gest_age"], tp_bins)
            .cat.rename_categories(["T1", "T2"])
    )
    fm = np.abs( (
        tp.cat.rename_categories([np.mean(T1), np.mean(T2)])
        .astype(np.float16)
        ) - df["gest_age"] 
    )
    min_dists = (
        df
        [["patient_id"]]
        .assign(timepoint = tp, from_middle = fm)
        .groupby(["patient_id", "timepoint"])
        ["from_middle"]
        .transform("min")
    )
    
    df.insert(loc = 1, column = "timepoint", value = tp)
    return (
        df
        .assign(from_middle = fm, min_dists = min_dists)
        .query("from_middle == min_dists")
        .drop(["from_middle", "min_dists"], axis = 1)
    )

# If there is a tie in the operation above, keep the array
# that was sampled later.
def keep_max_tp(df):
    
    max_per_tp = (
        df
        [["gest_age", "patient_id", "timepoint"]]
        .groupby(["patient_id", "timepoint"])
        ["gest_age"]
        .transform("max")
    )
    return (
        df
        .assign(max_per_tp = max_per_tp)
        .query("gest_age == max_per_tp")
        .drop(["max_per_tp"], axis = 1)
    )

# Insert a column that records the number of weeks 
# between the sampling of a patient's 2 arrays.
def insert_dif(df):

    gest_age = df["gest_age"].to_numpy()
    dif = gest_age[1::2] - gest_age[::2]
    dif = np.repeat(dif, 2).astype(np.float32)
    df.insert(loc = 3, column = "dif_gest", value = dif)
    return df

pheno_data = (
    pheno_data
    # Select columns.
    [pheno_columns_to_grab]
    # Assign proper types.
    .astype(types)
    # Rename columns.
    .rename(columns = new_names)
)

pheno_data = (
    pheno_data
    # Remove patients with induced SPTB.
    .query("birth_outcome != 'Early_Preeclampsia'")
    # Remove arrays acquired at or after delivery.
    .query("gest_age < time_delivery")
    # Remove time_delivery column.
    .drop(["time_delivery"], axis = 1)
    # Only keep those with probes at both timepoints.
    .groupby("patient_id").filter(lambda x: has_both_times(x))
    # Only keep samples closest to middle of timepoints.
    .pipe(keep_middle)
    # If there is a tie above, keep the sample that is later.
    .pipe(keep_max_tp)
    # Ensure patient_id is consecutive.
    .sort_values(by = ["patient_id", "gest_age"])
    # Acquire time difference between samples.
    .pipe(insert_dif)
)

pheno_data = (
    pheno_data
    # Remove unused categories.
    .assign(birth_outcome = pheno_data.birth_outcome.cat.remove_unused_categories())
    # Re-assign categories.
    .assign(birth_outcome = (
        pheno_data.birth_outcome
        .map({"Control" : "term", "PPROM" : "sptb", "sPTD" : "sptb"})
        .astype("category")
        .cat.reorder_categories(["term", "sptb"])
        )
    )
)

pheno_row_names = pheno_data.index.values

# Expression data cleaning.
exprs_data = pd.read_csv(
    os.path.join(GEO_DATA_PATH, "exprs_r.txt"),
    sep = "\t",
    header = 0,
    index_col = 0
).astype("float32")

# Select columns that correspond with pheno row names.
exprs_data = exprs_data[pheno_row_names]

# Feature data cleaning.
feature_data = pd.read_csv(
    os.path.join(GEO_DATA_PATH, "feature_data_r.csv"),
    header = 0,
    index_col = 0
)

feature_data = (
    feature_data
    # Drop columns.
    .drop(["ID", "Description", "SPOT_ID"], axis = 1)
    # Fill Nans and convert to int.
    .assign(ENTREZ_GENE_ID = feature_data.ENTREZ_GENE_ID.fillna(0).astype("int32"))
)

# Deposit data.
assert (exprs_data.columns.values == pheno_data.index.values).all()
assert (exprs_data.index.values == feature_data.index.values).all()

feather.write_feather(
    df = exprs_data.reset_index(),
    dest = os.path.join(DEPOSIT_PATH, "assay_data_p"),
    compression = "lz4"
)
feather.write_feather(
    df = pheno_data.reset_index(),
    dest = os.path.join(DEPOSIT_PATH, "pheno_data_p"),
    compression = "lz4"
)
feather.write_feather(
    df = feature_data.reset_index(),
    dest = os.path.join(DEPOSIT_PATH, "feature_data_p"),
    compression = "lz4"
)
