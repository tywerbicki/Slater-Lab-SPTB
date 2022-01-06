import os
import pandas as pd
import numpy as np
import pyarrow.feather as feather

CWD = os.getcwd()
GEO_DATA_PATH = os.path.join(CWD, "Raw_Data")
if not os.path.isdir(GEO_DATA_PATH):
    raise FileNotFoundError("GEO data directory not found.")
DEPOSIT_PATH = os.path.join(CWD, "Processed_Data")
if not os.path.isdir(DEPOSIT_PATH):
    os.mkdir(DEPOSIT_PATH)

# Phenotype Data Exploration
pheno_data = pd.read_csv(
    os.path.join(GEO_DATA_PATH, "pheno_data_r.csv"),
    header = 0,
    index_col = 0
)
pheno_data.columns = (pheno_data.columns
    .str.strip()
    .str.replace(" ", "_", regex = False)
    .str.replace(":", "_", regex = False)
)
pheno_data.index = (pheno_data.index
    .str.strip()
)
pheno_columns_to_grab = [
    "individual_id_ch1",
    "sample_collected_at_time_pt_(17-23weeks_1,_27-33weeks_2)_ch1",
    "gest_age_at_sampling_(weeks)_ch1",
    "birth_outcome_(sptb_1,_term_0)_ch1"
]
pheno_types = {
    pheno_columns_to_grab[0] : "int32",
    pheno_columns_to_grab[1] : "category",
    pheno_columns_to_grab[2] : "int16",
    pheno_columns_to_grab[3] : "category"
}
pheno_new_col_names = {
    pheno_columns_to_grab[0] : "patient_id", 
    pheno_columns_to_grab[1] : "timepoint",
    pheno_columns_to_grab[2] : "gest_age", 
    pheno_columns_to_grab[3] : "birth_outcome"
}

pheno_data = (
    pheno_data
    # Grab desired columns.
    [pheno_columns_to_grab]
    # Cast columns to appropriate types.
    .astype(pheno_types)
    # Rename columns.
    .rename(columns = pheno_new_col_names)
)

def insert_dif(df):

    gest_age = df["gest_age"].to_numpy()
    dif = gest_age[1::2] - gest_age[::2]
    dif = np.repeat(dif, 2).astype(np.float32)
    df.insert(loc = 3, column = "dif_gest", value = dif)
    return df

pheno_data = (
    pheno_data
    # Remove samples with only 1 array.
    .groupby("patient_id").filter(lambda x: len(x) > 1)
    # Rename categories.
    .assign(
        timepoint = pheno_data.timepoint.cat.rename_categories({1 : "T1", 2 : "T2"}),
        birth_outcome = pheno_data.birth_outcome.cat.rename_categories({0 : "term", 1 : "sptb"})
    )
    .sort_values(by = ["patient_id", "timepoint"])
    # Acquire time difference between samples.
    .pipe(insert_dif)
)
pheno_row_names = pheno_data.index.values

# Expression Data Exploration
exprs_data = pd.read_csv(
    os.path.join(GEO_DATA_PATH, "exprs_r.txt"),
    sep = "\t",
    header = 0,
    index_col = 0
).astype("float32")

exprs_data.columns = (exprs_data.columns
    .str.strip()
)
exprs_data.index = (exprs_data.index
    .str.strip()
)

exprs_data = exprs_data.loc[:, pheno_row_names]

# Feature Names Exploration
feature_data = pd.read_csv(
    os.path.join(GEO_DATA_PATH, "feature_data_r.csv"),
    header = 0,
    index_col = 0
)

feature_data.index = (feature_data.index
    .str.strip()
)

feature_data = feature_data[["ENTREZ_GENE_ID"]]

# Deposit Data.
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
