import os
import pyarrow.feather as feather
import pandas as pd
import numpy as np

# Data wrangling (any operations that could change
# the number of or orientation of rows of the data frame).
meta_data = (
    pd.read_excel(
        io = os.path.join(os.getcwd(), "metadata_AOF_RNAseq.xlsx"),
        header = 0
    )
    .drop(["run.name", "strat.group"], axis = 1)
    .rename({
            "run.no":"sample",
            "study.id":"studyid",
            "group":"outcome"
         }, axis = 1)
    .astype({
            "sample":"str",
            "studyid":"category",
            "timepoint":"category",
            "outcome":"category"
        })
    # Remove medically indicated births.
    .query("outcome != 'MI'")
    # Remove observations with only one sample (want 2: T1 and T2).
    .groupby("studyid").filter(lambda x: len(x) == 2)
    # Make sure there aren't any repeated data.
    .drop_duplicates(subset = ["sample"])
    # Orient rows appropriately.
    .sort_values(by = ["studyid", "timepoint"])
)

# Data cleaning (no operations that will change the number
# of nor orientation of the rows of the data frame).
meta_data = (
    meta_data
    # Reformat variables.
    .assign(
        outcome = meta_data["outcome"]
            .cat.remove_unused_categories()
            .cat.rename_categories({"sPTB":"sptb"}),
        studyid = meta_data["studyid"].cat.remove_unused_categories(),
        sample = "sample_" + meta_data["sample"] + "_id",
        # Add a "nested individual column" for DESeq2.
        ind_n = pd.Categorical( np.repeat( range( len(meta_data) // 2 ), 2 ) )
    )
    .set_index("sample")
)


read_data = (
    feather.read_feather(source = os.path.join(os.getcwd(), "read_data_agg") )
    .set_index("ID")
)

read_data.columns = "sample_" + (
    read_data.columns
    .str.replace("-new", "")
    .str.replace("b", "")
) + "_id"

# Only select reads for samples that remain in the metadata.
read_data = read_data.loc[:, meta_data.index.to_numpy() ]

# Sanity check.
assert all(meta_data.index == read_data.columns) 

feather.write_feather(
    df = read_data.reset_index(),
    dest = "read_data_de",
    compression = "lz4"
)
feather.write_feather(
    df = meta_data.reset_index(),
    dest = "meta_data_de",
    compression = "lz4"
)
