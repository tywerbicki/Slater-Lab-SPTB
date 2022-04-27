import os
import pyarrow.feather as feather
import numpy as np
import py_expressionset as pe
import reporting as re
from feature_selection import Stepwise_Additive_Selector
from sklearn.linear_model import LogisticRegression
import affymetrix_MLP as amlp
from ray import tune

# User-defined constants.
CWD = os.getcwd()
HENG_PROCESSED_DATA_PATH = "/home/ty.werbicki/data/Processed_Heng_Data"
TARCA_PROCESSED_DATA_PATH = "/home/ty.werbicki/data/Processed_Tarca_Data"
R_SCRIPT_NAME = os.path.join(CWD, "affy_2times.R")
DE_DEPOSIT_PATH = "/home/ty.werbicki/output/DE_Heng_data"
if not os.path.isdir(DE_DEPOSIT_PATH):
    os.mkdir(DE_DEPOSIT_PATH)
RESULTS_PATH = "/home/ty.werbicki/output"

N_CPUS = 40
# Do we want to randomly scramble the labels?
# This is often used as an exploratory excersise.
SCRAMBLE = False

# Read in Heng data.
assay_data = (
    feather.read_feather(os.path.join(HENG_PROCESSED_DATA_PATH, "assay_data_p"))
    .set_index("index")
    .rename_axis(None)
)
pheno_data = (
    feather.read_feather(os.path.join(HENG_PROCESSED_DATA_PATH, "pheno_data_p"))
    .set_index("index")
    .rename_axis(None)
)
feature_data = (
    feather.read_feather(os.path.join(HENG_PROCESSED_DATA_PATH, "feature_data_p"))
    .set_index("index")
    .rename_axis(None)
)

# Create full expressionset prep object.
og_heng_eset_pf = pe.EsetPrepFull(
    assay_data = assay_data,
    pheno_data = pheno_data,
    clin_data = None,
    feature_data = feature_data,
    owner = "patient_id",
    time = "timepoint",
    label = "birth_outcome"
)
# Validate data.
og_heng_eset_pf.validate()

# Define feature selection method.
sas = Stepwise_Additive_Selector(
    max_retained = 30,
    n_splits = 5,
    n_repeats = 2,
    n_processes = N_CPUS*2
)

# Define LR model to be assessed.
lr = LogisticRegression(
    penalty = "none",
    fit_intercept = True,
    tol = 1e-6,
    C = 100000,
    random_state = 9999,
    solver = "newton-cg",
    max_iter = 5000
)

# Define MLP, its config for hyper opt, and additional training parameters.
mlp = amlp.MLP_Binary_Classifier()

config = {
    # Number of neurons in 1st hidden layer.
    "l1" : tune.sample_from(lambda _: 2 ** np.random.randint(5, 9)),
    # Number of neurons in 2nd hidden layer.
    "l2" : tune.sample_from(lambda _: 2 ** np.random.randint(3, 6)),
    # Probability of dropout on the 1st hidden layer.
    "p1" : tune.uniform(0.2, 0.6),
    # Probability of dropout on the 2nd hidden layer.
    "p2" : tune.uniform(0.01, 0.3),
    # Learning rate: uniform on the logarithmic scale.
    "lr" : tune.loguniform(1e-5, 1e-3),
    # l2 regularization of the weights.
    "weight_decay" : tune.loguniform(1e-4, 1e-1),
    # Tune still treats this as continuous.
    "batch_size" : tune.choice([4, 8, 12, 16])
}

MAX_EPOCHS = 250
REDUCTION_FACTOR = 4
N_HYPER_SAMPLES = 500
VERBOSITY = 0

N_SPLITS = 5
N_REPEATS = 2
TEST_SIZE = 0.25

# Object to store test results.
bcr = re.binary_classification_report()

if SCRAMBLE:
    # Randomly scramble the labels.
    og_heng_eset_pf.scramble(inplace = True, seed = 9999)

# Extract modelling column names for feature selector.
model_feature_names = og_heng_eset_pf.extract_model_feature_names().to_numpy()
sas.set_feature_names(model_feature_names)

# Split full expressionset prep object into split expresionset_prep objects. 
for eset_ps in og_heng_eset_pf.split(n_splits = N_SPLITS, n_repeats = N_REPEATS, test_size = TEST_SIZE, seed = 9999):

    # Remove bad probes.
    eset_ps = eset_ps.screen_probes(0.001, 3, threshold = 5)
    # Deposit data for DE analysis.
    eset_ps.de_to_feather(DE_DEPOSIT_PATH)
    # Run limma analysis in R.
    eset_ps.perform_dea_r(
        R_SCRIPT_NAME, 
        os.path.join(DE_DEPOSIT_PATH, "de_genes.txt")
        )
    # Remove genes that aren't DE (NDE).
    eset_ps = eset_ps.remove_nde()
    # Transform eset data for modelling.
    eset_ms = pe.EsetModelSplit.from_eset_prep_split(eset_ps)

    # Acquire indices of features to use.
    sas.fit(eset_ms.X_tr, eset_ms.y_tr)
    print(sas.fold_best_scores)
    # Select features.
    eset_ms = eset_ms[:, sas.fold_best_indices]

    # Fit LR model.
    lr.fit(eset_ms.X_tr, eset_ms.y_tr)
    # Record results.
    bcr.record("lr_Heng_train_local", eset_ms.y_tr.to_numpy(), lr.predict(eset_ms.X_tr))
    bcr.record("lr_Heng_test_local", eset_ms.y_te.to_numpy(), lr.predict(eset_ms.X_te))

    # Fit and train MLP.
    mlp.fit(eset_ms.X_tr, eset_ms.y_tr)
    mlp.train(
        config = config,
        max_epochs = MAX_EPOCHS,
        reduction_factor = REDUCTION_FACTOR,
        n_hyper_samples = N_HYPER_SAMPLES,
        n_cpus = N_CPUS,
        verbosity = VERBOSITY
    )
    # Record results.
    bcr.record("mlp_Heng_train_local", eset_ms.y_tr.to_numpy(), mlp.predict(eset_ms.X_tr))
    bcr.record("mlp_Heng_test_local", eset_ms.y_te.to_numpy(), mlp.predict(eset_ms.X_te))

# Store the best indices tree for the above analysis.
sas.pickle_best_indices_tree( file_path = os.path.join(RESULTS_PATH, "Heng_local_scramble.pkl") )
# Reset the feature selector.
sas.reset()

##### Repeat process for Tarca data. #####

tarca_assay_data = (
    feather.read_feather(os.path.join(TARCA_PROCESSED_DATA_PATH, "assay_data_p"))
    .set_index("index")
    .rename_axis(None)
)
tarca_pheno_data = (
    feather.read_feather(os.path.join(TARCA_PROCESSED_DATA_PATH, "pheno_data_p"))
    .set_index("index")
    .rename_axis(None)
)
tarca_feature_data = (
    feather.read_feather(os.path.join(TARCA_PROCESSED_DATA_PATH, "feature_data_p"))
    .set_index("index")
    .rename_axis(None)
)

og_tarca_eset_pf = pe.EsetPrepFull(
    assay_data = tarca_assay_data,
    pheno_data = tarca_pheno_data,
    clin_data = None,
    feature_data = tarca_feature_data,
    owner = "patient_id",
    time = "timepoint",
    label = "birth_outcome"
)

if SCRAMBLE:
    # Randomly scramble the labels.
    og_tarca_eset_pf.scramble(inplace = True, seed = 9999)

# Extract modelling column names for feature selector.
model_feature_names = og_tarca_eset_pf.extract_model_feature_names().to_numpy()
sas.set_feature_names(model_feature_names)

for eset_ps in og_tarca_eset_pf.split(n_splits = N_SPLITS, n_repeats = N_REPEATS, test_size = TEST_SIZE, seed = 9999):

    # Remove bad probes.
    eset_ps = eset_ps.screen_probes(0.001, 3, threshold = 5)
    # Deposit data for DE analysis.
    eset_ps.de_to_feather(DE_DEPOSIT_PATH)
    # Run limma analysis in R.
    eset_ps.perform_dea_r(
        R_SCRIPT_NAME, 
        os.path.join(DE_DEPOSIT_PATH, "de_genes.txt")
        )
    # Remove genes that aren't DE (NDE).
    eset_ps = eset_ps.remove_nde()
    # Transform eset data for modelling.
    eset_ms = pe.EsetModelSplit.from_eset_prep_split(eset_ps)

    # Acquire indices of features to use.
    sas.fit(eset_ms.X_tr, eset_ms.y_tr)
    print(sas.fold_best_scores)
    # Select features.
    eset_ms = eset_ms[:, sas.fold_best_indices]

    # Fit LR model.
    lr.fit(eset_ms.X_tr, eset_ms.y_tr)
    # Record results.
    bcr.record("lr_Tarca_train_local", eset_ms.y_tr.to_numpy(), lr.predict(eset_ms.X_tr))
    bcr.record("lr_Tarca_test_local", eset_ms.y_te.to_numpy(), lr.predict(eset_ms.X_te))

    # Fit and train MLP.
    mlp.fit(eset_ms.X_tr, eset_ms.y_tr)
    mlp.train(
        config = config,
        max_epochs = MAX_EPOCHS,
        reduction_factor = REDUCTION_FACTOR,
        n_hyper_samples = N_HYPER_SAMPLES,
        n_cpus = N_CPUS,
        verbosity = VERBOSITY
    )
    # Record results.
    bcr.record("mlp_Tarca_train_local", eset_ms.y_tr.to_numpy(), mlp.predict(eset_ms.X_tr))
    bcr.record("mlp_Tarca_test_local", eset_ms.y_te.to_numpy(), mlp.predict(eset_ms.X_te))

# Store the best indices tree for the above analysis.
sas.pickle_best_indices_tree( file_path = os.path.join(RESULTS_PATH, "Tarca_local_scramble.pkl") )
# Reset the feature selector.
sas.reset()


##### The following pipeline is flawed - performing for comparison only. #####
eset_pf_flawed = og_heng_eset_pf.screen_probes(0.001, 3, 5)
eset_pf_flawed.de_to_feather(DE_DEPOSIT_PATH)
eset_pf_flawed.perform_dea_r(R_SCRIPT_NAME, os.path.join(DE_DEPOSIT_PATH, "de_genes.txt"))
eset_pf_flawed = eset_pf_flawed.remove_nde()
model_feature_names = eset_pf_flawed.extract_model_feature_names().to_numpy()
sas.set_feature_names(model_feature_names)

for eset_ps_flawed in eset_pf_flawed.split(N_SPLITS, N_REPEATS, TEST_SIZE, 9999):
    eset_ms_flawed = pe.EsetModelSplit.from_eset_prep_split(eset_ps_flawed)
    sas.fit(eset_ms_flawed.X_tr, eset_ms_flawed.y_tr)
    print(sas.fold_best_scores)
    eset_ms_flawed = eset_ms_flawed[:, sas.fold_best_indices]
    lr.fit(eset_ms_flawed.X_tr, eset_ms_flawed.y_tr)
    bcr.record("lr_train_flawed", eset_ms_flawed.y_tr.to_numpy(), lr.predict(eset_ms_flawed.X_tr))
    bcr.record("lr_test_flawed", eset_ms_flawed.y_te.to_numpy(), lr.predict(eset_ms_flawed.X_te))
    mlp.fit(eset_ms_flawed.X_tr, eset_ms_flawed.y_tr)
    mlp.train(
        config = config,
        max_epochs = MAX_EPOCHS,
        reduction_factor = REDUCTION_FACTOR,
        n_hyper_samples = N_HYPER_SAMPLES,
        n_cpus = N_CPUS,
        verbosity = VERBOSITY
    )
    bcr.record("mlp_train_flawed", eset_ms_flawed.y_tr.to_numpy(), mlp.predict(eset_ms_flawed.X_tr))
    bcr.record("mlp_test_flawed", eset_ms_flawed.y_te.to_numpy(), mlp.predict(eset_ms_flawed.X_te))

sas.pickle_best_indices_tree( file_path = os.path.join(RESULTS_PATH, "Heng_flawed_scrambled.pkl") )
sas.reset()
##### End of flawed pipeline. #####


##### Train model on all of Heng data to predict Tarca data #####

heng_eset_pf = og_heng_eset_pf.screen_probes(0.001, 3, threshold = 5)
heng_eset_pf.de_to_feather(DE_DEPOSIT_PATH)
heng_eset_pf.perform_dea_r(
    R_SCRIPT_NAME, 
    os.path.join(DE_DEPOSIT_PATH, "de_genes.txt")
    )
heng_eset_pf = heng_eset_pf.remove_nde()
# Acquire the intersection of the Heng and Tarca data.
heng_eset_pf, tarca_eset_pf = heng_eset_pf.intersect(og_tarca_eset_pf)
# Use expressionset_prep full objects to create expressionset_prep split object.
eset_ps = pe.EsetPrepSplit.from_eset_prep_fulls(heng_eset_pf, tarca_eset_pf)
# Scale data individually by train and test.
scaled_eset_ps = eset_ps.scale(cohort_label = None, with_mean = True, with_std = False)
# Use expressionset_prep split object to create expressionset_model split object.
eset_ms = pe.EsetModelSplit.from_eset_prep_split(scaled_eset_ps)
# Acquire indices of features to use.
sas.fit(eset_ms.X_tr, eset_ms.y_tr)
print(sas.fold_best_scores)
# Select features.
eset_ms = eset_ms[:, sas.fold_best_indices]

lr.fit(eset_ms.X_tr, eset_ms.y_tr)
bcr.record("lr_train_on_Heng", eset_ms.y_tr.to_numpy(), lr.predict(eset_ms.X_tr))
bcr.record("lr_test_on_Tarca", eset_ms.y_te.to_numpy(), lr.predict(eset_ms.X_te))

mlp.fit(eset_ms.X_tr, eset_ms.y_tr)
mlp.train(
    config = config,
    max_epochs = MAX_EPOCHS,
    reduction_factor = REDUCTION_FACTOR,
    n_hyper_samples = N_HYPER_SAMPLES,
    n_cpus = N_CPUS,
    verbosity = VERBOSITY
)
bcr.record("mlp_train_on_Heng", eset_ms.y_tr.to_numpy(), mlp.predict(eset_ms.X_tr))
bcr.record("mlp_test_on_Tarca", eset_ms.y_te.to_numpy(), mlp.predict(eset_ms.X_te))

sas.reset()

##### Train model on a mix of Heng and Tarca data to predict a mix. #####

heng_eset_pf, tarca_eset_pf = og_heng_eset_pf.intersect(og_tarca_eset_pf)
combined_eset_ps_og = pe.EsetPrepSplit.from_eset_prep_fulls(heng_eset_pf, tarca_eset_pf)
model_feature_names = combined_eset_ps_og.extract_model_feature_names().to_numpy()
sas.set_feature_names(model_feature_names)

for i in range( N_SPLITS * N_REPEATS ):
    combined_eset_ps = combined_eset_ps_og.swap(40, 80, seed = i)
    combined_eset_ps = combined_eset_ps.screen_probes(0.001, 3)
    combined_eset_ps.de_to_feather(DE_DEPOSIT_PATH)
    combined_eset_ps.perform_dea_r(
        R_SCRIPT_NAME, 
        os.path.join(DE_DEPOSIT_PATH, "de_genes.txt")
        )
    combined_eset_ps = combined_eset_ps.remove_nde()
    # Scale data individually by cohort.
    scaled_comb_eset_ps = combined_eset_ps.scale(cohort_label = "cohort", with_mean = True, with_std = False)
    combined_eset_ms = pe.EsetModelSplit.from_eset_prep_split(scaled_comb_eset_ps)

    sas.fit(combined_eset_ms.X_tr, combined_eset_ms.y_tr)
    print(sas.fold_best_scores)
    combined_eset_ms = combined_eset_ms[:, sas.fold_best_indices]

    lr.fit(combined_eset_ms.X_tr, combined_eset_ms.y_tr)
    bcr.record("lr_train_combined", combined_eset_ms.y_tr.to_numpy(), lr.predict(combined_eset_ms.X_tr))
    bcr.record("lr_test_combined", combined_eset_ms.y_te.to_numpy(), lr.predict(combined_eset_ms.X_te))

    mlp.fit(eset_ms.X_tr, eset_ms.y_tr)
    mlp.train(
        config = config,
        max_epochs = MAX_EPOCHS,
        reduction_factor = REDUCTION_FACTOR,
        n_hyper_samples = N_HYPER_SAMPLES,
        n_cpus = N_CPUS,
        verbosity = VERBOSITY
    )
    bcr.record("mlp_train_combined", eset_ms.y_tr.to_numpy(), mlp.predict(eset_ms.X_tr))
    bcr.record("mlp_test_combined", eset_ms.y_te.to_numpy(), mlp.predict(eset_ms.X_te))

sas.pickle_best_indices_tree( file_path = os.path.join(RESULTS_PATH, "combined_scrambled.pkl") )

# Deposit all prediction results.
bcr.to_csv( os.path.join(RESULTS_PATH, "Heng_vs_Tarca_scrambled.csv") )
