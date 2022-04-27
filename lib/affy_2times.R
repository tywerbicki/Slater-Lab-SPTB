library(arrow, warn.conflicts = FALSE) # Read feather data format.
library(Biobase) # ExpressionSet class.
library(statmod) # duplicateCorrelation method in limma. 
# USER GUIDE: https://www.bioconductor.org/packages/devel/bioc/vignettes/limma/inst/doc/usersguide.pdf
library(limma)

DE_DEPOSIT_PATH = "/home/ty.werbicki/output/DE_Heng_data"
setwd(DE_DEPOSIT_PATH)

# Read data in feather format that was written from pandas.
custom_read_feather_from_pd = function(fn) {
  data = read_feather(
    file = fn,
    as_data_frame = TRUE
  )
  rownames(data) = data$index
  return( data[, 2:ncol(data)] )
}

# Build ExpressionSet from scratch:
#https://www.bioconductor.org/packages/devel/bioc/vignettes/Biobase/inst/doc/ExpressionSetIntroduction.pdf
assay_data = custom_read_feather_from_pd("assay_data_de")
assay_data = as.matrix(assay_data)

pheno_data = custom_read_feather_from_pd("pheno_data_de")
pheno_data = new("AnnotatedDataFrame", data = data.frame(pheno_data))

# Feature data only has 1 column, so have to use a work-around.
feature_data = read_feather(
  file = "feature_data_de",
  as_data_frame = TRUE
)
feature_data = data.frame(
  feature_data$ENTREZ_GENE_ID,
  row.names = feature_data$index
)
colnames(feature_data) = "ENTREZ_GENE_ID"
feature_data = new("AnnotatedDataFrame", data = feature_data)

eset = ExpressionSet(
  assayData = assay_data,
  phenoData = pheno_data,
  featureData = feature_data
)

print("ExpressionSet assembled successfully.")

# Model condition & timepoint.
treatments = factor(paste(eset$birth_outcome, eset$timepoint, sep = "."))
design = model.matrix(~ 0 + treatments)
colnames(design) = levels(treatments)

# Calculate within-subject correlation.
# Must do this because we have repeated measures.
corfit = duplicateCorrelation(eset, design, block = eset$patient_id) 

fit = lmFit(eset, design, block = eset$patient_id, correlation = corfit$consensus.correlation)

contrast = makeContrasts(
  dif_T1vsT2 = (term.T2 - term.T1) - (sptb.T2 - sptb.T1),
  T1_Term_vs_SPTB = term.T1 - sptb.T1,
  T2_Term_vs_SPTB = term.T2 - sptb.T2,
  Time_Term = term.T2 - term.T1,
  Time_SPTB = sptb.T2 - sptb.T1,
  levels = design
)

fit2 = contrasts.fit(fit, contrast)
fit2 = eBayes(fit2)

de_genes = c()
p_vals = c(0.01, 0.05, 0.1, 0.2, 0.5, 0.9)
i = 0

while (length(de_genes) < 3) {
  
  i = i+1
  # Get table of genes with at least 1 significant contrast.
  tt = topTable(
    fit = fit2,
    number = 500,
    adjust.method = "BH",
    p.value = p_vals[i]
  )
  # Get list of DE genes.
  de_genes = rownames(tt)
  
}

cat("p-value of", p_vals[i], "gave", length(de_genes), "DE genes.", sep = " ")
# Write list of DE genes to text file.
cat(de_genes, file = "de_genes.txt", sep = ";")
