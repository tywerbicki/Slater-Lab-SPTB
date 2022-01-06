# Retrieve GEO packages.

BiocManager::install("GEOquery")
library(GEOquery)

# For Heng data.
geo_data = getGEO( 
  GEO = "GSE59491",
  GSEMatrix = TRUE
  )
  
# For Tarca data.
geo_data = getGEO( 
  GEO = "GSE149440",
  GSEMatrix = TRUE
)

geo_data = geo_data[[1]]

# Extract assay data.
write.exprs(geo_data, "exprs_r.txt")
# Extract pheno data.
write.csv(pData(geo_data), "pheno_data_r.csv")
# Extract feature data.
write.csv(fData(geo_data), "feature_data_r.csv")















