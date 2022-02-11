# Slater Lab - Spontaneous Preterm Birth Investigation

* Library Code
* [Affymetrix Study](How-To-Reproduce-Our-Affymetrix-Results)
* [Bulk RNAseq Study](How-To-Reproduce-Our-Bulk-RNASeq-Results)

---

## How To Reproduce Our Affymetrix Results

1. Acquire the data that was used in the [*Heng et al.* investigation](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0155191) (*Heng data*):
    > Follow the code in `acquire_data.R`.

    * Clean the *Heng data* for preprocessing and modelling:
    
        > Follow the code in `./Heng/clean_Heng_geoData.py`.

<br/>

2. Acquire the data that was used in the [*Tarca et al.* investigation](https://www.cell.com/cell-reports-medicine/pdfExtended/S2666-3791(21)00166-X) (*Tarca data*):
    > Follow the code in `acquire_data.R`.

    * Clean the *Tarca data* for preprocessing and modelling:
    
        > Follow the code in `./Tarca/clean_Tarca_geoData.py`.

<br/>

3. Preprocess and model the data:

    > Follow the code in `affy_model.py`.

<br/>
<br/>

UNDER CONSTRUCTION

---

## How To Reproduce Our Bulk RNASeq Results

1. Acquire our bulk RNASeq data by reaching out to [Dr. Donna Slater](mailto:dmslater@ucalgary.ca):
    
    * Clean our bulk RNASeq data for preprocessing:

        > Follow the code in `./Bulk_RNASeq/data_cleaning.py`.

<br/>
<br/>

UNDER CONSTRUCTION
