# EDM MIR Analysis Repository  

This repository contains code, datasets, and results for **Music Information Retrieval (MIR) analysis** focusing on **Electronic Dance Music (EDM) and its subgenres**.

---

## Repository Structure  

```
├── dataset/                  # Datasets used in the project
│   ├── curated/              # Curated subset of tracks used in experiments
│   └── original/             # Full/original dataset of tracks
│
results/                      # Results generated from analysis and experiments
├── descriptive/              # Descriptive statistics
├── dt/                       # Decision Tree 
├── lda/                      # Linear Discriminant Analysis 
├── lr/                       # Logistic Regression 
└── pca/                      # Principal Component Analysis 
│
├── src/                      # Source code
│   ├── analysis/             # Scripts for statistical and MIR analysis
│   ├── data_collection/      # Tools and scripts for dataset building
│   ├── expert_validation/    # Validation experiments with human experts
│   └── feature_extraction/   # MIR feature extraction scripts
```

---

## Workflow  

1. **Data Collection (`src/data_collection/`)**  
   - Scripts to gather and preprocess EDM audio datasets.
     
2. **Feature Extraction (`src/feature_extraction/`)**  
   - Extracts MIR features (spectral, rhythmic, timbral, etc.).  
   - Features are computed for both the curated and full datasets.  

3. **Analysis (`src/analysis/`)**  
   - Performs statistical analysis, classification, and subgenre exploration.  

4. **Expert Validation (`src/expert_validation/`)**  
   - Framework for validating results from music experts.  

5. **Results (`results/`)**  
   - Organized into outputs for the full dataset, curated dataset, and PCA experiments.  

---
