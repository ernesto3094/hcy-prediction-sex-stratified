# Sex-stratified prediction of homocysteine levels using regression and machine learning models

This repository contains the R scripts used to implement the analytical workflow for the sex-stratified prediction of continuous homocysteine (HCY) levels in young adults.

The workflow includes data preprocessing, preliminary diagnostic assessment, variable transformation, Elastic Net-based variable selection, construction of stability-based predictor groups, model training, learning-curve analysis, and independent test-set evaluation.

## Study overview

Homocysteine (HCY) is a biomarker associated with cardiovascular, metabolic, and physiological alterations. This project evaluates different predictive modeling strategies to determine which approaches better explain continuous HCY variability in women and men using anthropometric, biochemical, and hematological predictors.

The analysis was performed separately by sex to evaluate potential differences in predictive behavior between women and men.

## Modeling strategies

Two complementary modeling workflows were evaluated.

### 1. Transformed and regularized workflow

This workflow was applied to:

* Multiple Linear Regression (MLR)
* Polynomial Regression (PR)
* Support Vector Regression with linear kernel (SVR-L)
* Support Vector Regression with polynomial kernel (SVR-P)
* Support Vector Regression with radial basis function kernel (SVR-RBF)

It included:

* Preliminary regression diagnostics
* Variable transformations based on skewness and kurtosis
* Centering and scaling when required
* Elastic Net-based variable selection
* Repeated-seed assessment of predictor-selection stability
* Construction of stability-based predictor groups
* Model tuning using 10-fold cross-validation within the training set
* Final evaluation on an independent test set

### 2. Tree-based ensemble workflow

This workflow was applied to:

* Random Forest (RF)
* Extreme Gradient Boosting (XGBoost)

These models were trained using the full set of eligible untransformed predictors, without prior Elastic Net-based variable selection. This strategy allowed tree-based ensemble models to be evaluated as a parallel workflow applied directly to the original predictor space.

## General workflow

The analytical procedure followed these main steps:

1. Load the sex-specific datasets.
2. Remove variables not aligned with the predictive objective.
3. Perform preliminary regression diagnostics.
4. Apply variable transformations according to skewness and kurtosis criteria.
5. Split the data into training and test sets using HCY-stratified partitioning.
6. Apply Elastic Net-based variable selection within the transformed workflow.
7. Repeat Elastic Net selection across multiple random seeds to assess predictor-selection stability.
8. Construct stability-based predictor groups.
9. Train MLR, PR, and SVR models using 10-fold cross-validation within the training set.
10. Train RF and XGBoost models using untransformed predictors.
11. Evaluate final model performance on the independent test set using R², RMSE, and MAE.
12. Generate learning curves to support the interpretation of model training behavior and generalization.

## Repository structure

```text
.
├── scripts/
│   ├── women/
│   └── men/
│
├── results/
│   ├── women/
│   └── men/
│
├── data/
│   └── README.md
│
├── README.md
├── .gitignore
└── requirements-r.txt
```

## Data availability

The original dataset is not included in this repository due to confidentiality restrictions and authorization requirements associated with the original study.

The scripts are provided to document the analytical workflow and support reproducibility of the preprocessing, modeling, and evaluation procedures described in the manuscript.

Researchers interested in the dataset should contact the corresponding author, subject to approval from the data owner and applicable confidentiality conditions.

## Code availability

The R scripts used to implement the main analytical procedures are available in this repository:

https://github.com/ernesto3094/hcy-prediction-sex-stratified

## Required R packages

The analysis was implemented in R. The main packages used include:

* readr
* dplyr
* tidyr
* ggplot2
* caret
* glmnet
* e1071
* MASS
* car
* lmtest
* nortest
* randomForest
* xgboost
* tidyverse

Depending on the specific script, additional packages may be required. The main package list is also provided in `requirements-r.txt`.

## Reproducibility

The modeling scripts use fixed random seeds where applicable to support reproducibility. In addition, Elastic Net-based variable selection was repeated across multiple random seeds to assess the stability of retained predictors.

The stability-based predictor groups were defined according to repeated-seed selection frequency:

* Stable-100: predictors selected in 10 of 10 runs
* Stable-90: predictors selected in at least 9 of 10 runs
* Stable-80: predictors selected in at least 8 of 10 runs
* Stable-70: predictors selected in at least 7 of 10 runs
* Stable-60: predictors selected in at least 6 of 10 runs

## Notes

This repository is intended to document the computational workflow associated with the manuscript. Because the original dataset is not publicly distributed, the scripts may require access to the corresponding processed data files before execution.
