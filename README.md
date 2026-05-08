# Sex-stratified prediction of homocysteine levels using regression and machine learning models

This repository contains the R scripts used to implement the analytical workflow for the sex-stratified prediction of homocysteine (HCY) levels in young adults.

The workflow includes data preprocessing, preliminary diagnostic assessment, variable transformation, Elastic Net-based variable selection, construction of importance-based predictor groups, model training, and independent test-set evaluation.

## Study overview

Homocysteine (HCY) is a biomarker associated with cardiovascular, metabolic, and physiological alterations. This project evaluates different predictive modeling strategies to determine which approaches better explain HCY variability in women and men using anthropometric, biochemical, and hematological variables.

The analysis was performed separately by sex to explore potential differences in predictive behavior between women and men.

## Modeling strategies

Two complementary modeling workflows were evaluated.

### 1. Transformed and regularized workflow

This workflow was applied to:

- Multiple Linear Regression (MLR)
- Polynomial Regression (PR)
- Support Vector Regression (SVR)

It included:

- Preliminary regression diagnostics
- Variable transformations based on skewness and kurtosis
- Centering and scaling when required
- Elastic Net-based variable selection
- Construction of importance-based predictor groups
- Model tuning using 10-fold cross-validation within the training set
- Final evaluation on an independent test set

### 2. Tree-based ensemble workflow

This workflow was applied to:

- Random Forest (RF)
- Extreme Gradient Boosting (XGBoost)

These models were trained using the untransformed analytical predictors, without prior Elastic Net-based variable selection, because tree-based ensemble methods can capture nonlinear relationships, threshold effects, and variable interactions without requiring normality or linearity assumptions.

## General workflow

The analytical procedure followed these main steps:

1. Load the sex-specific datasets.
2. Remove variables not aligned with the predictive objective.
3. Perform preliminary regression diagnostics.
4. Apply variable transformations according to skewness and kurtosis criteria.
5. Split the data into training and test sets using HCY-stratified partitioning.
6. Apply Elastic Net-based variable selection within the transformed workflow.
7. Construct importance-based predictor groups.
8. Train MLR, PR, and SVR models using 10-fold cross-validation within the training set.
9. Train RF and XGBoost models using untransformed predictors.
10. Evaluate final model performance on the independent test set using R², RMSE, and MAE.

## Data availability

The original dataset is not included in this repository due to confidentiality restrictions and authorization requirements associated with the original study.

The scripts are provided to document the analytical workflow and support reproducibility of the preprocessing, modeling, and evaluation procedures described in the manuscript.

Researchers interested in the dataset should contact the corresponding author, subject to approval from the data owner and applicable confidentiality conditions.

## Code availability

The R scripts used to implement the main analytical procedures are available in this repository:

https://github.com/ernesto3094/hcy-prediction-sex-stratified

## Required R packages

The analysis was implemented in R. The main packages used include:

- readr
- dplyr
- tidyr
- ggplot2
- caret
- glmnet
- e1071
- MASS
- car
- lmtest
- nortest
- randomForest
- xgboost
- tidyverse

Depending on the specific script, additional packages may be required.

## Reproducibility

To improve reproducibility, a fixed random seed was used in the modeling procedures:

```r
set.seed(123)
