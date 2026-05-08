# ==========================================================
# Random Forest model for HCY prediction
# This script can be used for either the male or female dataset.
#
# Important methodological note:
# In this model, raw data without transformation, centering, or scaling
# are used because tree-based models such as Random Forest can handle
# variables measured on different scales and do not require normality
# or linearity assumptions.
#
# Protocol:
# - Raw dataset without preprocessing transformations
# - 80/20 stratified train-test split based on HCY quintiles
# - 10-fold cross-validation applied only within the training set
# - Hyperparameter tuning of mtry
# - Final evaluation on the independent test set
# ==========================================================

# ------------------------------
# 1. Load required libraries
# ------------------------------
library(tidyverse)
library(caret)
library(randomForest)
library(readr)

# ------------------------------
# 2. Define the study group and load the raw dataset
# ------------------------------

# Use "Men" when analyzing the male dataset.
# Use "Women" when analyzing the female dataset.
group_label <- "Men"

# Load the raw dataset.
# No transformation, centering, or scaling is applied for Random Forest.
#
# Example for men:
# "~/dataset_genero_2_sin_HcyABN3.csv"
#
# Example for women:
# "~/dataset_genero_1_sin_HcyABN3.csv"
input_dataset_path <- "~/dataset_genero_2_sin_HcyABN3.csv"

data <- read_csv(input_dataset_path, show_col_types = FALSE)

cat("\nRaw dataset loaded for the", group_label, "group.\n")
cat("Input dataset path:", input_dataset_path, "\n")
cat("Number of observations:", nrow(data), "\n")
cat("Number of variables:", ncol(data), "\n")

# ------------------------------
# 3. Exclude variables not included in the predictive analysis
# ------------------------------

# These variables are excluded because they correspond to epigenetic variables,
# auxiliary categorical variables, or variables not considered in this model.
excluded_variables <- c(
  "ALU", "LINE", "SAT", "HcyABN3",
  "GrasaCatMasc", "GrasaCatFem"
)

data <- data %>%
  select(-any_of(excluded_variables))

# Display the predictors used in the model
predictor_variables <- setdiff(names(data), "HCY")

cat("\nPredictor variables used for the", group_label, "dataset:\n")
print(predictor_variables)

# Check that the response variable exists
if (!"HCY" %in% names(data)) {
  stop("The response variable 'HCY' was not found in the dataset.")
}

# ------------------------------
# 4. Create HCY levels for stratification using quintiles
# ------------------------------

# HCY is divided into quintiles to preserve the response distribution
# in both training and testing subsets.
hcy_cut_points <- unique(
  quantile(
    data$HCY,
    probs = seq(0, 1, 0.2),
    na.rm = TRUE
  )
)

if (length(hcy_cut_points) < 3) {
  stop("Not enough distinct cut points could be generated for HCY stratification.")
}

data$HCY_level <- cut(
  data$HCY,
  breaks = hcy_cut_points,
  include.lowest = TRUE,
  labels = FALSE
)

cat("\nOverall distribution of HCY levels:\n")
print(table(data$HCY_level))

cat("\nOverall proportions of HCY levels:\n")
print(round(prop.table(table(data$HCY_level)), 3))

# ------------------------------
# 5. Create an 80/20 stratified train-test split
# ------------------------------

# The split is performed within each HCY level to preserve
# approximately 80% of observations for training and 20% for testing.
set.seed(123)

index_by_level <- split(seq_len(nrow(data)), data$HCY_level)

train_index <- unlist(
  lapply(index_by_level, function(index) {
    n_train <- round(length(index) * 0.80)
    sample(index, size = n_train)
  })
)

train_index <- sort(train_index)

train_data <- data[train_index, ]
test_data  <- data[-train_index, ]

cat("\nNumber of observations in the training set:", nrow(train_data), "\n")
cat("Number of observations in the testing set :", nrow(test_data), "\n")

cat("\nDistribution of HCY levels in the training set:\n")
print(table(train_data$HCY_level))

cat("\nDistribution of HCY levels in the testing set:\n")
print(table(test_data$HCY_level))

cat("\nPercentage of each HCY level assigned to the training set:\n")
print(round(100 * table(train_data$HCY_level) / table(data$HCY_level), 2))

cat("\nPercentage of each HCY level assigned to the testing set:\n")
print(round(100 * table(test_data$HCY_level) / table(data$HCY_level), 2))

# ------------------------------
# 6. Remove the auxiliary stratification variable
# ------------------------------

# HCY_level is used only for stratified splitting.
# It must not be included as a predictor.
train_data$HCY_level <- NULL
test_data$HCY_level  <- NULL

# Remove rows with missing values, if present.
# Random Forest in caret does not automatically handle missing values.
train_data <- na.omit(train_data)
test_data  <- na.omit(test_data)

cat("\nFinal number of observations in train_data:", nrow(train_data), "\n")
cat("Final number of observations in test_data :", nrow(test_data), "\n")

# ------------------------------
# 7. Configure cross-validation only within the training set
# ------------------------------

training_control <- trainControl(
  method = "cv",
  number = 10
)

# ------------------------------
# 8. Define the mtry grid
# ------------------------------

# mtry is the number of predictors randomly sampled at each split.
mtry_grid <- c(5, 10, 15, 20, 25, 30, 35)

# ------------------------------
# 9. Evaluate each mtry value
# ------------------------------

results <- data.frame()
saved_models <- list()

for (m in mtry_grid) {
  
  cat("\n====================================\n")
  cat("Training Random Forest model with mtry =", m, "\n")
  cat("====================================\n")
  
  set.seed(123)
  
  random_forest_model <- train(
    HCY ~ .,
    data = train_data,
    method = "rf",
    trControl = training_control,
    tuneGrid = data.frame(mtry = m),
    metric = "RMSE",
    ntree = 500,
    importance = TRUE
  )
  
  # Generate predictions
  train_predictions <- predict(random_forest_model, newdata = train_data)
  test_predictions  <- predict(random_forest_model, newdata = test_data)
  
  # Compute performance metrics
  train_metrics <- postResample(
    pred = train_predictions,
    obs = train_data$HCY
  )
  
  test_metrics <- postResample(
    pred = test_predictions,
    obs = test_data$HCY
  )
  
  # Cross-validation results for the evaluated mtry value
  cv_row <- random_forest_model$results[1, ]
  
  results <- rbind(
    results,
    data.frame(
      mtry = m,
      ntree = 500,
      CV_RMSE = cv_row$RMSE,
      CV_Rsquared = cv_row$Rsquared,
      CV_MAE = cv_row$MAE,
      TRAIN_RMSE = unname(train_metrics["RMSE"]),
      TRAIN_Rsquared = unname(train_metrics["Rsquared"]),
      TRAIN_MAE = unname(train_metrics["MAE"]),
      TEST_RMSE = unname(test_metrics["RMSE"]),
      TEST_Rsquared = unname(test_metrics["Rsquared"]),
      TEST_MAE = unname(test_metrics["MAE"])
    )
  )
  
  saved_models[[paste0("mtry_", m)]] <- random_forest_model
}

# ------------------------------
# 10. Select the best mtry based on cross-validation performance
# ------------------------------

# For methodological consistency, the best mtry is selected using
# cross-validation results from the training set, not the test set.
# The independent test set is used only for final evaluation.
ordered_results <- results[order(results$CV_RMSE, -results$CV_Rsquared), ]

cat("\n=============================\n")
cat("RESULTS ORDERED BY CROSS-VALIDATION RMSE\n")
cat("=============================\n")
print(ordered_results)

# ------------------------------
# 11. Best hyperparameter according to cross-validation
# ------------------------------

best_result <- ordered_results[1, ]

cat("\n=============================\n")
cat("BEST HYPERPARAMETER ACCORDING TO CROSS-VALIDATION\n")
cat("=============================\n")
print(best_result)

best_mtry <- best_result$mtry
best_model <- saved_models[[paste0("mtry_", best_mtry)]]

cat("\nBest mtry according to cross-validation:", best_mtry, "\n")
cat("Cross-validation RMSE:", round(best_result$CV_RMSE, 6), "\n")
cat("Cross-validation R2  :", round(best_result$CV_Rsquared, 6), "\n")

cat("\nIndependent test-set performance of the selected model:\n")
cat("Test-set R2  :", round(best_result$TEST_Rsquared, 6), "\n")
cat("Test-set RMSE:", round(best_result$TEST_RMSE, 6), "\n")
cat("Test-set MAE :", round(best_result$TEST_MAE, 6), "\n")

# ------------------------------
# 12. Variable importance of the best model
# ------------------------------

variable_importance <- varImp(
  best_model,
  scale = FALSE
)

ordered_importance <- variable_importance$importance %>%
  arrange(desc(Overall))

cat("\nTop 10 most important variables in the selected Random Forest model:\n")
print(head(ordered_importance, 10))

# ------------------------------
# 13. Save results
# ------------------------------

results_file <- paste0(
  "random_forest_mtry_results_",
  tolower(group_label),
  ".csv"
)

importance_file <- paste0(
  "random_forest_variable_importance_",
  tolower(group_label),
  ".csv"
)

write_csv(results, results_file)
write_csv(ordered_importance, importance_file)

cat("\nRandom Forest mtry results saved as:", results_file, "\n")
cat("Random Forest variable importance saved as:", importance_file, "\n")

# ------------------------------
# 14. Display and plot the selected model
# ------------------------------

print(best_model)

plot(
  best_model,
  main = paste("Random Forest tuning results -", group_label, "dataset")
)