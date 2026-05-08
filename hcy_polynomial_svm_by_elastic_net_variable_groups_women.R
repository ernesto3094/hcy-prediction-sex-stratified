# ==========================================================
# Polynomial SVM modeling for HCY prediction using
# Elastic Net importance-based variable groups
#
# This script can be used for either the male or female dataset.
# To use it for another group, change:
# - group_label
# - input_dataset_path
# - importance_table
#
# Protocol:
# - 80/20 stratified train-test split based on HCY quintiles
# - 10-fold cross-validation applied only within the training set
# - Variable groups based on Elastic Net importance thresholds:
#   20, 30, 40, 50, and 60
# - Polynomial SVM training for each variable group
# - Final comparison using test-set R2, RMSE, and MAE
# ==========================================================

# ------------------------------
# 1) Load required libraries
# ------------------------------
library(readr)
library(caret)
library(e1071)
library(kernlab)
library(dplyr)
library(tibble)

set.seed(123)

# =========================
# 2) Define the study group and load the dataset
# =========================

# Use "Men" when analyzing the male dataset.
# Use "Women" when analyzing the female dataset.
group_label <- "Women"

# Load the transformed dataset.
# Change this path according to the dataset being analyzed.
# Example for men:   "~/datos/datos_transformados_hombres.csv"
# Example for women: "~/datos/datos_transformados_mujeres.csv"
input_dataset_path <- "~/datos/datos_transformados_mujeres.csv"

data <- read_csv(input_dataset_path, show_col_types = FALSE)

cat("\nDataset loaded for the", group_label, "group.\n")
cat("Input dataset path:", input_dataset_path, "\n")

# =========================
# 3) Exclude variables not included in the predictive analysis
# =========================

# These variables are excluded because they correspond to epigenetic variables,
# auxiliary categorical variables, or variables not considered in this modeling step.
excluded_variables <- c(
  "ALU", "LINE", "SAT", "HcyABN3",
  "GrasaCatMasc", "GrasaCatFem"
)

excluded_variables <- intersect(excluded_variables, names(data))

if (length(excluded_variables) > 0) {
  data <- data[, !(names(data) %in% excluded_variables)]
  
  cat("\nExcluded variables:\n")
  print(excluded_variables)
} else {
  cat("\nNo excluded variables were found in the dataset.\n")
}

# Remove missing values if present
data <- na.omit(data)

cat("\nDataset dimensions after removing missing values:\n")
print(dim(data))

# Check that the response variable exists
if (!"HCY" %in% names(data)) {
  stop("The response variable 'HCY' was not found in the dataset.")
}

# =========================
# 4) Create HCY levels for stratification using quintiles
# =========================

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

# =========================
# 5) Create an 80/20 stratified train-test split
# =========================

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

cat("\nInternal proportions of HCY levels in the training set:\n")
print(round(prop.table(table(train_data$HCY_level)), 3))

cat("\nInternal proportions of HCY levels in the testing set:\n")
print(round(prop.table(table(test_data$HCY_level)), 3))

# =========================
# 6) Remove the auxiliary stratification variable
# =========================

# HCY_level is used only to perform the stratified split.
# It must not be included as a predictor.
train_data$HCY_level <- NULL
test_data$HCY_level  <- NULL

# =========================
# 7) Variable importance table from Elastic Net
# =========================

# This variable importance table was obtained from the previous
# Elastic Net variable selection script for the male dataset.
#
# The variables with Overall > 0 should coincide with the variables
# retained by the Elastic Net model through non-zero coefficients.
#
# For the female dataset, replace this table with the corresponding
# Elastic Net importance results obtained from the female training set.
importance_table <- tibble(
  Variable = c(
    "LDL", "GLU", "LDLHDLIndice", "AIP", "MetabBasal", "RDWCV",
    "WBC", "MCH", "Talla", "Edad", "VCM", "Peso", "Musculo",
    "GrasaVisc", "IMCCat", "VLDL", "IMC", "ColHDLIndice",
    "HDL", "RBC", "COL", "GrasaT", "Hto", "MCHC", "TRG",
    "LCI", "PLT", "Hgb"
  ),
  Overall = c(
    100.000000, 87.850589, 64.063379, 58.923877, 48.856427,
    48.842311, 48.065126, 47.280986, 46.182795, 46.046624,
    42.410257, 41.973125, 41.342987, 39.886402, 33.835395,
    33.796072, 31.738599, 31.716071, 29.682231, 29.412624,
    24.044107, 18.245849, 17.079260, 14.640615, 14.273382,
    3.199405, 1.833107, 0.000000
  )
)

cat("\n=========================================\n")
cat("VARIABLE IMPORTANCE TABLE -", toupper(group_label), "DATASET\n")
cat("=========================================\n")
print(importance_table)

# Optional check: variables with non-zero Elastic Net importance
nonzero_importance_variables <- importance_table %>%
  filter(Overall > 0) %>%
  arrange(desc(Overall)) %>%
  pull(Variable)

cat("\nVariables with non-zero Elastic Net importance:\n")
print(nonzero_importance_variables)

# Optional check: verify that all important variables exist in the dataset
missing_important_variables <- setdiff(nonzero_importance_variables, names(data))

if (length(missing_important_variables) > 0) {
  cat("\nWarning: The following important variables are not present in the dataset:\n")
  print(missing_important_variables)
} else {
  cat("\nAll variables with non-zero importance are present in the dataset.\n")
}

# =========================
# 8) Build variable groups by importance threshold
# =========================

# Variables are grouped according to their Elastic Net importance score.
# For example, Group_20 includes all variables with importance >= 20.
importance_thresholds <- c(20, 30, 40, 50, 60)

variable_groups <- setNames(
  lapply(importance_thresholds, function(threshold) {
    importance_table %>%
      filter(Overall >= threshold) %>%
      arrange(desc(Overall)) %>%
      pull(Variable)
  }),
  paste0("Group_", importance_thresholds)
)

cat("\n=========================================\n")
cat("VARIABLE GROUPS BY IMPORTANCE THRESHOLD\n")
cat("=========================================\n")

for (group_name in names(variable_groups)) {
  cat("\n", group_name, " (", length(variable_groups[[group_name]]), " variables):\n", sep = "")
  print(variable_groups[[group_name]])
}

# =========================
# 9) Cross-validation configuration applied only within training data
# =========================

# The same folds are used across all variable groups.
set.seed(123)

cv_folds <- createFolds(
  train_data$HCY,
  k = 10,
  returnTrain = TRUE
)

training_control <- trainControl(
  method = "cv",
  index = cv_folds
)

# =========================
# 10) Explicit hyperparameter grid for polynomial SVM
# =========================

# degree: polynomial degree
# scale: polynomial kernel scale parameter
# C: regularization cost parameter
polynomial_svm_grid <- expand.grid(
  degree = 2:8,
  scale  = c(0.001, 0.01, 0.1, 1),
  C      = c(0.25, 0.5, 1, 2, 4, 8, 16)
)

# =========================
# 11) Auxiliary functions
# =========================

# Format the best hyperparameters selected by caret
format_best_tune <- function(model) {
  
  if (is.null(model$bestTune) || nrow(model$bestTune) == 0) {
    return(NA_character_)
  }
  
  paste(
    paste(
      names(model$bestTune),
      as.character(unlist(model$bestTune)),
      sep = "="
    ),
    collapse = "; "
  )
}

# Train and evaluate a polynomial SVM model
train_and_evaluate_polynomial_svm <- function(train_set, test_set,
                                              training_control,
                                              polynomial_svm_grid) {
  
  output <- tryCatch({
    
    polynomial_svm_formula <- HCY ~ .
    
    model <- caret::train(
      polynomial_svm_formula,
      data = train_set,
      method = "svmPoly",
      trControl = training_control,
      preProcess = c("center", "scale"),
      tuneGrid = polynomial_svm_grid,
      metric = "RMSE"
    )
    
    train_predictions <- predict(model, newdata = train_set)
    test_predictions  <- predict(model, newdata = test_set)
    
    best_cv_row <- model$results[which.min(model$results$RMSE), ]
    
    list(
      error = FALSE,
      model = model,
      train_metrics = postResample(
        pred = train_predictions,
        obs = train_set$HCY
      ),
      test_metrics = postResample(
        pred = test_predictions,
        obs = test_set$HCY
      ),
      best_tune = format_best_tune(model),
      best_cv_row = best_cv_row
    )
    
  }, error = function(e) {
    
    list(
      error = TRUE,
      message = e$message,
      model = NULL,
      train_metrics = c(RMSE = NA, Rsquared = NA, MAE = NA),
      test_metrics  = c(RMSE = NA, Rsquared = NA, MAE = NA),
      best_tune = NA_character_,
      best_cv_row = NULL
    )
  })
  
  return(output)
}

# Build one result row for each variable group
build_result_row <- function(group, threshold, n_variables, model_result) {
  
  data.frame(
    Group = group,
    Threshold = threshold,
    N_Variables = n_variables,
    Model = "Polynomial_SVM",
    RMSE_Train = unname(model_result$train_metrics["RMSE"]),
    Rsquared_Train = unname(model_result$train_metrics["Rsquared"]),
    MAE_Train = unname(model_result$train_metrics["MAE"]),
    RMSE_Test = unname(model_result$test_metrics["RMSE"]),
    Rsquared_Test = unname(model_result$test_metrics["Rsquared"]),
    MAE_Test = unname(model_result$test_metrics["MAE"]),
    BestTune = model_result$best_tune,
    Error = ifelse(isTRUE(model_result$error), model_result$message, NA_character_),
    stringsAsFactors = FALSE
  )
}

# =========================
# 12) Evaluate polynomial SVM in each variable group
# =========================

all_results <- list()
saved_models <- list()

for (group_name in names(variable_groups)) {
  
  selected_predictors <- variable_groups[[group_name]]
  current_threshold <- as.numeric(sub("Group_", "", group_name))
  
  cat("\n=========================================\n")
  cat("EVALUATING ", group_name, " WITH POLYNOMIAL SVM\n", sep = "")
  cat("=========================================\n")
  
  cat("Variables included in this group:\n")
  print(selected_predictors)
  
  model_variables <- c("HCY", selected_predictors)
  
  missing_train_columns <- setdiff(model_variables, names(train_data))
  missing_test_columns  <- setdiff(model_variables, names(test_data))
  
  if (length(missing_train_columns) > 0) {
    stop(
      paste(
        "Missing columns in train_data for",
        group_name,
        ":",
        paste(missing_train_columns, collapse = ", ")
      )
    )
  }
  
  if (length(missing_test_columns) > 0) {
    stop(
      paste(
        "Missing columns in test_data for",
        group_name,
        ":",
        paste(missing_test_columns, collapse = ", ")
      )
    )
  }
  
  train_selected <- na.omit(train_data[, model_variables, drop = FALSE])
  test_selected  <- na.omit(test_data[, model_variables, drop = FALSE])
  
  cat("\nFinal number of observations in train_selected:", nrow(train_selected), "\n")
  cat("Final number of observations in test_selected :", nrow(test_selected), "\n")
  
  polynomial_svm_result <- train_and_evaluate_polynomial_svm(
    train_set = train_selected,
    test_set = test_selected,
    training_control = training_control,
    polynomial_svm_grid = polynomial_svm_grid
  )
  
  all_results[[group_name]] <- build_result_row(
    group = group_name,
    threshold = current_threshold,
    n_variables = length(selected_predictors),
    model_result = polynomial_svm_result
  )
  
  saved_models[[group_name]] <- polynomial_svm_result$model
  
  cat("\nBestTune for ", group_name, ":\n", sep = "")
  print(polynomial_svm_result$best_tune)
  
  cat("\nBest CV tuning row for ", group_name, ":\n", sep = "")
  print(polynomial_svm_result$best_cv_row)
  
  cat("\nTest-set metrics for ", group_name, ":\n", sep = "")
  print(
    all_results[[group_name]][, c(
      "Group", "Model", "RMSE_Test", "Rsquared_Test",
      "MAE_Test", "BestTune", "Error"
    )]
  )
}

# =========================
# 13) Combine all results
# =========================

final_comparison <- bind_rows(all_results)

cat("\n=========================================\n")
cat("FINAL COMPARISON OF POLYNOMIAL SVM BY VARIABLE GROUPS\n")
cat("=========================================\n")
print(final_comparison)

# =========================
# 14) Rank groups based on test-set performance
# =========================

# Main ranking criterion:
# 1) Highest test-set R²
# 2) Lowest test-set RMSE
# 3) Lowest test-set MAE
ordered_comparison <- final_comparison %>%
  filter(is.na(Error)) %>%
  arrange(desc(Rsquared_Test), RMSE_Test, MAE_Test)

cat("\n=========================================\n")
cat("ORDERED RESULTS FROM BEST TO WORST\n")
cat("=========================================\n")
print(ordered_comparison)

# =========================
# 15) Best group according to test-set R²
# =========================

if (nrow(ordered_comparison) == 0) {
  stop("No valid models were obtained. Check the errors printed in the final table.")
}

best_by_R2 <- ordered_comparison[1, ]

cat("\n=========================================\n")
cat("BEST GROUP ACCORDING TO TEST-SET R2\n")
cat("=========================================\n")
print(best_by_R2)

cat("\nVariables included in the winning group according to R2:\n")
print(variable_groups[[best_by_R2$Group]])

# =========================
# 16) Best group according to test-set RMSE
# =========================

best_by_RMSE <- final_comparison %>%
  filter(is.na(Error)) %>%
  arrange(RMSE_Test, desc(Rsquared_Test), MAE_Test) %>%
  slice(1)

cat("\n=========================================\n")
cat("BEST GROUP ACCORDING TO TEST-SET RMSE\n")
cat("=========================================\n")
print(best_by_RMSE)

# =========================
# 17) Best group according to test-set MAE
# =========================

best_by_MAE <- final_comparison %>%
  filter(is.na(Error)) %>%
  arrange(MAE_Test, desc(Rsquared_Test), RMSE_Test) %>%
  slice(1)

cat("\n=========================================\n")
cat("BEST GROUP ACCORDING TO TEST-SET MAE\n")
cat("=========================================\n")
print(best_by_MAE)

# =========================
# 18) Display bestTune of the R2-winning model
# =========================

winning_group <- best_by_R2$Group

cat("\n=========================================\n")
cat("HYPERPARAMETERS OF THE R2-WINNING MODEL\n")
cat("=========================================\n")

print(saved_models[[winning_group]]$bestTune)

# =========================
# 19) Final summary
# =========================

final_summary <- data.frame(
  Criterion = c("Best_R2", "Best_RMSE", "Best_MAE"),
  Group = c(best_by_R2$Group, best_by_RMSE$Group, best_by_MAE$Group),
  Threshold = c(best_by_R2$Threshold, best_by_RMSE$Threshold, best_by_MAE$Threshold),
  N_Variables = c(best_by_R2$N_Variables, best_by_RMSE$N_Variables, best_by_MAE$N_Variables),
  RMSE_Test = c(best_by_R2$RMSE_Test, best_by_RMSE$RMSE_Test, best_by_MAE$RMSE_Test),
  Rsquared_Test = c(best_by_R2$Rsquared_Test, best_by_RMSE$Rsquared_Test, best_by_MAE$Rsquared_Test),
  MAE_Test = c(best_by_R2$MAE_Test, best_by_RMSE$MAE_Test, best_by_MAE$MAE_Test),
  BestTune = c(best_by_R2$BestTune, best_by_RMSE$BestTune, best_by_MAE$BestTune),
  stringsAsFactors = FALSE
)

cat("\n=========================================\n")
cat("FINAL SUMMARY OF WINNING MODELS\n")
cat("=========================================\n")
print(final_summary)

# =========================
# 20) Export results to CSV files
# =========================

final_comparison_file <- paste0(
  "hcy_polynomial_svm_group_comparison_",
  tolower(group_label),
  ".csv"
)

final_summary_file <- paste0(
  "hcy_polynomial_svm_winners_summary_",
  tolower(group_label),
  ".csv"
)

write_csv(final_comparison, final_comparison_file)
write_csv(final_summary, final_summary_file)

cat("\nFinal polynomial SVM comparison table saved as:", final_comparison_file, "\n")
cat("Final winners summary saved as:", final_summary_file, "\n")

# =========================
# 21) Plot the R2-winning polynomial SVM tuning results
# =========================

plot(
  saved_models[[winning_group]],
  main = paste("Polynomial SVM tuning results -", group_label, "dataset")
)