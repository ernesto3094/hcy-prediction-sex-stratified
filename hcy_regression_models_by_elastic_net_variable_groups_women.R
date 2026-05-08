# ==========================================================
# HCY modeling using variable groups based on Elastic Net importance
# Complete version for the male dataset
#
# This script evaluates different predictive models using
# groups of variables derived from Elastic Net variable importance.
#
# The variable importance values used here were obtained from the
# previous Elastic Net variable selection script. In that analysis,
# the Elastic Net model was trained using only the training set after
# the stratified 80/20 split.
#
# Includes:
# - 80/20 stratified train-test split based on HCY quintiles
# - 10-fold cross-validation applied only within the training set
# - Variable groups based on importance thresholds: 20, 30, 40, 50, 60
# - Models: LM, second-degree polynomial regression, linear SVR, radial SVR
# - Final evaluation on the independent test set
# - Global comparison and selection of the best group/model combination
# - Export of results to CSV files
# ==========================================================

# =========================
# 1) Load required libraries
# =========================
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
# Change this path depending on the dataset being used.
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

data <- data[, !(names(data) %in% excluded_variables)]

# =========================
# 4) Remove missing values if present
# =========================
data <- na.omit(data)

cat("\nDataset dimensions after removing missing values:\n")
print(dim(data))

# Check that the response variable exists
if (!"HCY" %in% names(data)) {
  stop("The response variable 'HCY' was not found in the dataset.")
}

# =========================
# 5) Create HCY levels for stratification using quintiles
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
# 6) Create an 80/20 stratified train-test split
# =========================

# The split is performed within each HCY level to preserve
# approximately 80% of observations for training and 20% for testing.
set.seed(123)

index_by_level <- split(seq_len(nrow(data)), data$HCY_level)

train_index <- unlist(
  lapply(index_by_level, function(index) {
    n_train <- round(length(index) * 0.80)
    sample(index, n_train)
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
# 7) Remove the auxiliary stratification variable
# =========================

# HCY_level is used only to perform the stratified split.
# It must not be included as a predictor.
train_data$HCY_level <- NULL
test_data$HCY_level  <- NULL

# =========================
# 8) Variable importance table from Elastic Net
# =========================

# This variable importance table was obtained from the previous
# Elastic Net variable selection script for the male dataset.
#
# The Elastic Net model used to obtain these importance scores
# was trained only on the training set. Therefore, the variable
# grouping strategy is consistent with the train-test evaluation design.
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

# =========================
# 9) Build variable groups by importance threshold
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
# 10) Cross-validation configuration applied only within training data
# =========================

# Ten-fold cross-validation is applied only to the training set.
# The independent testing set is not used during model tuning.
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
# 11) Explicit hyperparameter grids for SVR models
# =========================

# Hyperparameter grid for linear SVR
linear_svr_grid <- expand.grid(
  C = 2^seq(-4, 4, by = 1)
)

# Hyperparameter grid for radial SVR
radial_svr_grid <- expand.grid(
  sigma = 2^seq(-10, -1, by = 1),
  C     = 2^seq(-4, 8, by = 1)
)

# =========================
# 12) Auxiliary functions
# =========================

# Create a standard linear regression formula
create_lm_formula <- function(predictors) {
  as.formula(
    paste("HCY ~", paste(predictors, collapse = " + "))
  )
}

# Create a second-degree polynomial formula for numeric predictors
create_polynomial_formula <- function(data_frame, predictors) {
  
  is_numeric_predictor <- sapply(
    data_frame[, predictors, drop = FALSE],
    function(x) {
      is.numeric(x) || is.integer(x)
    }
  )
  
  numeric_predictors <- predictors[is_numeric_predictor]
  non_numeric_predictors <- predictors[!is_numeric_predictor]
  
  numeric_terms <- if (length(numeric_predictors) > 0) {
    paste0("poly(", numeric_predictors, ", 2, raw = TRUE)")
  } else {
    character(0)
  }
  
  final_terms <- c(numeric_terms, non_numeric_predictors)
  
  if (length(final_terms) == 0) {
    stop("There are no valid predictors to build the polynomial formula.")
  }
  
  as.formula(
    paste("HCY ~", paste(final_terms, collapse = " + "))
  )
}

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

# Train a model and evaluate it on both training and testing sets
train_and_evaluate <- function(model_formula, train_set, test_set,
                               model_method, training_control,
                               tune_grid = NULL,
                               metric = NULL) {
  
  output <- tryCatch({
    
    training_arguments <- list(
      form = model_formula,
      data = train_set,
      method = model_method,
      trControl = training_control,
      preProcess = c("center", "scale")
    )
    
    if (!is.null(tune_grid)) training_arguments$tuneGrid <- tune_grid
    if (!is.null(metric))    training_arguments$metric <- metric
    
    model <- do.call(caret::train, training_arguments)
    
    train_predictions <- predict(model, newdata = train_set)
    test_predictions  <- predict(model, newdata = test_set)
    
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
      best_tune = format_best_tune(model)
    )
    
  }, error = function(e) {
    
    list(
      error = TRUE,
      message = e$message,
      model = NULL,
      train_metrics = c(RMSE = NA, Rsquared = NA, MAE = NA),
      test_metrics  = c(RMSE = NA, Rsquared = NA, MAE = NA),
      best_tune = NA_character_
    )
  })
  
  return(output)
}

# Build one row of results for each model and variable group
build_result_row <- function(group, threshold, n_variables, model_name, model_result) {
  
  data.frame(
    Group = group,
    Threshold = threshold,
    N_Variables = n_variables,
    Model = model_name,
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
# 13) Evaluate each variable group with all models
# =========================

all_results <- list()
saved_models <- list()

for (group_name in names(variable_groups)) {
  
  selected_predictors <- variable_groups[[group_name]]
  current_threshold <- as.numeric(sub("Group_", "", group_name))
  
  cat("\n=========================================\n")
  cat("EVALUATING ", group_name, "\n", sep = "")
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
  
  cat("\nNumber of observations in train_selected:", nrow(train_selected), "\n")
  cat("Number of observations in test_selected :", nrow(test_selected), "\n")
  
  predictors <- setdiff(names(train_selected), "HCY")
  
  lm_formula <- create_lm_formula(predictors)
  polynomial_formula <- create_polynomial_formula(train_selected, predictors)
  
  cat("\nLM formula:\n")
  print(lm_formula)
  
  cat("\nPolynomial formula:\n")
  print(polynomial_formula)
  
  # -------------------------
  # Linear model
  # -------------------------
  lm_result <- train_and_evaluate(
    model_formula = lm_formula,
    train_set = train_selected,
    test_set = test_selected,
    model_method = "lm",
    training_control = training_control
  )
  
  # -------------------------
  # Second-degree polynomial regression model
  # -------------------------
  polynomial_result <- train_and_evaluate(
    model_formula = polynomial_formula,
    train_set = train_selected,
    test_set = test_selected,
    model_method = "lm",
    training_control = training_control
  )
  
  # -------------------------
  # Linear SVR model
  # -------------------------
  linear_svr_result <- train_and_evaluate(
    model_formula = lm_formula,
    train_set = train_selected,
    test_set = test_selected,
    model_method = "svmLinear",
    training_control = training_control,
    tune_grid = linear_svr_grid,
    metric = "RMSE"
  )
  
  # -------------------------
  # Radial SVR model
  # -------------------------
  radial_svr_result <- train_and_evaluate(
    model_formula = lm_formula,
    train_set = train_selected,
    test_set = test_selected,
    model_method = "svmRadial",
    training_control = training_control,
    tune_grid = radial_svr_grid,
    metric = "RMSE"
  )
  
  group_summary <- bind_rows(
    build_result_row(group_name, current_threshold, length(selected_predictors), "LM", lm_result),
    build_result_row(group_name, current_threshold, length(selected_predictors), "Polynomial", polynomial_result),
    build_result_row(group_name, current_threshold, length(selected_predictors), "Linear_SVR", linear_svr_result),
    build_result_row(group_name, current_threshold, length(selected_predictors), "Radial_SVR", radial_svr_result)
  )
  
  all_results[[group_name]] <- group_summary
  
  saved_models[[group_name]] <- list(
    LM = lm_result$model,
    Polynomial = polynomial_result$model,
    Linear_SVR = linear_svr_result$model,
    Radial_SVR = radial_svr_result$model
  )
  
  cat("\nTest-set summary for ", group_name, ":\n", sep = "")
  print(
    group_summary[, c(
      "Group", "Model", "RMSE_Test", "Rsquared_Test",
      "MAE_Test", "BestTune", "Error"
    )]
  )
}

# =========================
# 14) Combine all results
# =========================

final_comparison <- bind_rows(all_results)

cat("\n=========================================\n")
cat("FINAL COMPARISON OF ALL VARIABLE GROUPS\n")
cat("=========================================\n")
print(final_comparison)

# =========================
# 15) Rank models based on test-set performance
# =========================

# Ranking criterion:
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
# 16) Select the best final combination
# =========================

if (nrow(ordered_comparison) == 0) {
  stop("No valid models were obtained. Check the errors printed in the final table.")
}

best_result <- ordered_comparison[1, ]

cat("\n=========================================\n")
cat("BEST VARIABLE GROUP AND BEST MODEL\n")
cat("=========================================\n")
print(best_result)

cat("\nVariables included in the winning group:\n")
print(variable_groups[[best_result$Group]])

# =========================
# 17) Display bestTune of the winning model
# =========================

winning_group <- best_result$Group
winning_model <- best_result$Model

cat("\n=========================================\n")
cat("HYPERPARAMETERS OF THE WINNING MODEL\n")
cat("=========================================\n")

if (winning_model %in% c("LM", "Polynomial")) {
  cat("The winning model does not have tunable hyperparameters in caret.\n")
} else {
  print(saved_models[[winning_group]][[winning_model]]$bestTune)
}

# =========================
# 18) Final summary of the winning model
# =========================

cat("\n=========================================\n")
cat("FINAL SUMMARY OF THE WINNING MODEL\n")
cat("=========================================\n")

cat("Winning group       :", best_result$Group, "\n")
cat("Threshold           :", best_result$Threshold, "\n")
cat("Number of variables :", best_result$N_Variables, "\n")
cat("Winning model       :", best_result$Model, "\n")
cat("Test-set R2         :", round(best_result$Rsquared_Test, 6), "\n")
cat("Test-set RMSE       :", round(best_result$RMSE_Test, 6), "\n")
cat("Test-set MAE        :", round(best_result$MAE_Test, 6), "\n")
cat("BestTune            :", best_result$BestTune, "\n")

# =========================
# 19) Export results to CSV files
# =========================

final_results_file <- paste0(
  "hcy_model_comparison_by_elastic_net_importance_",
  tolower(group_label),
  ".csv"
)

ordered_results_file <- paste0(
  "hcy_ordered_model_comparison_by_elastic_net_importance_",
  tolower(group_label),
  ".csv"
)

write_csv(final_comparison, final_results_file)
write_csv(ordered_comparison, ordered_results_file)

cat("\nFinal comparison table saved as:", final_results_file, "\n")
cat("Ordered comparison table saved as:", ordered_results_file, "\n")

# =========================
# 20) Final methodological reminder
# =========================

cat("\nModeling completed successfully for the", group_label, "dataset.\n")
cat("Final reporting should be based on the independent test-set metrics.\n")