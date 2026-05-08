# ============================================================
# Elastic Net regression model for HCY prediction
# This script can be used for either the male or female dataset
# by changing the group label and the input dataset path.
# ============================================================

# ------------------------------------------------------------
# 0. Load required libraries
# ------------------------------------------------------------
library(readr)
library(caret)
library(glmnet)

# ------------------------------------------------------------
# 1. Define the study group and load the dataset
# ------------------------------------------------------------

# Define the study group according to the dataset being analyzed.
# Use "Men" for the male dataset.
# Use "Women" for the female dataset.
group_label <- "Men"

# Load the transformed dataset.
# Change this path depending on the dataset being used.
# Example for men:   "~/datos/datos_transformados_hombres.csv"
# Example for women: "~/datos/datos_transformados_mujeres.csv"
input_dataset_path <- "~/datos/datos_transformados_hombres.csv"

data <- read_csv(input_dataset_path, show_col_types = FALSE)

cat("\nDataset loaded for the", group_label, "group.\n")
cat("Input dataset path:", input_dataset_path, "\n")
cat("Number of observations:", nrow(data), "\n")
cat("Number of variables:", ncol(data), "\n")

# ------------------------------------------------------------
# 2. Exclude variables not included in the predictive analysis
# ------------------------------------------------------------

# These variables are excluded because they are either epigenetic variables,
# auxiliary categorical variables, or variables not considered in this model.
excluded_variables <- c(
  "ALU", "LINE", "SAT", "HcyABN3",
  "GrasaCatMasc", "GrasaCatFem"
)

# Remove only the variables that are present in the dataset
present_excluded_variables <- intersect(excluded_variables, names(data))

if (length(present_excluded_variables) > 0) {
  data <- data[, !(names(data) %in% present_excluded_variables)]
  
  cat("\nExcluded variables:\n")
  print(present_excluded_variables)
} else {
  cat("\nNo excluded variables were found in the dataset.\n")
}

# Check that the response variable exists
if (!"HCY" %in% names(data)) {
  stop("The response variable 'HCY' was not found in the dataset.")
}

# Remove rows with missing values, if any
data <- data[complete.cases(data), ]

cat("\nDataset after removing missing values:\n")
cat("Number of observations:", nrow(data), "\n")
cat("Number of variables:", ncol(data), "\n")

# ------------------------------------------------------------
# 3. Create HCY levels for stratified train-test splitting
# ------------------------------------------------------------

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

# Display the overall HCY level distribution
cat("\nOverall distribution of HCY levels:\n")
print(table(data$HCY_level))

cat("\nOverall proportions of HCY levels:\n")
print(round(prop.table(table(data$HCY_level)), 3))

# ------------------------------------------------------------
# 4. Create an 80/20 stratified train-test split
# ------------------------------------------------------------

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
cat("Number of observations in the testing set:", nrow(test_data), "\n")

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

# ------------------------------------------------------------
# 5. Remove the auxiliary stratification variable
# ------------------------------------------------------------

# HCY_level is used only for stratified splitting.
# It must not be included as a predictor in the model.
train_data$HCY_level <- NULL
test_data$HCY_level  <- NULL

# ------------------------------------------------------------
# 6. Define the cross-validation training control
# ------------------------------------------------------------

# Ten-fold cross-validation is applied only within the training set.
training_control <- trainControl(
  method = "cv",
  number = 10
)

# ------------------------------------------------------------
# 7. Define the Elastic Net hyperparameter grid
# ------------------------------------------------------------

# alpha controls the balance between Ridge and Lasso regularization:
# alpha = 0 corresponds to Ridge regression.
# alpha = 1 corresponds to Lasso regression.
# 0 < alpha < 1 corresponds to Elastic Net regression.
#
# lambda controls the strength of regularization.
elastic_net_grid <- expand.grid(
  alpha = seq(0, 1, by = 0.1),
  lambda = 10^seq(1, -3, length.out = 50)
)

# ------------------------------------------------------------
# 8. Train the Elastic Net model
# ------------------------------------------------------------

set.seed(123)

elastic_net_model <- train(
  HCY ~ .,
  data = train_data,
  method = "glmnet",
  trControl = training_control,
  tuneGrid = elastic_net_grid,
  preProcess = c("center", "scale"),
  metric = "RMSE"
)

# ------------------------------------------------------------
# 9. Display the best Elastic Net model
# ------------------------------------------------------------

cat("\n=============================\n")
cat("BEST ELASTIC NET MODEL -", group_label, "DATASET\n")
cat("=============================\n")

print(elastic_net_model)

cat("\nBest hyperparameters:\n")
print(elastic_net_model$bestTune)

cat("\nBest tuning result based on RMSE:\n")
print(elastic_net_model$results[which.min(elastic_net_model$results$RMSE), ])

# ------------------------------------------------------------
# 10. Generate predictions for training and testing sets
# ------------------------------------------------------------

train_predictions <- predict(elastic_net_model, newdata = train_data)
test_predictions  <- predict(elastic_net_model, newdata = test_data)

# ------------------------------------------------------------
# 11. Compute model performance metrics
# ------------------------------------------------------------

training_metrics <- postResample(
  pred = train_predictions,
  obs = train_data$HCY
)

testing_metrics <- postResample(
  pred = test_predictions,
  obs = test_data$HCY
)

cat("\nPerformance metrics on the training set:\n")
print(training_metrics)

cat("\nPerformance metrics on the independent testing set:\n")
print(testing_metrics)

# ------------------------------------------------------------
# 12. Variable importance analysis
# ------------------------------------------------------------

cat("\nVariable importance results:\n")

variable_importance <- varImp(
  elastic_net_model,
  scale = TRUE
)

# Extract the full variable importance table
importance_table <- variable_importance$importance
importance_table$Variable <- rownames(importance_table)

# Reorder columns
importance_table <- importance_table[, c("Variable", "Overall")]

# Sort variables from highest to lowest importance
importance_table <- importance_table[
  order(importance_table$Overall, decreasing = TRUE),
]

# Display all variables
print(importance_table, row.names = FALSE)

# Plot all variables according to their importance
plot(
  variable_importance,
  top = nrow(importance_table),
  main = paste("Variable importance - Elastic Net -", group_label, "dataset")
)

# ------------------------------------------------------------
# 13. Save variable importance results
# ------------------------------------------------------------

# Save the variable importance table as a CSV file.
# The output file name changes depending on the group being analyzed.
importance_output_file <- paste0(
  "elastic_net_variable_importance_",
  tolower(group_label),
  ".csv"
)

write_csv(
  importance_table,
  importance_output_file
)

cat("\nVariable importance table saved as:", importance_output_file, "\n")

# ------------------------------------------------------------
# 14. Extract coefficients from the best Elastic Net model
# ------------------------------------------------------------

cat("\nCoefficients of the best Elastic Net model:\n")

best_coefficients <- coef(
  elastic_net_model$finalModel,
  s = elastic_net_model$bestTune$lambda
)

print(best_coefficients)

# ------------------------------------------------------------
# 15. Extract non-zero coefficients
# ------------------------------------------------------------

# Non-zero coefficients represent the variables retained by the model
# after regularization.
coefficient_matrix <- as.matrix(best_coefficients)

nonzero_coefficients <- data.frame(
  Variable = rownames(coefficient_matrix),
  Coefficient = coefficient_matrix[, 1]
)

nonzero_coefficients <- nonzero_coefficients[
  nonzero_coefficients$Coefficient != 0,
]

cat("\nNon-zero coefficients selected by Elastic Net:\n")
print(nonzero_coefficients, row.names = FALSE)

# ------------------------------------------------------------
# 16. Save non-zero coefficients
# ------------------------------------------------------------

coefficients_output_file <- paste0(
  "elastic_net_nonzero_coefficients_",
  tolower(group_label),
  ".csv"
)

write_csv(
  nonzero_coefficients,
  coefficients_output_file
)

cat("\nNon-zero coefficients saved as:", coefficients_output_file, "\n")

# ------------------------------------------------------------
# 17. Final message
# ------------------------------------------------------------

cat("\nElastic Net modeling completed successfully for the", group_label, "dataset.\n")
cat("Independent test-set metrics should be used for final model reporting.\n")