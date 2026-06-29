# ==========================================================
# Random Forest model for HCY prediction
# Repeated-seed complete pipeline
# Female dataset
#
# Protocol:
# - Raw dataset without transformation, centering, or scaling
# - Full set of eligible predictors
# - 80/20 stratified train-test split based on HCY quintiles
# - 10 repeated seeds
# - 10-fold cross-validation within the training set only
# - Hyperparameter tuning of mtry
# - Final evaluation on independent test sets
# - Variable importance
# - Learning curves
# - Excel and figures exported to user-defined folder
# ==========================================================

# ------------------------------
# 1. Load required libraries
# ------------------------------

library(tidyverse)
library(caret)
library(randomForest)
library(readr)
library(openxlsx)
library(ggplot2)

# ------------------------------
# 2. General configuration
# ------------------------------

group_label <- "Women"

input_dataset_path <- "data/processed/dataset_genero_1_sin_HcyABN3.csv"

output_dir <- "results/women/random_forest"

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

seed_values <- c(
  123, 321, 456, 654, 789,
  987, 111, 222, 333, 444
)

ntree_value <- 500

learning_fractions <- c(0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00)

# ------------------------------
# 3. Load raw dataset
# ------------------------------

data <- read_csv(input_dataset_path, show_col_types = FALSE)

cat("\nRaw dataset loaded for the", group_label, "group.\n")
cat("Input dataset path:", input_dataset_path, "\n")
cat("Number of observations:", nrow(data), "\n")
cat("Number of variables:", ncol(data), "\n")

# ------------------------------
# 4. Exclude non-eligible variables
# ------------------------------

excluded_variables <- c(
  "ID", "Id", "id",
  "Gender", "Genero", "Género",
  "ALU", "LINE", "SAT",
  "HcyABN3",
  "GrasaCatMasc", "GrasaCatFem",
  "FatCatMasc", "FatCatFem"
)

data <- data %>%
  select(-any_of(excluded_variables))

if (!"HCY" %in% names(data)) {
  stop("The response variable 'HCY' was not found in the dataset.")
}

data <- data %>%
  drop_na()

predictor_variables <- setdiff(names(data), "HCY")
num_predictors <- length(predictor_variables)

cat("\nPredictor variables used in the Random Forest model:\n")
print(predictor_variables)

cat("\nNumber of eligible predictors:", num_predictors, "\n")
cat("Final number of complete observations:", nrow(data), "\n")

# ------------------------------
# 5. Metric function
# ------------------------------

compute_metrics <- function(obs, pred) {
  
  rmse <- sqrt(mean((obs - pred)^2))
  mae  <- mean(abs(obs - pred))
  
  sse <- sum((obs - pred)^2)
  sst <- sum((obs - mean(obs))^2)
  
  r2 <- 1 - (sse / sst)
  
  return(
    data.frame(
      RMSE = rmse,
      R2 = r2,
      MAE = mae
    )
  )
}

# ------------------------------
# 6. Stratified split function
# ------------------------------

create_stratified_split <- function(dataset, seed_value, train_prop = 0.80) {
  
  set.seed(seed_value)
  
  hcy_cut_points <- unique(
    quantile(
      dataset$HCY,
      probs = seq(0, 1, 0.2),
      na.rm = TRUE
    )
  )
  
  if (length(hcy_cut_points) < 3) {
    stop("Not enough distinct cut points could be generated for HCY stratification.")
  }
  
  dataset$HCY_level <- cut(
    dataset$HCY,
    breaks = hcy_cut_points,
    include.lowest = TRUE,
    labels = FALSE
  )
  
  index_by_level <- split(seq_len(nrow(dataset)), dataset$HCY_level)
  
  train_index <- unlist(
    lapply(index_by_level, function(index) {
      n_train <- round(length(index) * train_prop)
      sample(index, size = n_train)
    })
  )
  
  train_index <- sort(train_index)
  
  train_data <- dataset[train_index, ]
  test_data  <- dataset[-train_index, ]
  
  train_data$HCY_level <- NULL
  test_data$HCY_level  <- NULL
  
  return(
    list(
      train_data = train_data,
      test_data = test_data
    )
  )
}

# ------------------------------
# 7. Stratified subsampling function for learning curves
# ------------------------------

create_learning_subset <- function(train_dataset, fraction, seed_value) {
  
  set.seed(seed_value)
  
  if (fraction >= 1) {
    return(train_dataset)
  }
  
  hcy_cut_points <- unique(
    quantile(
      train_dataset$HCY,
      probs = seq(0, 1, 0.2),
      na.rm = TRUE
    )
  )
  
  train_dataset$HCY_level_tmp <- cut(
    train_dataset$HCY,
    breaks = hcy_cut_points,
    include.lowest = TRUE,
    labels = FALSE
  )
  
  index_by_level <- split(seq_len(nrow(train_dataset)), train_dataset$HCY_level_tmp)
  
  subset_index <- unlist(
    lapply(index_by_level, function(index) {
      n_subset <- max(2, round(length(index) * fraction))
      sample(index, size = min(n_subset, length(index)))
    })
  )
  
  subset_index <- sort(subset_index)
  
  subset_data <- train_dataset[subset_index, ]
  subset_data$HCY_level_tmp <- NULL
  
  return(subset_data)
}

# ------------------------------
# 8. Dynamic mtry grid
# ------------------------------

mtry_grid <- unique(
  pmax(
    1,
    pmin(
      num_predictors,
      c(
        1,
        floor(sqrt(num_predictors)),
        floor(num_predictors / 4),
        floor(num_predictors / 3),
        floor(num_predictors / 2),
        num_predictors
      )
    )
  )
)

mtry_grid <- sort(unique(mtry_grid))

cat("\nDynamic mtry grid:\n")
print(mtry_grid)

# ------------------------------
# 9. Containers
# ------------------------------

all_run_metrics <- data.frame()
all_cv_results <- data.frame()
all_hyperparameters <- data.frame()
all_importance <- data.frame()
all_learning_curves <- data.frame()

# ------------------------------
# 10. Repeated-seed Random Forest pipeline
# ------------------------------

for (current_seed in seed_values) {
  
  cat("\n====================================================\n")
  cat("Running seed:", current_seed, "\n")
  cat("====================================================\n")
  
  split_object <- create_stratified_split(
    dataset = data,
    seed_value = current_seed,
    train_prop = 0.80
  )
  
  train_data <- split_object$train_data
  test_data  <- split_object$test_data
  
  cat("Training observations:", nrow(train_data), "\n")
  cat("Testing observations :", nrow(test_data), "\n")
  
  training_control <- trainControl(
    method = "cv",
    number = 10
  )
  
  set.seed(current_seed)
  
  rf_model <- train(
    HCY ~ .,
    data = train_data,
    method = "rf",
    trControl = training_control,
    tuneGrid = data.frame(mtry = mtry_grid),
    metric = "RMSE",
    ntree = ntree_value,
    importance = TRUE
  )
  
  best_mtry <- rf_model$bestTune$mtry
  
  cat("Best mtry:", best_mtry, "\n")
  
  cv_results_seed <- rf_model$results %>%
    mutate(
      Seed = current_seed,
      Group_label = group_label,
      ntree = ntree_value
    ) %>%
    select(
      Group_label, Seed, ntree, mtry,
      RMSE, Rsquared, MAE,
      RMSESD, RsquaredSD, MAESD
    )
  
  all_cv_results <- bind_rows(all_cv_results, cv_results_seed)
  
  train_predictions <- predict(rf_model, newdata = train_data)
  test_predictions  <- predict(rf_model, newdata = test_data)
  
  train_metrics <- compute_metrics(
    obs = train_data$HCY,
    pred = train_predictions
  )
  
  test_metrics <- compute_metrics(
    obs = test_data$HCY,
    pred = test_predictions
  )
  
  run_metrics <- data.frame(
    Group_label = group_label,
    Seed = current_seed,
    Model = "Random_Forest",
    Data_type = "Original_raw_HCY",
    Num_predictors = num_predictors,
    Train_n = nrow(train_data),
    Test_n = nrow(test_data),
    ntree = ntree_value,
    best_mtry = best_mtry,
    Train_R2 = train_metrics$R2,
    Train_RMSE = train_metrics$RMSE,
    Train_MAE = train_metrics$MAE,
    Test_R2 = test_metrics$R2,
    Test_RMSE = test_metrics$RMSE,
    Test_MAE = test_metrics$MAE,
    R2_gap = train_metrics$R2 - test_metrics$R2,
    RMSE_gap = train_metrics$RMSE - test_metrics$RMSE,
    MAE_gap = train_metrics$MAE - test_metrics$MAE
  )
  
  all_run_metrics <- bind_rows(all_run_metrics, run_metrics)
  
  hyper_row <- data.frame(
    Group_label = group_label,
    Seed = current_seed,
    Model = "Random_Forest",
    ntree = ntree_value,
    best_mtry = best_mtry,
    Num_predictors = num_predictors
  )
  
  all_hyperparameters <- bind_rows(all_hyperparameters, hyper_row)
  
  importance_seed <- varImp(rf_model, scale = FALSE)$importance %>%
    rownames_to_column("Variable") %>%
    arrange(desc(Overall)) %>%
    mutate(
      Group_label = group_label,
      Seed = current_seed,
      Model = "Random_Forest",
      best_mtry = best_mtry
    ) %>%
    select(Group_label, Seed, Model, best_mtry, Variable, Overall)
  
  all_importance <- bind_rows(all_importance, importance_seed)
  
  for (fraction_value in learning_fractions) {
    
    cat("Learning curve fraction:", fraction_value, "\n")
    
    subset_data <- create_learning_subset(
      train_dataset = train_data,
      fraction = fraction_value,
      seed_value = current_seed
    )
    
    set.seed(current_seed)
    
    rf_lc_model <- randomForest(
      HCY ~ .,
      data = subset_data,
      ntree = ntree_value,
      mtry = best_mtry,
      importance = TRUE
    )
    
    lc_train_predictions <- predict(rf_lc_model, newdata = subset_data)
    lc_test_predictions  <- predict(rf_lc_model, newdata = test_data)
    
    lc_train_metrics <- compute_metrics(
      obs = subset_data$HCY,
      pred = lc_train_predictions
    )
    
    lc_test_metrics <- compute_metrics(
      obs = test_data$HCY,
      pred = lc_test_predictions
    )
    
    lc_row <- data.frame(
      Group_label = group_label,
      Seed = current_seed,
      Model = "Random_Forest",
      Fraction = fraction_value,
      Train_n = nrow(subset_data),
      Test_n = nrow(test_data),
      ntree = ntree_value,
      mtry = best_mtry,
      Train_R2 = lc_train_metrics$R2,
      Train_RMSE = lc_train_metrics$RMSE,
      Train_MAE = lc_train_metrics$MAE,
      Test_R2 = lc_test_metrics$R2,
      Test_RMSE = lc_test_metrics$RMSE,
      Test_MAE = lc_test_metrics$MAE,
      R2_gap = lc_train_metrics$R2 - lc_test_metrics$R2,
      RMSE_gap = lc_train_metrics$RMSE - lc_test_metrics$RMSE,
      MAE_gap = lc_train_metrics$MAE - lc_test_metrics$MAE
    )
    
    all_learning_curves <- bind_rows(all_learning_curves, lc_row)
  }
}

# ------------------------------
# 11. Summary metrics
# ------------------------------

summary_metrics <- all_run_metrics %>%
  summarise(
    Group_label = first(Group_label),
    Model = first(Model),
    Data_type = first(Data_type),
    Runs = n(),
    Num_predictors = first(Num_predictors),
    
    Mean_Train_R2 = mean(Train_R2),
    SD_Train_R2 = sd(Train_R2),
    CI95_Train_R2 = qt(0.975, df = n() - 1) * sd(Train_R2) / sqrt(n()),
    
    Mean_Train_RMSE = mean(Train_RMSE),
    SD_Train_RMSE = sd(Train_RMSE),
    CI95_Train_RMSE = qt(0.975, df = n() - 1) * sd(Train_RMSE) / sqrt(n()),
    
    Mean_Train_MAE = mean(Train_MAE),
    SD_Train_MAE = sd(Train_MAE),
    CI95_Train_MAE = qt(0.975, df = n() - 1) * sd(Train_MAE) / sqrt(n()),
    
    Mean_Test_R2 = mean(Test_R2),
    SD_Test_R2 = sd(Test_R2),
    CI95_Test_R2 = qt(0.975, df = n() - 1) * sd(Test_R2) / sqrt(n()),
    
    Mean_Test_RMSE = mean(Test_RMSE),
    SD_Test_RMSE = sd(Test_RMSE),
    CI95_Test_RMSE = qt(0.975, df = n() - 1) * sd(Test_RMSE) / sqrt(n()),
    
    Mean_Test_MAE = mean(Test_MAE),
    SD_Test_MAE = sd(Test_MAE),
    CI95_Test_MAE = qt(0.975, df = n() - 1) * sd(Test_MAE) / sqrt(n()),
    
    Mean_R2_gap = mean(R2_gap),
    Mean_RMSE_gap = mean(RMSE_gap),
    Mean_MAE_gap = mean(MAE_gap),
    
    Most_frequent_mtry = as.numeric(
      names(sort(table(all_hyperparameters$best_mtry), decreasing = TRUE)[1])
    ),
    Median_mtry = median(all_hyperparameters$best_mtry)
  )

hyperparameter_frequency <- all_hyperparameters %>%
  count(best_mtry, name = "Frequency") %>%
  mutate(
    Percentage = 100 * Frequency / sum(Frequency)
  ) %>%
  arrange(desc(Frequency))

importance_summary <- all_importance %>%
  group_by(Variable) %>%
  summarise(
    Mean_Importance = mean(Overall),
    SD_Importance = sd(Overall),
    CI95_Importance = qt(0.975, df = n() - 1) * sd(Overall) / sqrt(n()),
    Times_Selected = n(),
    .groups = "drop"
  ) %>%
  arrange(desc(Mean_Importance))

learning_curve_summary <- all_learning_curves %>%
  group_by(Group_label, Model, Fraction) %>%
  summarise(
    Mean_Train_n = mean(Train_n),
    Mean_Test_n = mean(Test_n),
    
    Mean_Train_R2 = mean(Train_R2),
    SD_Train_R2 = sd(Train_R2),
    CI95_Train_R2 = qt(0.975, df = n() - 1) * sd(Train_R2) / sqrt(n()),
    
    Mean_Test_R2 = mean(Test_R2),
    SD_Test_R2 = sd(Test_R2),
    CI95_Test_R2 = qt(0.975, df = n() - 1) * sd(Test_R2) / sqrt(n()),
    
    Mean_Train_RMSE = mean(Train_RMSE),
    SD_Train_RMSE = sd(Train_RMSE),
    CI95_Train_RMSE = qt(0.975, df = n() - 1) * sd(Train_RMSE) / sqrt(n()),
    
    Mean_Test_RMSE = mean(Test_RMSE),
    SD_Test_RMSE = sd(Test_RMSE),
    CI95_Test_RMSE = qt(0.975, df = n() - 1) * sd(Test_RMSE) / sqrt(n()),
    
    Mean_Train_MAE = mean(Train_MAE),
    SD_Train_MAE = sd(Train_MAE),
    CI95_Train_MAE = qt(0.975, df = n() - 1) * sd(Train_MAE) / sqrt(n()),
    
    Mean_Test_MAE = mean(Test_MAE),
    SD_Test_MAE = sd(Test_MAE),
    CI95_Test_MAE = qt(0.975, df = n() - 1) * sd(Test_MAE) / sqrt(n()),
    
    Mean_R2_gap = mean(R2_gap),
    Mean_RMSE_gap = mean(RMSE_gap),
    Mean_MAE_gap = mean(MAE_gap),
    .groups = "drop"
  )

predictors_used <- data.frame(
  Predictor = predictor_variables
)

configuration <- data.frame(
  Item = c(
    "Group",
    "Input dataset",
    "Output folder",
    "Model",
    "Data type",
    "Preprocessing",
    "Excluded variables",
    "Number of eligible predictors",
    "Seeds",
    "Train-test split",
    "Stratification",
    "Cross-validation",
    "Tuning parameter",
    "Number of trees",
    "Learning-curve fractions"
  ),
  Value = c(
    group_label,
    input_dataset_path,
    output_dir,
    "Random Forest",
    "Original raw HCY",
    "No transformation, no centering, no scaling",
    paste(excluded_variables, collapse = ", "),
    num_predictors,
    paste(seed_values, collapse = ", "),
    "80/20",
    "HCY quintiles",
    "10-fold CV within training set only",
    "mtry",
    ntree_value,
    paste(learning_fractions, collapse = ", ")
  )
)

# ------------------------------
# 12. Plots
# ------------------------------

learning_curve_r2_long <- learning_curve_summary %>%
  select(Fraction, Mean_Train_R2, Mean_Test_R2) %>%
  pivot_longer(
    cols = c(Mean_Train_R2, Mean_Test_R2),
    names_to = "Set",
    values_to = "R2"
  ) %>%
  mutate(
    Set = recode(
      Set,
      "Mean_Train_R2" = "Training",
      "Mean_Test_R2" = "Testing"
    )
  )

plot_learning_r2 <- ggplot(
  learning_curve_r2_long,
  aes(x = Fraction, y = R2, group = Set, linetype = Set, shape = Set)
) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2.5) +
  labs(
    title = paste("Random Forest learning curve - R2 -", group_label),
    x = "Training-set fraction",
    y = expression(R^2)
  ) +
  theme_minimal()

learning_curve_rmse_long <- learning_curve_summary %>%
  select(Fraction, Mean_Train_RMSE, Mean_Test_RMSE) %>%
  pivot_longer(
    cols = c(Mean_Train_RMSE, Mean_Test_RMSE),
    names_to = "Set",
    values_to = "RMSE"
  ) %>%
  mutate(
    Set = recode(
      Set,
      "Mean_Train_RMSE" = "Training",
      "Mean_Test_RMSE" = "Testing"
    )
  )

plot_learning_rmse <- ggplot(
  learning_curve_rmse_long,
  aes(x = Fraction, y = RMSE, group = Set, linetype = Set, shape = Set)
) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2.5) +
  labs(
    title = paste("Random Forest learning curve - RMSE -", group_label),
    x = "Training-set fraction",
    y = "RMSE"
  ) +
  theme_minimal()

learning_curve_mae_long <- learning_curve_summary %>%
  select(Fraction, Mean_Train_MAE, Mean_Test_MAE) %>%
  pivot_longer(
    cols = c(Mean_Train_MAE, Mean_Test_MAE),
    names_to = "Set",
    values_to = "MAE"
  ) %>%
  mutate(
    Set = recode(
      Set,
      "Mean_Train_MAE" = "Training",
      "Mean_Test_MAE" = "Testing"
    )
  )

plot_learning_mae <- ggplot(
  learning_curve_mae_long,
  aes(x = Fraction, y = MAE, group = Set, linetype = Set, shape = Set)
) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2.5) +
  labs(
    title = paste("Random Forest learning curve - MAE -", group_label),
    x = "Training-set fraction",
    y = "MAE"
  ) +
  theme_minimal()

top_importance <- importance_summary %>%
  slice_max(order_by = Mean_Importance, n = 15) %>%
  mutate(
    Variable = reorder(Variable, Mean_Importance)
  )

plot_importance <- ggplot(
  top_importance,
  aes(x = Variable, y = Mean_Importance)
) +
  geom_col() +
  coord_flip() +
  labs(
    title = paste("Random Forest variable importance -", group_label),
    x = "Variable",
    y = "Mean importance"
  ) +
  theme_minimal()

# ------------------------------
# 13. Save plots
# ------------------------------

r2_plot_file <- file.path(
  output_dir,
  paste0("RF_learning_curve_R2_", tolower(group_label), ".png")
)

rmse_plot_file <- file.path(
  output_dir,
  paste0("RF_learning_curve_RMSE_", tolower(group_label), ".png")
)

mae_plot_file <- file.path(
  output_dir,
  paste0("RF_learning_curve_MAE_", tolower(group_label), ".png")
)

importance_plot_file <- file.path(
  output_dir,
  paste0("RF_variable_importance_", tolower(group_label), ".png")
)

ggsave(r2_plot_file, plot_learning_r2, width = 8, height = 5, dpi = 300)
ggsave(rmse_plot_file, plot_learning_rmse, width = 8, height = 5, dpi = 300)
ggsave(mae_plot_file, plot_learning_mae, width = 8, height = 5, dpi = 300)
ggsave(importance_plot_file, plot_importance, width = 8, height = 6, dpi = 300)

# ------------------------------
# 14. Save Excel workbook
# ------------------------------

excel_file <- file.path(
  output_dir,
  paste0("Random_forest_original_HCY_complete_pipeline_", tolower(group_label), ".xlsx")
)

workbook <- createWorkbook()

addWorksheet(workbook, "Summary")
writeData(workbook, "Summary", summary_metrics)

addWorksheet(workbook, "Run_metrics")
writeData(workbook, "Run_metrics", all_run_metrics)

addWorksheet(workbook, "CV_results")
writeData(workbook, "CV_results", all_cv_results)

addWorksheet(workbook, "Hyperparameters_by_seed")
writeData(workbook, "Hyperparameters_by_seed", all_hyperparameters)

addWorksheet(workbook, "Hyperparameter_frequency")
writeData(workbook, "Hyperparameter_frequency", hyperparameter_frequency)

addWorksheet(workbook, "Variable_importance_by_seed")
writeData(workbook, "Variable_importance_by_seed", all_importance)

addWorksheet(workbook, "Variable_importance_summary")
writeData(workbook, "Variable_importance_summary", importance_summary)

addWorksheet(workbook, "Learning_curve_raw")
writeData(workbook, "Learning_curve_raw", all_learning_curves)

addWorksheet(workbook, "Learning_curve_summary")
writeData(workbook, "Learning_curve_summary", learning_curve_summary)

addWorksheet(workbook, "Predictors_used")
writeData(workbook, "Predictors_used", predictors_used)

addWorksheet(workbook, "Configuration")
writeData(workbook, "Configuration", configuration)

saveWorkbook(
  workbook,
  file = excel_file,
  overwrite = TRUE
)

# ------------------------------
# 15. Print final outputs
# ------------------------------

cat("\nExcel workbook saved as:\n")
cat(excel_file, "\n")

cat("\nPlots saved in:\n")
cat(r2_plot_file, "\n")
cat(rmse_plot_file, "\n")
cat(mae_plot_file, "\n")
cat(importance_plot_file, "\n")

cat("\n====================================================\n")
cat("FINAL RANDOM FOREST SUMMARY -", group_label, "\n")
cat("====================================================\n")
print(summary_metrics)

cat("\nMost frequent mtry:\n")
print(hyperparameter_frequency)

cat("\nTop 15 variables by mean importance:\n")
print(head(importance_summary, 15))