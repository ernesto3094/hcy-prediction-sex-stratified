# ============================================================
# Repeated-seed XGBoost Regression
# Full eligible predictors - Male subgroup
#
# - Original-scale HCY
# - No transformation
# - No centering
# - No scaling
# - 80/20 HCY-stratified train-test split
# - 10 repeated seeds
# - 10-fold cross-validation within training set only
# - Controlled XGBoost hyperparameter tuning
# - Independent test-set evaluation
# - Variable importance
# - Learning curves
# - Excel and figures exported to the same previous XGBoost folder
# ============================================================

# ------------------------------------------------------------
# 1. Load required packages
# ------------------------------------------------------------

required_packages <- c(
  "caret",
  "xgboost",
  "dplyr",
  "tidyr",
  "openxlsx",
  "ggplot2",
  "tibble"
)

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

# ------------------------------------------------------------
# 2. General configuration
# ------------------------------------------------------------

group_label <- "Men"

input_dataset_path <- "data/processed/dataset_genero_2_sin_HcyABN3.csv"

response_variable <- "HCY"

model_label <- "XGBoost_full_predictors_original_HCY"

# Same route used in the previous XGBoost code
output_dir <- "results/men/xgboost"

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

seed_values <- c(
  123, 321, 456, 654, 789,
  987, 111, 222, 333, 444
)

cv_folds <- 10

learning_fractions <- c(0.40, 0.60, 0.80, 1.00)

# ------------------------------------------------------------
# 3. Controlled XGBoost tuning grid
# ------------------------------------------------------------
# Moderate grid to avoid excessive runtime.
# Main hyperparameters are tuned, but the grid is intentionally compact.

xgb_grid <- expand.grid(
  nrounds = c(75, 100, 125, 150),
  max_depth = c(1, 2),
  eta = c(0.03, 0.05),
  gamma = c(0, 0.1, 0.5),
  colsample_bytree = c(0.5, 0.6),
  min_child_weight = c(1),
  subsample = c(0.8)
)

cat("\nNumber of XGBoost hyperparameter combinations:", nrow(xgb_grid), "\n")

# ------------------------------------------------------------
# 4. Load original dataset
# ------------------------------------------------------------

data <- read.csv(
  input_dataset_path,
  check.names = TRUE
)

excluded_variables <- c(
  "ID", "Id", "id",
  "Gender", "Genero", "Género", "G.nero",
  "ALU", "LINE", "SAT",
  "HcyABN3",
  "FatCatMasc", "FatCatFem",
  "GrasaCatMasc", "GrasaCatFem"
)

data <- data[, !(names(data) %in% excluded_variables)]

# Keep only numeric variables because caret::xgbTree requires numeric predictors
data <- data %>%
  select(where(is.numeric))

if (!(response_variable %in% names(data))) {
  stop("The response variable 'HCY' was not found in the dataset.")
}

data <- na.omit(data)

predictor_variables <- setdiff(names(data), response_variable)
num_predictors <- length(predictor_variables)

if (num_predictors < 1) {
  stop("No eligible predictors were found after preprocessing.")
}

cat("\nDataset loaded for:", group_label, "\n")
cat("Input dataset:", input_dataset_path, "\n")
cat("Final dimensions:", paste(dim(data), collapse = " x "), "\n")
cat("Number of eligible predictors:", num_predictors, "\n")
cat("HCY scale: original scale\n")
cat("Preprocessing: no transformation, no centering, no scaling\n\n")

cat("Predictors used:\n")
print(predictor_variables)

# ------------------------------------------------------------
# 5. Metric function
# ------------------------------------------------------------

compute_metrics <- function(obs, pred) {
  
  complete_cases <- complete.cases(obs, pred)
  
  obs <- obs[complete_cases]
  pred <- pred[complete_cases]
  
  if (length(obs) < 2) {
    return(data.frame(
      R2 = NA_real_,
      RMSE = NA_real_,
      MAE = NA_real_
    ))
  }
  
  sse <- sum((obs - pred)^2)
  sst <- sum((obs - mean(obs))^2)
  
  r2 <- ifelse(sst == 0, NA_real_, 1 - (sse / sst))
  rmse <- sqrt(mean((obs - pred)^2))
  mae <- mean(abs(obs - pred))
  
  return(data.frame(
    R2 = r2,
    RMSE = rmse,
    MAE = mae
  ))
}

# ------------------------------------------------------------
# 6. Confidence interval function
# ------------------------------------------------------------

calculate_ci95 <- function(x) {
  
  x <- na.omit(x)
  
  if (length(x) < 2) {
    return(NA_real_)
  }
  
  qt(0.975, df = length(x) - 1) * sd(x) / sqrt(length(x))
}

# ------------------------------------------------------------
# 7. Mode function
# ------------------------------------------------------------

get_mode_value <- function(x) {
  
  x <- na.omit(x)
  
  if (length(x) == 0) {
    return(NA_real_)
  }
  
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

# ------------------------------------------------------------
# 8. Robust HCY-stratified train-test split
# ------------------------------------------------------------

create_hcy_stratified_split <- function(dataset, response_variable, seed_value, train_prop = 0.80) {
  
  set.seed(seed_value)
  
  hcy_values <- dataset[[response_variable]]
  
  hcy_cut_points <- unique(
    quantile(
      hcy_values,
      probs = seq(0, 1, by = 0.20),
      na.rm = TRUE
    )
  )
  
  # If quantile stratification is not possible, use a random split
  if (length(hcy_cut_points) < 3) {
    
    train_index <- sample(
      seq_len(nrow(dataset)),
      size = floor(train_prop * nrow(dataset)),
      replace = FALSE
    )
    
    train_data <- dataset[train_index, ]
    test_data  <- dataset[-train_index, ]
    
    return(list(
      train_data = train_data,
      test_data = test_data
    ))
  }
  
  dataset$HCY_strata_tmp <- cut(
    hcy_values,
    breaks = hcy_cut_points,
    include.lowest = TRUE,
    labels = FALSE
  )
  
  index_by_strata <- split(seq_len(nrow(dataset)), dataset$HCY_strata_tmp)
  
  train_index <- unlist(
    lapply(index_by_strata, function(index) {
      n_train <- max(1, floor(length(index) * train_prop))
      sample(index, size = n_train, replace = FALSE)
    })
  )
  
  train_index <- sort(train_index)
  
  train_data <- dataset[train_index, ]
  test_data  <- dataset[-train_index, ]
  
  train_data$HCY_strata_tmp <- NULL
  test_data$HCY_strata_tmp  <- NULL
  
  return(list(
    train_data = train_data,
    test_data = test_data
  ))
}

# ------------------------------------------------------------
# 9. Stratified subsampling function for learning curves
# ------------------------------------------------------------

create_learning_subset <- function(train_dataset, response_variable, fraction, seed_value) {
  
  set.seed(seed_value)
  
  if (fraction >= 1) {
    return(train_dataset)
  }
  
  hcy_values <- train_dataset[[response_variable]]
  
  hcy_cut_points <- unique(
    quantile(
      hcy_values,
      probs = seq(0, 1, by = 0.20),
      na.rm = TRUE
    )
  )
  
  # If stratified subsampling is not possible, use random subsampling
  if (length(hcy_cut_points) < 3) {
    
    subset_index <- sample(
      seq_len(nrow(train_dataset)),
      size = max(2, floor(fraction * nrow(train_dataset))),
      replace = FALSE
    )
    
    return(train_dataset[subset_index, ])
  }
  
  train_dataset$HCY_strata_lc_tmp <- cut(
    hcy_values,
    breaks = hcy_cut_points,
    include.lowest = TRUE,
    labels = FALSE
  )
  
  index_by_strata <- split(seq_len(nrow(train_dataset)), train_dataset$HCY_strata_lc_tmp)
  
  subset_index <- unlist(
    lapply(index_by_strata, function(index) {
      n_subset <- max(1, floor(length(index) * fraction))
      sample(index, size = min(n_subset, length(index)), replace = FALSE)
    })
  )
  
  subset_index <- sort(subset_index)
  
  subset_data <- train_dataset[subset_index, ]
  subset_data$HCY_strata_lc_tmp <- NULL
  
  return(subset_data)
}

# ------------------------------------------------------------
# 10. Containers
# ------------------------------------------------------------

all_run_metrics <- data.frame()
all_cv_results <- data.frame()
all_hyperparameters <- data.frame()
all_importance <- data.frame()
all_learning_curves <- data.frame()

# ------------------------------------------------------------
# 11. Repeated-seed XGBoost pipeline
# ------------------------------------------------------------

for (current_seed in seed_values) {
  
  cat("\n============================================================\n")
  cat("Running XGBoost seed:", current_seed, "\n")
  cat("============================================================\n")
  
  split_object <- create_hcy_stratified_split(
    dataset = data,
    response_variable = response_variable,
    seed_value = current_seed,
    train_prop = 0.80
  )
  
  train_data <- split_object$train_data
  test_data  <- split_object$test_data
  
  cat("Training observations:", nrow(train_data), "\n")
  cat("Testing observations :", nrow(test_data), "\n")
  
  train_control <- trainControl(
    method = "cv",
    number = cv_folds,
    allowParallel = FALSE
  )
  
  set.seed(current_seed)
  
  xgb_model <- tryCatch(
    {
      train(
        HCY ~ .,
        data = train_data,
        method = "xgbTree",
        trControl = train_control,
        tuneGrid = xgb_grid,
        metric = "RMSE",
        verbose = FALSE
      )
    },
    error = function(e) {
      message("Training failed for seed ", current_seed, ": ", e$message)
      return(NULL)
    }
  )
  
  if (is.null(xgb_model)) {
    next
  }
  
  best_params <- xgb_model$bestTune
  
  cat("Best parameters for seed", current_seed, ":\n")
  print(best_params)
  
  # ------------------------------------------------------------
  # 11.1 CV results for this seed
  # ------------------------------------------------------------
  
  cv_results_seed <- xgb_model$results %>%
    mutate(
      Group_label = group_label,
      Seed = current_seed,
      Model = model_label,
      Num_predictors = num_predictors
    ) %>%
    select(
      Group_label, Seed, Model, Num_predictors,
      nrounds, max_depth, eta, gamma,
      colsample_bytree, min_child_weight, subsample,
      RMSE, Rsquared, MAE,
      RMSESD, RsquaredSD, MAESD
    )
  
  all_cv_results <- bind_rows(all_cv_results, cv_results_seed)
  
  # ------------------------------------------------------------
  # 11.2 Train and test predictions
  # ------------------------------------------------------------
  
  train_predictions <- predict(xgb_model, newdata = train_data)
  test_predictions  <- predict(xgb_model, newdata = test_data)
  
  train_metrics <- compute_metrics(
    obs = train_data[[response_variable]],
    pred = train_predictions
  )
  
  test_metrics <- compute_metrics(
    obs = test_data[[response_variable]],
    pred = test_predictions
  )
  
  run_metrics <- data.frame(
    Group_label = group_label,
    Seed = current_seed,
    Model = model_label,
    Data_type = "Original_raw_HCY",
    Num_predictors = num_predictors,
    Train_n = nrow(train_data),
    Test_n = nrow(test_data),
    
    nrounds = best_params$nrounds,
    max_depth = best_params$max_depth,
    eta = best_params$eta,
    gamma = best_params$gamma,
    colsample_bytree = best_params$colsample_bytree,
    min_child_weight = best_params$min_child_weight,
    subsample = best_params$subsample,
    
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
  
  # ------------------------------------------------------------
  # 11.3 Hyperparameters by seed
  # ------------------------------------------------------------
  
  hyper_row <- data.frame(
    Group_label = group_label,
    Seed = current_seed,
    Model = model_label,
    Num_predictors = num_predictors,
    nrounds = best_params$nrounds,
    max_depth = best_params$max_depth,
    eta = best_params$eta,
    gamma = best_params$gamma,
    colsample_bytree = best_params$colsample_bytree,
    min_child_weight = best_params$min_child_weight,
    subsample = best_params$subsample
  )
  
  all_hyperparameters <- bind_rows(all_hyperparameters, hyper_row)
  
  # ------------------------------------------------------------
  # 11.4 Variable importance
  # ------------------------------------------------------------
  
  importance_seed <- tryCatch(
    {
      varImp(xgb_model, scale = FALSE)$importance %>%
        tibble::rownames_to_column("Variable") %>%
        arrange(desc(Overall)) %>%
        mutate(
          Group_label = group_label,
          Seed = current_seed,
          Model = model_label
        ) %>%
        select(Group_label, Seed, Model, Variable, Overall)
    },
    error = function(e) {
      data.frame(
        Group_label = group_label,
        Seed = current_seed,
        Model = model_label,
        Variable = NA_character_,
        Overall = NA_real_
      )
    }
  )
  
  all_importance <- bind_rows(all_importance, importance_seed)
  
  # ------------------------------------------------------------
  # 11.5 Learning curves using selected hyperparameters
  # ------------------------------------------------------------
  
  for (fraction_value in learning_fractions) {
    
    cat("Learning curve fraction:", fraction_value, "\n")
    
    subset_train_data <- create_learning_subset(
      train_dataset = train_data,
      response_variable = response_variable,
      fraction = fraction_value,
      seed_value = current_seed
    )
    
    fixed_grid <- expand.grid(
      nrounds = best_params$nrounds,
      max_depth = best_params$max_depth,
      eta = best_params$eta,
      gamma = best_params$gamma,
      colsample_bytree = best_params$colsample_bytree,
      min_child_weight = best_params$min_child_weight,
      subsample = best_params$subsample
    )
    
    set.seed(current_seed)
    
    xgb_lc_model <- tryCatch(
      {
        train(
          HCY ~ .,
          data = subset_train_data,
          method = "xgbTree",
          trControl = trainControl(method = "none"),
          tuneGrid = fixed_grid,
          metric = "RMSE",
          verbose = FALSE
        )
      },
      error = function(e) {
        message(
          "Learning curve training failed for seed ",
          current_seed,
          " and fraction ",
          fraction_value,
          ": ",
          e$message
        )
        return(NULL)
      }
    )
    
    if (is.null(xgb_lc_model)) {
      next
    }
    
    lc_train_predictions <- predict(xgb_lc_model, newdata = subset_train_data)
    lc_test_predictions  <- predict(xgb_lc_model, newdata = test_data)
    
    lc_train_metrics <- compute_metrics(
      obs = subset_train_data[[response_variable]],
      pred = lc_train_predictions
    )
    
    lc_test_metrics <- compute_metrics(
      obs = test_data[[response_variable]],
      pred = lc_test_predictions
    )
    
    lc_row <- data.frame(
      Group_label = group_label,
      Seed = current_seed,
      Model = model_label,
      Data_type = "Original_raw_HCY",
      Fraction = fraction_value,
      Num_predictors = num_predictors,
      Train_n = nrow(subset_train_data),
      Test_n = nrow(test_data),
      
      nrounds = best_params$nrounds,
      max_depth = best_params$max_depth,
      eta = best_params$eta,
      gamma = best_params$gamma,
      colsample_bytree = best_params$colsample_bytree,
      min_child_weight = best_params$min_child_weight,
      subsample = best_params$subsample,
      
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

# ------------------------------------------------------------
# 12. Stop if no models were successfully fitted
# ------------------------------------------------------------

if (nrow(all_run_metrics) == 0) {
  stop("No XGBoost model was successfully fitted. Please check the dataset and predictors.")
}

# ------------------------------------------------------------
# 13. Summary metrics
# ------------------------------------------------------------

summary_metrics <- all_run_metrics %>%
  summarise(
    Group_label = first(Group_label),
    Model = first(Model),
    Data_type = first(Data_type),
    Runs = n(),
    Num_predictors = first(Num_predictors),
    
    Mean_Train_R2 = mean(Train_R2, na.rm = TRUE),
    SD_Train_R2 = sd(Train_R2, na.rm = TRUE),
    CI95_Train_R2 = calculate_ci95(Train_R2),
    
    Mean_Train_RMSE = mean(Train_RMSE, na.rm = TRUE),
    SD_Train_RMSE = sd(Train_RMSE, na.rm = TRUE),
    CI95_Train_RMSE = calculate_ci95(Train_RMSE),
    
    Mean_Train_MAE = mean(Train_MAE, na.rm = TRUE),
    SD_Train_MAE = sd(Train_MAE, na.rm = TRUE),
    CI95_Train_MAE = calculate_ci95(Train_MAE),
    
    Mean_Test_R2 = mean(Test_R2, na.rm = TRUE),
    SD_Test_R2 = sd(Test_R2, na.rm = TRUE),
    CI95_Test_R2 = calculate_ci95(Test_R2),
    
    Mean_Test_RMSE = mean(Test_RMSE, na.rm = TRUE),
    SD_Test_RMSE = sd(Test_RMSE, na.rm = TRUE),
    CI95_Test_RMSE = calculate_ci95(Test_RMSE),
    
    Mean_Test_MAE = mean(Test_MAE, na.rm = TRUE),
    SD_Test_MAE = sd(Test_MAE, na.rm = TRUE),
    CI95_Test_MAE = calculate_ci95(Test_MAE),
    
    Mean_R2_gap = mean(R2_gap, na.rm = TRUE),
    Mean_RMSE_gap = mean(RMSE_gap, na.rm = TRUE),
    Mean_MAE_gap = mean(MAE_gap, na.rm = TRUE),
    
    Most_frequent_nrounds = get_mode_value(nrounds),
    Most_frequent_max_depth = get_mode_value(max_depth),
    Most_frequent_eta = get_mode_value(eta),
    Most_frequent_gamma = get_mode_value(gamma),
    Most_frequent_colsample_bytree = get_mode_value(colsample_bytree),
    Most_frequent_min_child_weight = get_mode_value(min_child_weight),
    Most_frequent_subsample = get_mode_value(subsample)
  )

# ------------------------------------------------------------
# 14. Hyperparameter frequency table
# ------------------------------------------------------------

hyperparameter_frequency <- all_hyperparameters %>%
  count(
    nrounds,
    max_depth,
    eta,
    gamma,
    colsample_bytree,
    min_child_weight,
    subsample,
    name = "Frequency"
  ) %>%
  mutate(
    Percentage = 100 * Frequency / sum(Frequency)
  ) %>%
  arrange(
    desc(Frequency),
    max_depth,
    nrounds,
    eta
  )

# ------------------------------------------------------------
# 15. Variable importance summary
# ------------------------------------------------------------

importance_summary <- all_importance %>%
  filter(!is.na(Variable)) %>%
  group_by(Variable) %>%
  summarise(
    Mean_Importance = mean(Overall, na.rm = TRUE),
    SD_Importance = sd(Overall, na.rm = TRUE),
    CI95_Importance = calculate_ci95(Overall),
    Times_Reported = n(),
    .groups = "drop"
  ) %>%
  arrange(desc(Mean_Importance))

predictors_used <- data.frame(
  Predictor = predictor_variables
)

# ------------------------------------------------------------
# 16. Learning curve summary
# ------------------------------------------------------------

learning_curve_summary <- all_learning_curves %>%
  group_by(Group_label, Model, Fraction) %>%
  summarise(
    Mean_Train_n = mean(Train_n, na.rm = TRUE),
    Mean_Test_n = mean(Test_n, na.rm = TRUE),
    
    Mean_Train_R2 = mean(Train_R2, na.rm = TRUE),
    SD_Train_R2 = sd(Train_R2, na.rm = TRUE),
    CI95_Train_R2 = calculate_ci95(Train_R2),
    
    Mean_Test_R2 = mean(Test_R2, na.rm = TRUE),
    SD_Test_R2 = sd(Test_R2, na.rm = TRUE),
    CI95_Test_R2 = calculate_ci95(Test_R2),
    
    Mean_Train_RMSE = mean(Train_RMSE, na.rm = TRUE),
    SD_Train_RMSE = sd(Train_RMSE, na.rm = TRUE),
    CI95_Train_RMSE = calculate_ci95(Train_RMSE),
    
    Mean_Test_RMSE = mean(Test_RMSE, na.rm = TRUE),
    SD_Test_RMSE = sd(Test_RMSE, na.rm = TRUE),
    CI95_Test_RMSE = calculate_ci95(Test_RMSE),
    
    Mean_Train_MAE = mean(Train_MAE, na.rm = TRUE),
    SD_Train_MAE = sd(Train_MAE, na.rm = TRUE),
    CI95_Train_MAE = calculate_ci95(Train_MAE),
    
    Mean_Test_MAE = mean(Test_MAE, na.rm = TRUE),
    SD_Test_MAE = sd(Test_MAE, na.rm = TRUE),
    CI95_Test_MAE = calculate_ci95(Test_MAE),
    
    Mean_R2_gap = mean(R2_gap, na.rm = TRUE),
    Mean_RMSE_gap = mean(RMSE_gap, na.rm = TRUE),
    Mean_MAE_gap = mean(MAE_gap, na.rm = TRUE),
    
    .groups = "drop"
  )

# ------------------------------------------------------------
# 17. Configuration table
# ------------------------------------------------------------

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
    "Tuning grid size",
    "Learning-curve fractions"
  ),
  Value = c(
    group_label,
    input_dataset_path,
    output_dir,
    "XGBoost regression using caret::xgbTree",
    "Original raw HCY",
    "No transformation, no centering, no scaling",
    paste(excluded_variables, collapse = ", "),
    num_predictors,
    paste(seed_values, collapse = ", "),
    "80/20",
    "HCY quintiles",
    paste0(cv_folds, "-fold CV within training set only"),
    nrow(xgb_grid),
    paste(learning_fractions, collapse = ", ")
  )
)

# ------------------------------------------------------------
# 18. Learning curve plots
# ------------------------------------------------------------

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
    title = paste("XGBoost learning curve - R2 -", group_label),
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
    title = paste("XGBoost learning curve - RMSE -", group_label),
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
    title = paste("XGBoost learning curve - MAE -", group_label),
    x = "Training-set fraction",
    y = "MAE"
  ) +
  theme_minimal()

# ------------------------------------------------------------
# 19. Variable importance plot
# ------------------------------------------------------------

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
    title = paste("XGBoost variable importance -", group_label),
    x = "Variable",
    y = "Mean importance"
  ) +
  theme_minimal()

# ------------------------------------------------------------
# 20. Save plots
# ------------------------------------------------------------

r2_plot_file <- file.path(
  output_dir,
  paste0("XGBoost_full_predictors_learning_curve_R2_", tolower(group_label), ".png")
)

rmse_plot_file <- file.path(
  output_dir,
  paste0("XGBoost_full_predictors_learning_curve_RMSE_", tolower(group_label), ".png")
)

mae_plot_file <- file.path(
  output_dir,
  paste0("XGBoost_full_predictors_learning_curve_MAE_", tolower(group_label), ".png")
)

importance_plot_file <- file.path(
  output_dir,
  paste0("XGBoost_full_predictors_variable_importance_", tolower(group_label), ".png")
)

ggsave(r2_plot_file, plot_learning_r2, width = 8, height = 5, dpi = 300)
ggsave(rmse_plot_file, plot_learning_rmse, width = 8, height = 5, dpi = 300)
ggsave(mae_plot_file, plot_learning_mae, width = 8, height = 5, dpi = 300)
ggsave(importance_plot_file, plot_importance, width = 8, height = 6, dpi = 300)

# ------------------------------------------------------------
# 21. Save Excel workbook
# ------------------------------------------------------------

excel_file <- file.path(
  output_dir,
  paste0(model_label, "_complete_pipeline_", tolower(group_label), ".xlsx")
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

for (sheet in names(workbook)) {
  setColWidths(workbook, sheet = sheet, cols = 1:100, widths = "auto")
}

saveWorkbook(
  workbook,
  file = excel_file,
  overwrite = TRUE
)

# ------------------------------------------------------------
# 22. Print final outputs
# ------------------------------------------------------------

cat("\n============================================================\n")
cat("FINAL XGBOOST SUMMARY -", group_label, "\n")
cat("============================================================\n")

cat("\nExcel workbook saved as:\n")
cat(excel_file, "\n")

cat("\nPlots saved as:\n")
cat(r2_plot_file, "\n")
cat(rmse_plot_file, "\n")
cat(mae_plot_file, "\n")
cat(importance_plot_file, "\n")

cat("\nSummary metrics:\n")
print(summary_metrics)

cat("\nHyperparameter frequency:\n")
print(hyperparameter_frequency)

cat("\nTop 15 variables by mean importance:\n")
print(head(importance_summary, 15))

cat("\nLearning curve summary:\n")
print(learning_curve_summary)