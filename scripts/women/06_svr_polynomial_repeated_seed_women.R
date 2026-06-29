# ============================================================
# Repeated-seed Support Vector Regression (SVR)
# Polynomial kernel - Female subgroup
# Final hyperparameters from screening:
# degree = 2, scale = 0.01, C = 1
# Transformed-response workflow with HCY back-transformation
# Groups: Stable_100 to Stable_60
# Outputs: train/test metrics, mean, SD, 95% CI, Excel, learning curves
# ============================================================

# ------------------------------------------------------------
# 1. Load packages
# ------------------------------------------------------------

required_packages <- c(
  "caret",
  "kernlab",
  "dplyr",
  "tidyr",
  "openxlsx",
  "ggplot2"
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

group_label <- "Women"

input_dataset_path <- "data/processed/datos_transformados_mujeres.csv"

response_variable <- "HCY"

model_label <- "SVR_polynomial"

kernel_label <- "polynomial"

# Final selected hyperparameters from screening
final_degree <- 2
final_scale  <- 0.01
final_C      <- 1

seeds <- c(123, 321, 456, 654, 789, 987, 111, 222, 333, 444)

excluded_variables <- c(
  "ID",
  "Gender",
  "Genero",
  "ALU",
  "LINE",
  "SAT",
  "HcyABN3",
  "FatCatMasc",
  "FatCatFem",
  "GrasaCatMasc",
  "GrasaCatFem"
)

# ------------------------------------------------------------
# 3. Stability-based accumulated groups
# Female subgroup
# ------------------------------------------------------------

stable_groups <- list(
  
  Stable_100 = c(
    "Edad", "GLU", "IMCCat", "LDL", "MetabBasal"
  ),
  
  Stable_90 = c(
    "Edad", "GLU", "IMCCat", "LDL", "MetabBasal",
    "LDLHDLIndice", "MCHC"
  ),
  
  Stable_80 = c(
    "Edad", "GLU", "IMCCat", "LDL", "MetabBasal",
    "LDLHDLIndice", "MCHC", "ColHDLIndice", "HDL"
  ),
  
  Stable_70 = c(
    "Edad", "GLU", "IMCCat", "LDL", "MetabBasal",
    "LDLHDLIndice", "MCHC", "ColHDLIndice", "HDL",
    "GrasaVisc", "LCI", "MCH", "PLT", "Peso"
  ),
  
  Stable_60 = c(
    "Edad", "GLU", "IMCCat", "LDL", "MetabBasal",
    "LDLHDLIndice", "MCHC", "ColHDLIndice", "HDL",
    "GrasaVisc", "LCI", "MCH", "PLT", "Peso",
    "AIP", "IMC", "RBC", "RDWCV", "VCM"
  )
)

# ------------------------------------------------------------
# 4. Load transformed dataset
# ------------------------------------------------------------

data <- read.csv(input_dataset_path)

data <- data[, !(names(data) %in% excluded_variables)]

data <- data %>%
  select(where(is.numeric))

if (!(response_variable %in% names(data))) {
  stop("The response variable HCY was not found in the dataset.")
}

data <- na.omit(data)

cat("\nDataset loaded for:", group_label, "\n")
cat("Dimensions after preprocessing:", dim(data), "\n")

# ------------------------------------------------------------
# 5. Check predictors
# ------------------------------------------------------------

all_group_predictors <- unique(unlist(stable_groups))
missing_predictors <- setdiff(all_group_predictors, names(data))

if (length(missing_predictors) > 0) {
  stop(
    paste(
      "The following predictors are missing in the dataset:",
      paste(missing_predictors, collapse = ", ")
    )
  )
}

# ------------------------------------------------------------
# 6. HCY-stratified 80/20 split
# ------------------------------------------------------------

create_hcy_stratified_split <- function(data, response_variable, seed) {
  
  set.seed(seed)
  
  quantile_breaks <- quantile(
    data[[response_variable]],
    probs = seq(0, 1, by = 0.20),
    na.rm = TRUE
  )
  
  quantile_breaks <- unique(quantile_breaks)
  
  if (length(quantile_breaks) < 3) {
    stop("Not enough unique HCY quantile breaks to create strata.")
  }
  
  data$strata_hcy <- cut(
    data[[response_variable]],
    breaks = quantile_breaks,
    include.lowest = TRUE,
    labels = FALSE
  )
  
  train_indices <- c()
  
  for (s in unique(data$strata_hcy)) {
    
    stratum_indices <- which(data$strata_hcy == s)
    n_train <- floor(0.80 * length(stratum_indices))
    
    sampled_indices <- sample(
      stratum_indices,
      size = n_train,
      replace = FALSE
    )
    
    train_indices <- c(train_indices, sampled_indices)
  }
  
  train_data <- data[train_indices, ]
  test_data  <- data[-train_indices, ]
  
  train_data$strata_hcy <- NULL
  test_data$strata_hcy  <- NULL
  
  return(list(
    train_data = train_data,
    test_data = test_data
  ))
}

# ------------------------------------------------------------
# 7. Back-transformation
# ------------------------------------------------------------
# HCY was transformed as log(HCY + 1).
# Therefore, original HCY = exp(transformed HCY) - 1.

back_transform_hcy <- function(x) {
  exp(x) - 1
}

# ------------------------------------------------------------
# 8. Metric function in original HCY scale
# ------------------------------------------------------------

calculate_metrics <- function(obs_original, pred_original) {
  
  complete_cases <- complete.cases(obs_original, pred_original)
  
  obs_clean <- obs_original[complete_cases]
  pred_clean <- pred_original[complete_cases]
  
  if (length(obs_clean) < 2) {
    return(data.frame(
      R2 = NA_real_,
      RMSE = NA_real_,
      MAE = NA_real_
    ))
  }
  
  sst <- sum((obs_clean - mean(obs_clean))^2)
  sse <- sum((obs_clean - pred_clean)^2)
  
  r2_value <- ifelse(sst == 0, NA_real_, 1 - (sse / sst))
  rmse_value <- sqrt(mean((obs_clean - pred_clean)^2))
  mae_value  <- mean(abs(obs_clean - pred_clean))
  
  return(data.frame(
    R2 = r2_value,
    RMSE = rmse_value,
    MAE = mae_value
  ))
}

# ------------------------------------------------------------
# 9. Formula builder
# ------------------------------------------------------------

create_svr_formula <- function(response_variable, predictors) {
  as.formula(
    paste(response_variable, "~", paste(predictors, collapse = " + "))
  )
}

# ------------------------------------------------------------
# 10. Train and evaluate final polynomial SVR
# ------------------------------------------------------------

train_and_evaluate_svr_poly <- function(
    group_name,
    predictors,
    train_data,
    test_data,
    seed
) {
  
  set.seed(seed)
  
  model_formula <- create_svr_formula(
    response_variable = response_variable,
    predictors = predictors
  )
  
  fitted_model <- tryCatch(
    {
      ksvm(
        model_formula,
        data = train_data,
        kernel = "polydot",
        kpar = list(
          degree = final_degree,
          scale = final_scale,
          offset = 1
        ),
        C = final_C,
        type = "eps-svr",
        scaled = TRUE
      )
    },
    error = function(e) {
      message("Training failed for ", group_name, " seed ", seed, ": ", e$message)
      return(NULL)
    }
  )
  
  if (is.null(fitted_model)) {
    
    return(data.frame(
      Group_label = group_label,
      Seed = seed,
      Stability_group = group_name,
      Model = model_label,
      Kernel = kernel_label,
      Degree = final_degree,
      Scale = final_scale,
      C = final_C,
      Num_predictors = length(predictors),
      Predictors = paste(predictors, collapse = ", "),
      Model_status = "Training failed",
      
      Train_R2 = NA_real_,
      Train_RMSE = NA_real_,
      Train_MAE = NA_real_,
      
      Test_R2 = NA_real_,
      Test_RMSE = NA_real_,
      Test_MAE = NA_real_,
      
      R2_gap = NA_real_,
      RMSE_gap = NA_real_,
      MAE_gap = NA_real_
    ))
  }
  
  pred_train_transformed <- predict(fitted_model, newdata = train_data)
  pred_test_transformed  <- predict(fitted_model, newdata = test_data)
  
  pred_train_original <- back_transform_hcy(pred_train_transformed)
  pred_test_original  <- back_transform_hcy(pred_test_transformed)
  
  obs_train_original <- back_transform_hcy(train_data[[response_variable]])
  obs_test_original  <- back_transform_hcy(test_data[[response_variable]])
  
  train_metrics <- calculate_metrics(
    obs_original = obs_train_original,
    pred_original = pred_train_original
  )
  
  test_metrics <- calculate_metrics(
    obs_original = obs_test_original,
    pred_original = pred_test_original
  )
  
  result <- data.frame(
    Group_label = group_label,
    Seed = seed,
    Stability_group = group_name,
    Model = model_label,
    Kernel = kernel_label,
    Degree = final_degree,
    Scale = final_scale,
    C = final_C,
    Num_predictors = length(predictors),
    Predictors = paste(predictors, collapse = ", "),
    Model_status = "OK",
    
    Train_R2 = train_metrics$R2,
    Train_RMSE = train_metrics$RMSE,
    Train_MAE = train_metrics$MAE,
    
    Test_R2 = test_metrics$R2,
    Test_RMSE = test_metrics$RMSE,
    Test_MAE = test_metrics$MAE,
    
    R2_gap = train_metrics$R2 - test_metrics$R2,
    RMSE_gap = test_metrics$RMSE - train_metrics$RMSE,
    MAE_gap = test_metrics$MAE - train_metrics$MAE
  )
  
  return(result)
}

# ------------------------------------------------------------
# 11. Run repeated-seed polynomial SVR experiment
# ------------------------------------------------------------

all_results <- list()
counter <- 1

for (current_seed in seeds) {
  
  cat("\n============================================================\n")
  cat("Running polynomial SVR for seed:", current_seed, "\n")
  cat("============================================================\n")
  
  split_data <- create_hcy_stratified_split(
    data = data,
    response_variable = response_variable,
    seed = current_seed
  )
  
  train_data <- split_data$train_data
  test_data  <- split_data$test_data
  
  for (group_name in names(stable_groups)) {
    
    predictors <- stable_groups[[group_name]]
    
    cat(
      "Seed:", current_seed,
      "| Group:", group_name,
      "| Model:", model_label,
      "| degree:", final_degree,
      "| scale:", final_scale,
      "| C:", final_C,
      "\n"
    )
    
    model_result <- train_and_evaluate_svr_poly(
      group_name = group_name,
      predictors = predictors,
      train_data = train_data,
      test_data = test_data,
      seed = current_seed
    )
    
    all_results[[counter]] <- model_result
    counter <- counter + 1
  }
}

results_df <- bind_rows(all_results)

# ------------------------------------------------------------
# 12. Confidence interval function
# ------------------------------------------------------------

calculate_ci95 <- function(x) {
  
  x_clean <- na.omit(x)
  n <- length(x_clean)
  
  if (n < 2) {
    return(NA_real_)
  }
  
  error <- qt(0.975, df = n - 1) * sd(x_clean) / sqrt(n)
  
  return(error)
}

# ------------------------------------------------------------
# 13. Summary results
# ------------------------------------------------------------

summary_results <- results_df %>%
  group_by(
    Group_label,
    Stability_group,
    Model,
    Kernel,
    Degree,
    Scale,
    C,
    Num_predictors
  ) %>%
  summarise(
    Runs = sum(!is.na(Test_RMSE)),
    
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
    
    .groups = "drop"
  ) %>%
  arrange(desc(Mean_Test_R2), Mean_Test_RMSE, Mean_Test_MAE)

# ------------------------------------------------------------
# 14. Learning curves for polynomial SVR
# ------------------------------------------------------------

training_fractions <- c(0.4, 0.6, 0.8, 1.0)

learning_curve_results <- list()
lc_counter <- 1

for (current_seed in seeds) {
  
  cat("\n============================================================\n")
  cat("Learning curve polynomial SVR - seed:", current_seed, "\n")
  cat("============================================================\n")
  
  split_data <- create_hcy_stratified_split(
    data = data,
    response_variable = response_variable,
    seed = current_seed
  )
  
  full_train_data <- split_data$train_data
  test_data <- split_data$test_data
  
  for (group_name in names(stable_groups)) {
    
    predictors <- stable_groups[[group_name]]
    
    for (fraction in training_fractions) {
      
      set.seed(current_seed)
      
      n_fraction <- floor(nrow(full_train_data) * fraction)
      
      sampled_rows <- sample(
        seq_len(nrow(full_train_data)),
        size = n_fraction,
        replace = FALSE
      )
      
      partial_train_data <- full_train_data[sampled_rows, ]
      
      cat(
        "Seed:", current_seed,
        "| Group:", group_name,
        "| Fraction:", fraction,
        "| Train n:", nrow(partial_train_data), "\n"
      )
      
      lc_result <- train_and_evaluate_svr_poly(
        group_name = group_name,
        predictors = predictors,
        train_data = partial_train_data,
        test_data = test_data,
        seed = current_seed
      )
      
      lc_result$Training_fraction <- fraction
      lc_result$Training_n <- nrow(partial_train_data)
      
      learning_curve_results[[lc_counter]] <- lc_result
      lc_counter <- lc_counter + 1
    }
  }
}

learning_curve_df <- bind_rows(learning_curve_results)

learning_curve_summary <- learning_curve_df %>%
  group_by(
    Group_label,
    Stability_group,
    Model,
    Kernel,
    Degree,
    Scale,
    C,
    Training_fraction
  ) %>%
  summarise(
    Mean_Training_n = mean(Training_n, na.rm = TRUE),
    
    Mean_Train_R2 = mean(Train_R2, na.rm = TRUE),
    SD_Train_R2 = sd(Train_R2, na.rm = TRUE),
    Mean_Test_R2 = mean(Test_R2, na.rm = TRUE),
    SD_Test_R2 = sd(Test_R2, na.rm = TRUE),
    
    Mean_Train_RMSE = mean(Train_RMSE, na.rm = TRUE),
    SD_Train_RMSE = sd(Train_RMSE, na.rm = TRUE),
    Mean_Test_RMSE = mean(Test_RMSE, na.rm = TRUE),
    SD_Test_RMSE = sd(Test_RMSE, na.rm = TRUE),
    
    Mean_Train_MAE = mean(Train_MAE, na.rm = TRUE),
    SD_Train_MAE = sd(Train_MAE, na.rm = TRUE),
    Mean_Test_MAE = mean(Test_MAE, na.rm = TRUE),
    SD_Test_MAE = sd(Test_MAE, na.rm = TRUE),
    
    .groups = "drop"
  )

# ------------------------------------------------------------
# 15. Learning curve plots
# ------------------------------------------------------------

learning_curve_r2_long <- learning_curve_summary %>%
  select(
    Stability_group,
    Training_fraction,
    Mean_Train_R2,
    Mean_Test_R2
  ) %>%
  pivot_longer(
    cols = c(Mean_Train_R2, Mean_Test_R2),
    names_to = "Curve",
    values_to = "R2"
  )

p_r2 <- ggplot(
  learning_curve_r2_long,
  aes(
    x = Training_fraction,
    y = R2,
    linetype = Curve,
    group = Curve
  )
) +
  geom_line() +
  geom_point() +
  facet_wrap(~ Stability_group, scales = "free_y") +
  labs(
    title = paste("Polynomial SVR learning curves based on R2 -", group_label),
    x = "Training fraction",
    y = "R2"
  ) +
  theme_bw()

learning_curve_rmse_long <- learning_curve_summary %>%
  select(
    Stability_group,
    Training_fraction,
    Mean_Train_RMSE,
    Mean_Test_RMSE
  ) %>%
  pivot_longer(
    cols = c(Mean_Train_RMSE, Mean_Test_RMSE),
    names_to = "Curve",
    values_to = "RMSE"
  )

p_rmse <- ggplot(
  learning_curve_rmse_long,
  aes(
    x = Training_fraction,
    y = RMSE,
    linetype = Curve,
    group = Curve
  )
) +
  geom_line() +
  geom_point() +
  facet_wrap(~ Stability_group, scales = "free_y") +
  labs(
    title = paste("Polynomial SVR learning curves based on RMSE -", group_label),
    x = "Training fraction",
    y = "RMSE"
  ) +
  theme_bw()

print(p_r2)
print(p_rmse)

learning_curve_r2_file <- paste0(
  "SVR_poly_learning_curve_R2_",
  tolower(group_label),
  ".png"
)

learning_curve_rmse_file <- paste0(
  "SVR_poly_learning_curve_RMSE_",
  tolower(group_label),
  ".png"
)

ggsave(
  filename = learning_curve_r2_file,
  plot = p_r2,
  width = 12,
  height = 7,
  dpi = 300
)

ggsave(
  filename = learning_curve_rmse_file,
  plot = p_rmse,
  width = 12,
  height = 7,
  dpi = 300
)

# ------------------------------------------------------------
# 16. Export to Excel
# ------------------------------------------------------------

results_df_export <- results_df %>%
  select(-any_of("Model_status"))

summary_results_export <- summary_results %>%
  select(-any_of("Model_status"))

learning_curve_df_export <- learning_curve_df %>%
  select(-any_of("Model_status"))

learning_curve_summary_export <- learning_curve_summary %>%
  select(-any_of("Model_status"))

output_excel_file <- paste0(
  "SVR_poly_repeated_seed_results_",
  tolower(group_label),
  ".xlsx"
)

wb <- createWorkbook()

addWorksheet(wb, "Detailed_by_seed")
writeData(wb, "Detailed_by_seed", results_df_export)

addWorksheet(wb, "Summary")
writeData(wb, "Summary", summary_results_export)

addWorksheet(wb, "Learning_curve_raw")
writeData(wb, "Learning_curve_raw", learning_curve_df_export)

addWorksheet(wb, "Learning_curve_summary")
writeData(wb, "Learning_curve_summary", learning_curve_summary_export)

for (sheet in names(wb)) {
  setColWidths(wb, sheet = sheet, cols = 1:100, widths = "auto")
}

saveWorkbook(wb, output_excel_file, overwrite = TRUE)

# ------------------------------------------------------------
# 17. Final output
# ------------------------------------------------------------

cat("\n============================================================\n")
cat("Polynomial SVR repeated-seed modeling completed for:", group_label, "\n")
cat("============================================================\n\n")

cat("Final hyperparameters:\n")
cat("degree =", final_degree, "\n")
cat("scale  =", final_scale, "\n")
cat("C      =", final_C, "\n\n")

cat("Excel file saved as:", output_excel_file, "\n")
cat("Learning curve R2 plot saved as:", learning_curve_r2_file, "\n")
cat("Learning curve RMSE plot saved as:", learning_curve_rmse_file, "\n\n")

cat("Summary results:\n")
print(summary_results, n = Inf, width = Inf)