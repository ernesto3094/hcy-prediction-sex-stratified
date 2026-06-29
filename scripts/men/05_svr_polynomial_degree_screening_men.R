# ============================================================
# SVR Polynomial Kernel Hyperparameter Screening
# Male subgroup
#
# Purpose:
#   - Test polynomial kernel degrees from 2 to 9
#   - Tune scale and C
#   - Use 10 random seeds
#   - Use stability-based accumulated groups: Stable_100 to Stable_60
#   - Use HCY-stratified 80/20 split
#   - Evaluate tuning by 10-fold CV inside the training set
#   - Select the simplest useful polynomial degree
#   - Export screening results to Excel
#   - Generate CV R2 and CV RMSE plots
#
# Note:
#   This is only the tuning/screening phase.
#   Final train/test metrics, back-transformation and learning curves
#   will be computed later using the selected hyperparameters.
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

group_label <- "Men"

input_dataset_path <- "data/processed/datos_transformados_hombres.csv"

response_variable <- "HCY"

seeds <- c(123, 321, 456, 654, 789, 987, 111, 222, 333, 444)

# Polynomial SVR tuning grid
# degree = polynomial kernel degree
# scale  = polynomial kernel scale
# C      = penalty parameter

svr_poly_grid <- expand.grid(
  degree = 2:9,
  scale  = c(0.01, 0.05, 0.1),
  C      = c(0.1, 1, 10)
)

# Minimum R2 difference allowed from the best hyperparameter setting.
# The selected configuration will favor simpler degree if performance is similar.
minimum_r2_difference <- 0.01

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
# Male subgroup
# ------------------------------------------------------------

stable_groups <- list(
  
  Stable_100 = c(
    "GrasaVisCat", "HDL", "IMCCat"
  ),
  
  Stable_90 = c(
    "GrasaVisCat", "HDL", "IMCCat",
    "IMC", "LDLHDLIndice"
  ),
  
  Stable_80 = c(
    "GrasaVisCat", "HDL", "IMCCat",
    "IMC", "LDLHDLIndice",
    "AIP", "COL", "GrasaVisc", "Hto", "RDWCV", "Talla"
  ),
  
  Stable_70 = c(
    "GrasaVisCat", "HDL", "IMCCat",
    "IMC", "LDLHDLIndice",
    "AIP", "COL", "GrasaVisc", "Hto", "RDWCV", "Talla",
    "ColHDLIndice", "Hgb"
  ),
  
  Stable_60 = c(
    "GrasaVisCat", "HDL", "IMCCat",
    "IMC", "LDLHDLIndice",
    "AIP", "COL", "GrasaVisc", "Hto", "RDWCV", "Talla",
    "ColHDLIndice", "Hgb", "MCH", "PLT", "VCM"
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
# 7. Formula builder
# ------------------------------------------------------------

create_svr_formula <- function(response_variable, predictors) {
  as.formula(
    paste(response_variable, "~", paste(predictors, collapse = " + "))
  )
}

# ------------------------------------------------------------
# 8. Run polynomial SVR hyperparameter screening
# ------------------------------------------------------------

screening_results <- list()
counter <- 1

for (current_seed in seeds) {
  
  cat("\n============================================================\n")
  cat("SVR polynomial screening - seed:", current_seed, "\n")
  cat("============================================================\n")
  
  split_data <- create_hcy_stratified_split(
    data = data,
    response_variable = response_variable,
    seed = current_seed
  )
  
  train_data <- split_data$train_data
  
  for (group_name in names(stable_groups)) {
    
    predictors <- stable_groups[[group_name]]
    
    cat(
      "Seed:", current_seed,
      "| Group:", group_name,
      "| SVR polynomial tuning\n"
    )
    
    set.seed(current_seed)
    
    train_control <- trainControl(
      method = "cv",
      number = 10
    )
    
    model_formula <- create_svr_formula(
      response_variable = response_variable,
      predictors = predictors
    )
    
    fitted_model <- tryCatch(
      {
        train(
          model_formula,
          data = train_data,
          method = "svmPoly",
          trControl = train_control,
          tuneGrid = svr_poly_grid,
          preProcess = c("center", "scale"),
          metric = "RMSE"
        )
      },
      error = function(e) {
        message("Training error: ", e$message)
        return(NULL)
      }
    )
    
    if (is.null(fitted_model)) {
      
      screening_results[[counter]] <- data.frame(
        Group_label = group_label,
        Seed = current_seed,
        Stability_group = group_name,
        Num_predictors = length(predictors),
        Best_degree = NA_real_,
        Best_scale = NA_real_,
        Best_C = NA_real_,
        CV_RMSE = NA_real_,
        CV_R2 = NA_real_,
        CV_MAE = NA_real_,
        Model_status = "Training failed"
      )
      
    } else {
      
      best_params <- fitted_model$bestTune
      
      best_row <- fitted_model$results %>%
        filter(
          degree == best_params$degree,
          scale == best_params$scale,
          C == best_params$C
        )
      
      screening_results[[counter]] <- data.frame(
        Group_label = group_label,
        Seed = current_seed,
        Stability_group = group_name,
        Num_predictors = length(predictors),
        Best_degree = best_params$degree,
        Best_scale = best_params$scale,
        Best_C = best_params$C,
        CV_RMSE = best_row$RMSE,
        CV_R2 = best_row$Rsquared,
        CV_MAE = best_row$MAE,
        Model_status = "OK"
      )
    }
    
    counter <- counter + 1
  }
}

screening_df <- bind_rows(screening_results)

# ------------------------------------------------------------
# 9. Confidence interval function
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
# 10. Function for most frequent hyperparameter
# ------------------------------------------------------------

get_mode_value <- function(x) {
  
  x_clean <- na.omit(x)
  
  if (length(x_clean) == 0) {
    return(NA_real_)
  }
  
  ux <- unique(x_clean)
  ux[which.max(tabulate(match(x_clean, ux)))]
}

# ------------------------------------------------------------
# 11. Summarize screening results by stability group
# ------------------------------------------------------------

screening_summary <- screening_df %>%
  group_by(
    Group_label,
    Stability_group,
    Num_predictors
  ) %>%
  summarise(
    Runs = sum(Model_status == "OK"),
    
    Most_frequent_degree = get_mode_value(Best_degree),
    Most_frequent_scale = get_mode_value(Best_scale),
    Most_frequent_C = get_mode_value(Best_C),
    
    Median_degree = median(Best_degree, na.rm = TRUE),
    Median_scale = median(Best_scale, na.rm = TRUE),
    Median_C = median(Best_C, na.rm = TRUE),
    
    Mean_CV_RMSE = mean(CV_RMSE, na.rm = TRUE),
    SD_CV_RMSE = sd(CV_RMSE, na.rm = TRUE),
    CI95_CV_RMSE = calculate_ci95(CV_RMSE),
    
    Mean_CV_R2 = mean(CV_R2, na.rm = TRUE),
    SD_CV_R2 = sd(CV_R2, na.rm = TRUE),
    CI95_CV_R2 = calculate_ci95(CV_R2),
    
    Mean_CV_MAE = mean(CV_MAE, na.rm = TRUE),
    SD_CV_MAE = sd(CV_MAE, na.rm = TRUE),
    CI95_CV_MAE = calculate_ci95(CV_MAE),
    
    .groups = "drop"
  ) %>%
  arrange(desc(Mean_CV_R2), Mean_CV_RMSE, Mean_CV_MAE)

# ------------------------------------------------------------
# 12. Hyperparameter frequency table
# ------------------------------------------------------------

hyperparameter_frequency <- screening_df %>%
  filter(Model_status == "OK") %>%
  group_by(
    Group_label,
    Stability_group,
    Best_degree,
    Best_scale,
    Best_C
  ) %>%
  summarise(
    Frequency = n(),
    Mean_CV_RMSE = mean(CV_RMSE, na.rm = TRUE),
    Mean_CV_R2 = mean(CV_R2, na.rm = TRUE),
    Mean_CV_MAE = mean(CV_MAE, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(
    Stability_group,
    desc(Frequency),
    Mean_CV_RMSE,
    desc(Mean_CV_R2),
    Best_degree
  )

# ------------------------------------------------------------
# 13. Select best useful hyperparameters by group
# ------------------------------------------------------------
# Logic:
#   1. For each stability group, find the best CV R2.
#   2. Keep configurations within 0.01 of the best CV R2.
#   3. Select the simplest degree among those.
#   4. If tied, choose lower RMSE, then lower C, then lower scale.

selected_hyperparameters_by_group <- hyperparameter_frequency %>%
  group_by(Group_label, Stability_group) %>%
  mutate(
    Best_CV_R2_in_group = max(Mean_CV_R2, na.rm = TRUE),
    R2_difference_from_best = Best_CV_R2_in_group - Mean_CV_R2
  ) %>%
  filter(R2_difference_from_best <= minimum_r2_difference) %>%
  arrange(
    Best_degree,
    Mean_CV_RMSE,
    Best_C,
    Best_scale
  ) %>%
  slice(1) %>%
  ungroup()

# ------------------------------------------------------------
# 14. Overall selected hyperparameters
# ------------------------------------------------------------
# Selects the most frequently selected combination across all groups/seeds.
# If tied, chooses the simpler degree and then lower RMSE.

overall_selected_hyperparameters <- screening_df %>%
  filter(Model_status == "OK") %>%
  group_by(Best_degree, Best_scale, Best_C) %>%
  summarise(
    Frequency = n(),
    Mean_CV_RMSE = mean(CV_RMSE, na.rm = TRUE),
    Mean_CV_R2 = mean(CV_R2, na.rm = TRUE),
    Mean_CV_MAE = mean(CV_MAE, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(
    desc(Frequency),
    Best_degree,
    Mean_CV_RMSE,
    desc(Mean_CV_R2),
    Best_C,
    Best_scale
  ) %>%
  slice(1)

# ------------------------------------------------------------
# 15. Plots
# ------------------------------------------------------------

p_best_degree <- ggplot(
  screening_df %>% filter(Model_status == "OK"),
  aes(
    x = factor(Best_degree)
  )
) +
  geom_bar() +
  facet_wrap(~ Stability_group) +
  labs(
    title = paste("Selected polynomial degree frequency -", group_label),
    x = "Selected degree",
    y = "Frequency"
  ) +
  theme_bw()

print(p_best_degree)

best_degree_file <- paste0(
  "SVR_poly_selected_degree_frequency_",
  tolower(group_label),
  ".png"
)

ggsave(
  filename = best_degree_file,
  plot = p_best_degree,
  width = 10,
  height = 6,
  dpi = 300
)

p_cv_r2 <- ggplot(
  screening_df %>% filter(Model_status == "OK"),
  aes(
    x = Stability_group,
    y = CV_R2
  )
) +
  geom_boxplot() +
  labs(
    title = paste("SVR polynomial CV R2 by stability group -", group_label),
    x = "Stability group",
    y = "CV R2"
  ) +
  theme_bw()

print(p_cv_r2)

cv_r2_file <- paste0(
  "SVR_poly_CV_R2_screening_",
  tolower(group_label),
  ".png"
)

ggsave(
  filename = cv_r2_file,
  plot = p_cv_r2,
  width = 10,
  height = 6,
  dpi = 300
)

p_cv_rmse <- ggplot(
  screening_df %>% filter(Model_status == "OK"),
  aes(
    x = Stability_group,
    y = CV_RMSE
  )
) +
  geom_boxplot() +
  labs(
    title = paste("SVR polynomial CV RMSE by stability group -", group_label),
    x = "Stability group",
    y = "CV RMSE"
  ) +
  theme_bw()

print(p_cv_rmse)

cv_rmse_file <- paste0(
  "SVR_poly_CV_RMSE_screening_",
  tolower(group_label),
  ".png"
)

ggsave(
  filename = cv_rmse_file,
  plot = p_cv_rmse,
  width = 10,
  height = 6,
  dpi = 300
)

# ------------------------------------------------------------
# 16. Export screening results to Excel
# ------------------------------------------------------------

screening_df_export <- screening_df %>%
  select(-any_of("Model_status"))

output_excel_file <- paste0(
  "SVR_poly_hyperparameter_screening_",
  tolower(group_label),
  ".xlsx"
)

wb <- createWorkbook()

addWorksheet(wb, "Screening_by_seed")
writeData(wb, "Screening_by_seed", screening_df_export)

addWorksheet(wb, "Screening_summary")
writeData(wb, "Screening_summary", screening_summary)

addWorksheet(wb, "Hyperparameter_frequency")
writeData(wb, "Hyperparameter_frequency", hyperparameter_frequency)

addWorksheet(wb, "Selected_by_group")
writeData(wb, "Selected_by_group", selected_hyperparameters_by_group)

addWorksheet(wb, "Overall_selected")
writeData(wb, "Overall_selected", overall_selected_hyperparameters)

for (sheet in names(wb)) {
  setColWidths(wb, sheet = sheet, cols = 1:100, widths = "auto")
}

saveWorkbook(wb, output_excel_file, overwrite = TRUE)

# ------------------------------------------------------------
# 17. Final output
# ------------------------------------------------------------

cat("\n============================================================\n")
cat("SVR polynomial hyperparameter screening completed for:", group_label, "\n")
cat("============================================================\n\n")

cat("Excel file saved as:", output_excel_file, "\n")
cat("Selected degree plot saved as:", best_degree_file, "\n")
cat("CV R2 plot saved as:", cv_r2_file, "\n")
cat("CV RMSE plot saved as:", cv_rmse_file, "\n\n")

cat("Screening summary:\n")
print(screening_summary, n = Inf, width = Inf)

cat("\nSelected hyperparameters by group:\n")
print(selected_hyperparameters_by_group, n = Inf, width = Inf)

cat("\nOverall selected hyperparameters:\n")
print(overall_selected_hyperparameters, n = Inf, width = Inf)