# ============================================================
# Polynomial Regression Degree Screening
# Male subgroup
#
# Purpose:
#   - Test polynomial degrees from 2 to 9
#   - Use 10 random seeds
#   - Use stability-based accumulated groups: Stable_100 to Stable_60
#   - Evaluate performance using 10-fold CV within the training set
#   - Select the simplest useful polynomial degree
#   - Export screening results to Excel
#   - Generate CV R2 and CV RMSE plots
#
# Note:
#   This is only the tuning/screening phase.
#   Learning curves and final train/test metrics will be computed later
#   using the selected polynomial degree.
# ============================================================

# ------------------------------------------------------------
# 1. Load packages
# ------------------------------------------------------------

required_packages <- c(
  "caret",
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

polynomial_degrees <- 2:9

# Minimum R2 difference allowed from the best degree.
# The selected degree will be the simplest degree within 0.01 of the best CV R2.
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
# 7. Polynomial formula builder
# ------------------------------------------------------------
# raw = FALSE uses orthogonal polynomials, which are more stable
# for higher degrees than direct powers.

create_pr_formula <- function(response_variable, predictors, degree) {
  
  polynomial_terms <- paste0(
    "poly(", predictors, ", ", degree, ", raw = FALSE)",
    collapse = " + "
  )
  
  as.formula(
    paste(response_variable, "~", polynomial_terms)
  )
}

# ------------------------------------------------------------
# 8. Run polynomial degree screening
# ------------------------------------------------------------

degree_screening_results <- list()
counter <- 1

for (current_seed in seeds) {
  
  cat("\n============================================================\n")
  cat("Polynomial degree screening - seed:", current_seed, "\n")
  cat("============================================================\n")
  
  split_data <- create_hcy_stratified_split(
    data = data,
    response_variable = response_variable,
    seed = current_seed
  )
  
  train_data <- split_data$train_data
  
  for (group_name in names(stable_groups)) {
    
    predictors <- stable_groups[[group_name]]
    
    for (degree_value in polynomial_degrees) {
      
      cat(
        "Seed:", current_seed,
        "| Group:", group_name,
        "| Degree:", degree_value, "\n"
      )
      
      set.seed(current_seed)
      
      train_control <- trainControl(
        method = "cv",
        number = 10
      )
      
      model_formula <- create_pr_formula(
        response_variable = response_variable,
        predictors = predictors,
        degree = degree_value
      )
      
      fitted_model <- tryCatch(
        {
          train(
            model_formula,
            data = train_data,
            method = "lm",
            trControl = train_control,
            preProcess = c("center", "scale"),
            metric = "Rsquared"
          )
        },
        error = function(e) {
          message("Training error: ", e$message)
          return(NULL)
        }
      )
      
      if (is.null(fitted_model)) {
        
        degree_screening_results[[counter]] <- data.frame(
          Group_label = group_label,
          Seed = current_seed,
          Stability_group = group_name,
          Polynomial_degree = degree_value,
          Num_predictors = length(predictors),
          CV_R2 = NA_real_,
          CV_RMSE = NA_real_,
          CV_MAE = NA_real_
        )
        
      } else {
        
        cv_result <- fitted_model$results[1, ]
        
        degree_screening_results[[counter]] <- data.frame(
          Group_label = group_label,
          Seed = current_seed,
          Stability_group = group_name,
          Polynomial_degree = degree_value,
          Num_predictors = length(predictors),
          CV_R2 = cv_result$Rsquared,
          CV_RMSE = cv_result$RMSE,
          CV_MAE = cv_result$MAE
        )
      }
      
      counter <- counter + 1
    }
  }
}

degree_screening_df <- bind_rows(degree_screening_results)

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
# 10. Summarize degree screening
# ------------------------------------------------------------

degree_screening_summary <- degree_screening_df %>%
  group_by(
    Group_label,
    Stability_group,
    Polynomial_degree,
    Num_predictors
  ) %>%
  summarise(
    Runs = sum(!is.na(CV_R2)),
    
    Mean_CV_R2 = mean(CV_R2, na.rm = TRUE),
    SD_CV_R2 = sd(CV_R2, na.rm = TRUE),
    CI95_CV_R2 = calculate_ci95(CV_R2),
    
    Mean_CV_RMSE = mean(CV_RMSE, na.rm = TRUE),
    SD_CV_RMSE = sd(CV_RMSE, na.rm = TRUE),
    CI95_CV_RMSE = calculate_ci95(CV_RMSE),
    
    Mean_CV_MAE = mean(CV_MAE, na.rm = TRUE),
    SD_CV_MAE = sd(CV_MAE, na.rm = TRUE),
    CI95_CV_MAE = calculate_ci95(CV_MAE),
    
    .groups = "drop"
  ) %>%
  arrange(Stability_group, Polynomial_degree)

# ------------------------------------------------------------
# 11. Select the simplest useful degree by group
# ------------------------------------------------------------

selected_degree_by_group <- degree_screening_summary %>%
  group_by(Group_label, Stability_group) %>%
  mutate(
    Best_CV_R2_in_group = max(Mean_CV_R2, na.rm = TRUE),
    R2_difference_from_best = Best_CV_R2_in_group - Mean_CV_R2
  ) %>%
  filter(R2_difference_from_best <= minimum_r2_difference) %>%
  arrange(Polynomial_degree) %>%
  slice(1) %>%
  ungroup() %>%
  select(
    Group_label,
    Stability_group,
    Selected_degree = Polynomial_degree,
    Num_predictors,
    Mean_CV_R2,
    SD_CV_R2,
    CI95_CV_R2,
    Mean_CV_RMSE,
    SD_CV_RMSE,
    CI95_CV_RMSE,
    Mean_CV_MAE,
    SD_CV_MAE,
    CI95_CV_MAE,
    Best_CV_R2_in_group,
    R2_difference_from_best
  )

# ------------------------------------------------------------
# 12. Select one overall degree for the subgroup
# ------------------------------------------------------------

overall_selected_degree <- selected_degree_by_group %>%
  count(Selected_degree, name = "Frequency") %>%
  arrange(desc(Frequency), Selected_degree) %>%
  slice(1)

# ------------------------------------------------------------
# 13. Screening plots
# ------------------------------------------------------------

p_degree_r2 <- ggplot(
  degree_screening_summary,
  aes(
    x = Polynomial_degree,
    y = Mean_CV_R2,
    group = Stability_group,
    linetype = Stability_group
  )
) +
  geom_line() +
  geom_point() +
  labs(
    title = paste("Polynomial degree screening based on CV R2 -", group_label),
    x = "Polynomial degree",
    y = "Mean CV R2"
  ) +
  theme_bw()

print(p_degree_r2)

degree_r2_file <- paste0(
  "PR_degree_screening_CV_R2_",
  tolower(group_label),
  ".png"
)

ggsave(
  filename = degree_r2_file,
  plot = p_degree_r2,
  width = 10,
  height = 6,
  dpi = 300
)

p_degree_rmse <- ggplot(
  degree_screening_summary,
  aes(
    x = Polynomial_degree,
    y = Mean_CV_RMSE,
    group = Stability_group,
    linetype = Stability_group
  )
) +
  geom_line() +
  geom_point() +
  labs(
    title = paste("Polynomial degree screening based on CV RMSE -", group_label),
    x = "Polynomial degree",
    y = "Mean CV RMSE"
  ) +
  theme_bw()

print(p_degree_rmse)

degree_rmse_file <- paste0(
  "PR_degree_screening_CV_RMSE_",
  tolower(group_label),
  ".png"
)

ggsave(
  filename = degree_rmse_file,
  plot = p_degree_rmse,
  width = 10,
  height = 6,
  dpi = 300
)

# ------------------------------------------------------------
# 14. Export screening results to Excel
# ------------------------------------------------------------

output_excel_file <- paste0(
  "PR_degree_screening_",
  tolower(group_label),
  ".xlsx"
)

wb <- createWorkbook()

addWorksheet(wb, "Degree_screening_by_seed")
writeData(wb, "Degree_screening_by_seed", degree_screening_df)

addWorksheet(wb, "Degree_screening_summary")
writeData(wb, "Degree_screening_summary", degree_screening_summary)

addWorksheet(wb, "Selected_degree_by_group")
writeData(wb, "Selected_degree_by_group", selected_degree_by_group)

addWorksheet(wb, "Overall_selected_degree")
writeData(wb, "Overall_selected_degree", overall_selected_degree)

for (sheet in names(wb)) {
  setColWidths(wb, sheet = sheet, cols = 1:100, widths = "auto")
}

saveWorkbook(wb, output_excel_file, overwrite = TRUE)

# ------------------------------------------------------------
# 15. Final output
# ------------------------------------------------------------

cat("\n============================================================\n")
cat("Polynomial degree screening completed for:", group_label, "\n")
cat("============================================================\n\n")

cat("Excel file saved as:", output_excel_file, "\n")
cat("CV R2 plot saved as:", degree_r2_file, "\n")
cat("CV RMSE plot saved as:", degree_rmse_file, "\n\n")

cat("Degree screening summary:\n")
print(degree_screening_summary, n = Inf, width = Inf)

cat("\nSelected degree by group:\n")
print(selected_degree_by_group, n = Inf, width = Inf)

cat("\nOverall selected degree:\n")
print(overall_selected_degree, n = Inf, width = Inf)