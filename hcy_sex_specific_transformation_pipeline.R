# ============================================================
# Data transformation pipeline for HCY prediction
# Group-specific dataset: Men or Women
# ============================================================

# Load required libraries
library(readxl)
library(e1071)
library(car)
library(MASS)
library(ggplot2)
library(gridExtra)
library(readr)

# ------------------------------------------------------------
# Define the study group according to the dataset being used
# Change this value to "Women" when using the female dataset
# ------------------------------------------------------------
group_label <- "Men"

# ------------------------------------------------------------
# Load the dataset
# Replace this path if the dataset is stored in another location
# Example for men:   "~/dataset_genero_2_sin_HcyABN3.csv"
# Example for women: "~/dataset_genero_1_sin_HcyABN3.csv"
# ------------------------------------------------------------
input_dataset_path <- "~/dataset_genero_2_sin_HcyABN3.csv"

data <- read_csv(input_dataset_path)
summary(data)

# ------------------------------------------------------------
# Save the categorical variable before excluding it from the analysis
# Uncomment this line if GrasaVisCat needs to be added again later
# ------------------------------------------------------------
# visceral_fat_category <- data$GrasaVisCat

# ------------------------------------------------------------
# Select only numeric variables and remove GrasaVisCat if present
# ------------------------------------------------------------
numeric_data <- data[sapply(data, is.numeric)]

if ("GrasaVisCat" %in% colnames(numeric_data)) {
  numeric_data$GrasaVisCat <- NULL
}

# ------------------------------------------------------------
# Create lists to store the recommended transformations
# ------------------------------------------------------------
log_transformation <- c()
square_root_transformation <- c()
cube_root_transformation <- c()
inverse_exponential_transformation <- c()
power_2_transformation <- c()
power_3_transformation <- c()
yeo_johnson_transformation <- c()
no_transformation <- c()

# ------------------------------------------------------------
# Function to recommend transformations based on skewness and kurtosis
# ------------------------------------------------------------
recommend_transformations <- function(variable, variable_name) {
  
  skewness_value <- skewness(variable)
  kurtosis_value <- kurtosis(variable)
  minimum_value <- min(variable)
  
  cat("Variable:", variable_name, "\n")
  cat("Skewness:", skewness_value, "\n")
  cat("Kurtosis:", kurtosis_value, "\n")
  
  if (skewness_value > 1.5 && minimum_value > 0) {
    log_transformation <<- c(log_transformation, variable_name)
    
  } else if (skewness_value > 1) {
    square_root_transformation <<- c(square_root_transformation, variable_name)
    
  } else if (skewness_value > 0.5) {
    cube_root_transformation <<- c(cube_root_transformation, variable_name)
    
  } else if (skewness_value < -1.5) {
    inverse_exponential_transformation <<- c(inverse_exponential_transformation, variable_name)
    
  } else if (skewness_value < -1) {
    power_2_transformation <<- c(power_2_transformation, variable_name)
    
  } else if (skewness_value < -0.5) {
    power_3_transformation <<- c(power_3_transformation, variable_name)
    
  } else if (kurtosis_value > 3) {
    yeo_johnson_transformation <<- c(yeo_johnson_transformation, variable_name)
    
  } else {
    no_transformation <<- c(no_transformation, variable_name)
  }
  
  cat("-------------------------------\n")
}

# ------------------------------------------------------------
# Apply the recommendation function to all numeric columns
# ------------------------------------------------------------
for (variable_name in colnames(numeric_data)) {
  recommend_transformations(numeric_data[[variable_name]], variable_name)
}

# ------------------------------------------------------------
# Apply the selected transformations
# ------------------------------------------------------------
transformed_data <- numeric_data

for (variable in log_transformation) {
  transformed_data[[variable]] <- log(transformed_data[[variable]] + 1)
}

for (variable in square_root_transformation) {
  transformed_data[[variable]] <- sqrt(
    transformed_data[[variable]] + abs(min(transformed_data[[variable]])) + 1
  )
}

for (variable in cube_root_transformation) {
  transformed_data[[variable]] <- sign(transformed_data[[variable]]) *
    abs(transformed_data[[variable]])^(1 / 3)
}

for (variable in inverse_exponential_transformation) {
  transformed_data[[variable]] <- 1 / (
    transformed_data[[variable]] + abs(min(transformed_data[[variable]])) + 1
  )
}

for (variable in power_2_transformation) {
  transformed_data[[variable]] <- transformed_data[[variable]]^2
}

for (variable in power_3_transformation) {
  transformed_data[[variable]] <- transformed_data[[variable]]^3
}

for (variable in yeo_johnson_transformation) {
  power_transform_model <- powerTransform(
    transformed_data[[variable]] ~ 1,
    family = "yjPower"
  )
  
  transformed_data[[variable]] <- yjPower(
    transformed_data[[variable]],
    power_transform_model$lambda
  )
}

# ------------------------------------------------------------
# Add GrasaVisCat back at the end if needed
# Uncomment this line only if the categorical variable is required
# ------------------------------------------------------------
# transformed_data$GrasaVisCat <- visceral_fat_category

# ------------------------------------------------------------
# Function to plot histograms and QQ plots before and after transformation
# ------------------------------------------------------------
plot_histogram_qq_gaussian <- function(original, transformed, variable_name) {
  
  hist_before <- ggplot(data.frame(y = original), aes(x = y)) +
    geom_histogram(
      aes(y = ..density..),
      bins = 30,
      fill = "blue",
      alpha = 0.3,
      color = "black"
    ) +
    stat_function(
      fun = dnorm,
      args = list(
        mean = mean(original, na.rm = TRUE),
        sd = sd(original, na.rm = TRUE)
      ),
      color = "red",
      size = 1
    ) +
    ggtitle(paste("Histogram before transformation:", variable_name, "-", group_label)) +
    theme_minimal()
  
  hist_after <- ggplot(data.frame(y = transformed), aes(x = y)) +
    geom_histogram(
      aes(y = ..density..),
      bins = 30,
      fill = "red",
      alpha = 0.3,
      color = "black"
    ) +
    stat_function(
      fun = dnorm,
      args = list(
        mean = mean(transformed, na.rm = TRUE),
        sd = sd(transformed, na.rm = TRUE)
      ),
      color = "blue",
      size = 1
    ) +
    ggtitle(paste("Histogram after transformation:", variable_name, "-", group_label)) +
    theme_minimal()
  
  qqplot_before <- ggplot(data.frame(y = original), aes(sample = y)) +
    stat_qq() +
    stat_qq_line() +
    ggtitle(paste("QQ plot before transformation:", variable_name, "-", group_label)) +
    theme_minimal()
  
  qqplot_after <- ggplot(data.frame(y = transformed), aes(sample = y)) +
    stat_qq() +
    stat_qq_line() +
    ggtitle(paste("QQ plot after transformation:", variable_name, "-", group_label)) +
    theme_minimal()
  
  grid.arrange(
    hist_before,
    hist_after,
    qqplot_before,
    qqplot_after,
    ncol = 2
  )
}

# ------------------------------------------------------------
# Plot only the variables that were transformed
# ------------------------------------------------------------
all_transformed_variables <- unique(c(
  log_transformation,
  square_root_transformation,
  cube_root_transformation,
  inverse_exponential_transformation,
  power_2_transformation,
  power_3_transformation,
  yeo_johnson_transformation
))

for (variable_name in all_transformed_variables) {
  original_variable <- numeric_data[[variable_name]]
  transformed_variable <- transformed_data[[variable_name]]
  
  plot_histogram_qq_gaussian(
    original_variable,
    transformed_variable,
    variable_name
  )
}

# ------------------------------------------------------------
# Save the transformed dataset
# Change the output file name depending on the group analyzed
# Example for men:   "transformed_data_men.csv"
# Example for women: "transformed_data_women.csv"
# ------------------------------------------------------------
output_csv_path <- "transformed_data_men.csv"

write.csv(
  transformed_data,
  output_csv_path,
  row.names = FALSE
)

# ------------------------------------------------------------
# Summary of applied transformations
# ------------------------------------------------------------
cat("Summary of Applied Transformations -", group_label, "Dataset\n")
cat("Log transformation:", paste(log_transformation, collapse = ", "), "\n")
cat("Square root transformation:", paste(square_root_transformation, collapse = ", "), "\n")
cat("Cube root transformation:", paste(cube_root_transformation, collapse = ", "), "\n")
cat("Inverse exponential transformation:", paste(inverse_exponential_transformation, collapse = ", "), "\n")
cat("Power transformation squared:", paste(power_2_transformation, collapse = ", "), "\n")
cat("Power transformation cubed:", paste(power_3_transformation, collapse = ", "), "\n")
cat("Yeo-Johnson transformation:", paste(yeo_johnson_transformation, collapse = ", "), "\n")
cat("No transformation:", paste(no_transformation, collapse = ", "), "\n")
cat("Transformations completed successfully.\n")
cat("Transformed dataset saved as:", output_csv_path, "\n")
