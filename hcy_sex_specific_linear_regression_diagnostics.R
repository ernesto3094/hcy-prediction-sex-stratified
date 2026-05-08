# ============================================================
# Linear regression analysis and diagnostic validation
# Sex-specific HCY prediction dataset: Men or Women
# ============================================================

# ------------------------------------------------------------
# 0. Install and load required libraries
# ------------------------------------------------------------
packages <- c(
  "readr", "car", "ggfortify", "MASS", "lmtest",
  "lawstat", "gvlma", "caret", "nortest"
)

missing_packages <- packages[!(packages %in% installed.packages()[, "Package"])]

if (length(missing_packages)) {
  install.packages(missing_packages)
}

lapply(packages, library, character.only = TRUE)

# ------------------------------------------------------------
# 1. Load and prepare the dataset
# ------------------------------------------------------------

# Define the study group according to the dataset being analyzed.
# Use "Men" when analyzing the male dataset.
# Use "Women" when analyzing the female dataset.
group_label <- "Men"

# Load the dataset.
# Change this path depending on the dataset being used.
# Example for men:   "~/dataset_genero_2_sin_HcyABN3.csv"
# Example for women: "~/dataset_genero_1_sin_HcyABN3.csv"
# Example if using transformed data:
# Men:   "~/transformed_data_men.csv"
# Women: "~/transformed_data_women.csv"
input_dataset_path <- "~/dataset_genero_2_sin_HcyABN3.csv"

df <- read_csv(input_dataset_path)

# Remove rows with missing values
df <- df[complete.cases(df), ]

# Exclude variables that should not be included in the regression analysis.
# These include epigenetic variables and auxiliary categorical variables.
vars_out <- c(
  "ALU", "LINE", "SAT", "HcyABN3",
  "GrasaCatMasc", "GrasaCatFem"
)

present_vars <- intersect(vars_out, names(df))

if (length(present_vars)) {
  df[present_vars] <- NULL
}

# Create the initial formula using all predictors except HCY
predictors <- setdiff(names(df), "HCY")
formula_initial <- as.formula(
  paste("HCY ~", paste(predictors, collapse = " + "))
)

# Fit the initial model to detect aliased variables
initial_model <- lm(formula_initial, data = df)

alias_matrix <- alias(initial_model)$Complete

# Detect variables with perfect collinearity
if (!is.null(alias_matrix)) {
  alias_vars <- rownames(alias_matrix)
  cat("Variables removed due to perfect collinearity:\n")
  print(alias_vars)
} else {
  cat("No variables with perfect collinearity were detected.\n")
  alias_vars <- character(0)
}

# Create a new formula excluding aliased variables
clean_predictors <- setdiff(predictors, alias_vars)

formula_clean <- as.formula(
  paste("HCY ~", paste(clean_predictors, collapse = " + "))
)

# ------------------------------------------------------------
# 2. Ten-fold cross-validation
# ------------------------------------------------------------
set.seed(123)

control <- trainControl(
  method = "cv",
  number = 10
)

cv_model <- train(
  formula_clean,
  data = df,
  method = "lm",
  trControl = control
)

# Display cross-validation results
cat("\nCross-validation results for the", group_label, "dataset:\n")
print(cv_model)

# Extract the final linear regression model
final_model <- cv_model$finalModel

cat("\nFinal linear regression model summary for the", group_label, "dataset:\n")
summary(final_model)

# ------------------------------------------------------------
# 3. Linear model assumptions
# ------------------------------------------------------------

# ------------------------------------------------------------
# 3.1 Linearity assessment
# ------------------------------------------------------------
cat("\nLinearity assessment:\n")

# The mean of residuals should be close to zero
cat("Mean of residuals:", mean(final_model$residuals), "\n")

# Component + residual plots
crPlots(final_model)

# Residuals vs fitted values
plot(final_model, 1)

# ggfortify diagnostic plot
autoplot(final_model, 1)

# ------------------------------------------------------------
# 3.2 Normality of residuals
# ------------------------------------------------------------
cat("\nNormality assessment of residuals:\n")

residuals_model <- final_model$residuals

# Lilliefors normality test
normality_test <- lillie.test(residuals_model)
print(normality_test)

# Histogram with fitted normal density curve
mean_residuals <- mean(residuals_model)
sd_residuals <- sd(residuals_model)

# Extend the x-axis range to improve visualization
x_limits <- extendrange(range(residuals_model), f = 0.25)

# Save current plot margins
old_par <- par(mar = c(4.8, 4.8, 3, 1) + 0.1)

# Create a histogram as density
histogram_object <- hist(
  residuals_model,
  freq = FALSE,
  main = paste("Residuals distribution -", group_label, "dataset"),
  xlab = "Residuals",
  col = "gray80",
  border = "white",
  xlim = x_limits
)

# Create the fitted normal curve
x_fit <- seq(x_limits[1], x_limits[2], length.out = 400)

y_fit <- dnorm(
  x_fit,
  mean = mean_residuals,
  sd = sd_residuals
)

# Define y-axis limits to include both histogram and normal curve
y_limits <- c(
  0,
  1.05 * max(histogram_object$density, y_fit, na.rm = TRUE)
)

# Plot the histogram and fitted normal curve
plot(
  0,
  0,
  type = "n",
  xlab = "Residuals",
  ylab = "Density",
  main = paste("Residuals distribution -", group_label, "dataset"),
  xlim = x_limits,
  ylim = y_limits
)

hist(
  residuals_model,
  freq = FALSE,
  col = "gray80",
  border = "white",
  breaks = histogram_object$breaks,
  add = TRUE
)

lines(
  x_fit,
  y_fit,
  col = "blue",
  lwd = 3
)

# Restore previous plot margins
par(old_par)

# Q-Q plot of residuals
qqnorm(
  residuals_model,
  main = paste("Q-Q plot of residuals -", group_label, "dataset")
)

qqline(
  residuals_model,
  col = 2
)

# ------------------------------------------------------------
# 3.3 Homoscedasticity assessment
# ------------------------------------------------------------
cat("\nHomoscedasticity assessment:\n")

# Non-constant variance test
ncvTest(final_model)

# Breusch-Pagan test
bptest(final_model)

# Scale-location plot
plot(
  final_model,
  3,
  main = paste("Scale-location plot -", group_label, "dataset"),
  ylab = expression(sqrt("|Standardized residuals|"))
)

# ggfortify scale-location plot
autoplot(final_model, 3)

# ------------------------------------------------------------
# 3.4 Independence of residuals
# ------------------------------------------------------------
cat("\nIndependence assessment of residuals:\n")

# Durbin-Watson test
durbinWatsonTest(final_model)

# Runs test
runs.test(final_model$residuals)

# Residuals over observation order
plot(
  final_model$residuals,
  type = "l",
  main = paste("Residuals over observation order -", group_label, "dataset"),
  xlab = "Observation",
  ylab = "Residuals"
)

# Autocorrelation function of residuals
acf(
  final_model$residuals,
  main = paste("ACF of residuals -", group_label, "dataset")
)

# ------------------------------------------------------------
# 3.5 Global validation of linear model assumptions
# ------------------------------------------------------------

# Condition number of the model matrix
model_matrix <- model.matrix(final_model)

cat(
  "Condition number:",
  kappa(model_matrix),
  "\n"
)

cat("\nGlobal validation using gvlma:\n")

gvlma_model <- gvlma(final_model)

summary(gvlma_model)

# Close open graphical device only if needed
# dev.off()

plot(
  gvlma_model,
  main = paste("Global validation of linear model assumptions -", group_label, "dataset")
)

# ------------------------------------------------------------
# 3.6 General diagnostic plots of the final model
# ------------------------------------------------------------
cat("\nGeneral diagnostic plots of the final model:\n")

plot(final_model)

# ------------------------------------------------------------
# 4. Multicollinearity analysis
# ------------------------------------------------------------
cat("\nMulticollinearity analysis using VIF:\n")

vif_values <- vif(final_model)

# Display all VIF values
print(vif_values)

# Identify variables exceeding the selected VIF threshold
vif_threshold <- 5

critical_vif <- vif_values[vif_values > vif_threshold]

if (length(critical_vif) > 0) {
  cat(
    "\nVariables with potentially concerning multicollinearity",
    paste0("(VIF > ", vif_threshold, "):\n")
  )
  print(critical_vif)
} else {
  cat(
    "\nNo relevant multicollinearity was detected",
    paste0("(all VIF values <= ", vif_threshold, ").\n")
  )
}

# ------------------------------------------------------------
# 5. Variable selection using stepAIC
# ------------------------------------------------------------
cat("\nVariable selection using stepAIC:\n")

manual_model <- lm(formula_clean, data = df)

step_model <- stepAIC(
  manual_model,
  direction = "both",
  trace = FALSE
)

# Display the summary of the selected model
cat("\nSummary of the model selected by stepAIC for the", group_label, "dataset:\n")
summary(step_model)

# Display the variables selected by stepAIC
cat("\nVariables selected by stepAIC:\n")
print(names(coef(step_model)))

# Create a new formula using only the variables selected by stepAIC
selected_variables <- names(coef(step_model))[-1]

formula_step <- as.formula(
  paste("HCY ~", paste(selected_variables, collapse = " + "))
)

# Fit the selected model using ten-fold cross-validation
set.seed(123)

control <- trainControl(
  method = "cv",
  number = 10
)

cv_step_model <- train(
  formula_step,
  data = df,
  method = "lm",
  trControl = control
)

# Display final cross-validation results after stepAIC selection
cat("\nCross-validation results after stepAIC selection for the", group_label, "dataset:\n")
print(cv_step_model)

# Extract the final model after stepAIC selection
final_step_model <- cv_step_model$finalModel

cat("\nFinal model summary after stepAIC selection for the", group_label, "dataset:\n")
summary(final_step_model)

