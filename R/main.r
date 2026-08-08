################################################################################
# CREDIT CARD DEFAULT PREDICTION
################################################################################
#
# This script contains the complete workflow used for predicting credit card
# default. It covers the entire process from loading and exploring the data
# to training, tuning and evaluating multiple machine learning models.
#
# Workflow:
# - Load and inspect the data
# - Perform exploratory data analysis
# - Preprocess the data
# - Create additional features
# - Split the data into training, validation and holdout sets
# - Train and tune different classification models
# - Compare model performance using multiple evaluation metrics
# - Optimize classification thresholds based on economic costs
# - Evaluate the final models on unseen data
#
# Models included:
# - Logistic Regression
# - Lasso Regression
# - Ridge Regression
# - Decision Tree
# - Random Forest
# - Gradient Boosting
#
# Since the dataset is imbalanced, class weights are incorporated during model
# training. Instead of relying only on traditional metrics such as accuracy or
# ROC AUC, the models are also evaluated from a business perspective by assigning
# different costs to false positives and false negatives.
#
# A fixed random seed is used throughout the analysis to ensure reproducibility.
#
################################################################################


################################################################################
# SECTION 1: SETUP AND PACKAGE INSTALLATION (5% of workflow)
################################################################################

# Load required libraries
library(MASS)          # For example datasets
library(ISLR)          # For example datasets
library(caret)         # For machine learning framework
library(glmnet)        # For regularized regression (Lasso/Ridge)
library(randomForest)  # For random forest models
library(gbm)           # For gradient boosting
library(e1071)         # For SVM
library(nnet)          # For neural networks
library(corrplot)      # For correlation plots
library(ggplot2)       # For advanced visualizations
library(gridExtra)     # For arranging multiple plots
library(pROC)          # For ROC curves and AUC
library(rpart.plot)    # For decision tree visualization
library(readxl)
library(tidyverse)
library(ranger)        # For ranger Random Forest model


# Set random seed for reproducibility (IMPORTANT!)
# This ensures your results can be replicated
set.seed(123)

################################################################################
# SECTION 2: DATA LOADING AND INITIAL INSPECTION  
################################################################################

# This research employed a binary variable, default payment (Yes = 1, No = 0), as the response variable. This study reviewed the literature and used the following 23 variables as explanatory variables:
# X1: Amount of the given credit (NT dollar): it includes both the individual consumer credit and his/her family (supplementary) credit.
#X2: Gender (1 = male; 2 = female).
#X3: Education (1 = graduate school; 2 = university; 3 = high school; 4 = others).
#X4: Marital status (1 = married; 2 = single; 3 = others).
#X5: Age (year).
#X6 - X11: History of past payment. We tracked the past monthly payment records (from April to September, 2005) as follows: X6 = the repayment status in September, 2005; X7 = the repayment status in August, 2005; . . .;X11 = the repayment status in April, 2005. The measurement scale for the repayment status is: -1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above.
#X12-X17: Amount of bill statement (NT dollar). X12 = amount of bill statement in September, 2005; X13 = amount of bill statement in August, 2005; . . .; X17 = amount of bill statement in April, 2005. 
#X18-X23: Amount of previous payment (NT dollar). X18 = amount paid in September, 2005; X19 = amount paid in August, 2005; . . .;X23 = amount paid in April, 2005.

list.files()

# Loading data
data <- read_excel("data/data_creditcard.xls")

data <- data[-1, ]

# Initial data inspection
cat("=== DATA STRUCTURE ===\n")
str(data)  # Shows variable types and first few values

# make all integers
data <- data %>% mutate(across(everything(), as.numeric))

glimpse(data)

cat("\n=== DATA DIMENSIONS ===\n")
cat("Number of observations:", nrow(data), "\n")
cat("Number of variables:", ncol(data), "\n")

cat("\n=== FIRST FEW ROWS ===\n")
head(data, 10)  # Show first 10 rows

cat("\n=== LAST FEW ROWS ===\n")
tail(data, 10)  # Check last few rows for any issues

# Check for duplicate rows
cat("\n=== DUPLICATE CHECK ===\n")
cat("Number of duplicate rows:", sum(duplicated(data)), "\n")

# Select relevant columns (keep data in RAW format for now)
# We will convert to proper types in Section 4 (Preprocessing)
data <- data %>% 
  select(-...1)

# Renaming everything so it makes more sense

data <- data %>%
  rename(
    LIMIT_BAL     = X1,
    SEX           = X2,
    EDUCATION     = X3,
    MARRIAGE      = X4,
    AGE           = X5,
    PAY_SEPT      = X6,
    PAY_AUG       = X7,
    PAY_JUL       = X8,
    PAY_JUN       = X9,
    PAY_MAY       = X10,
    PAY_APR       = X11,
    BILL_SEPT     = X12,
    BILL_AUG      = X13,
    BILL_JUL      = X14,
    BILL_JUN      = X15,
    BILL_MAY      = X16,
    BILL_APR      = X17,
    PAY_SEPT_AMT  = X18,
    PAY_AUG_AMT   = X19,
    PAY_JUL_AMT   = X20,
    PAY_JUN_AMT   = X21,
    PAY_MAY_AMT   = X22,
    PAY_APR_AMT   = X23,
    DEFAULT       = Y       # Our Target Variable
  )


################################################################################
# SECTION 3: EXPLORATORY DATA ANALYSIS   
################################################################################

cat("\n\n########## EXPLORATORY DATA ANALYSIS ##########\n\n")

# 3.1: Summary Statistics
cat("=== SUMMARY STATISTICS ===\n")
summary(data)

# 3.2: Missing Values Analysis
cat("\n=== MISSING VALUES ANALYSIS ===\n")
missing_count <- colSums(is.na(data))
missing_percent <- round(100 * missing_count / nrow(data), 2)
missing_df <- data.frame(
  Variable = names(missing_count),
  Missing_Count = missing_count,
  Missing_Percent = missing_percent
)
print(missing_df)

# Visualize missing values (if any)
if(sum(missing_count) > 0) {
  barplot(missing_percent, 
          main = "Percentage of Missing Values by Variable",
          ylab = "Percentage Missing (%)",
          las = 2,
          col = "coral")
}

# 3.3: Target Variable Analysis (MOST IMPORTANT!)
cat("\n=== TARGET VARIABLE ANALYSIS ===\n")
target_var <- "DEFAULT"


# Frequency table
cat("\nClass Distribution:\n")
table(data[[target_var]])
cat("\nClass Proportions:\n")
prop.table(table(data[[target_var]]))

# Visualize class distribution
par(mfrow = c(1, 2))
barplot(table(data[[target_var]]), 
        main = "Target Variable Distribution",
        xlab = target_var,
        ylab = "Frequency",
        col = c("lightblue", "coral"))

pie(table(data[[target_var]]), 
    main = "Target Variable Proportion",
    col = c("lightblue", "coral"))


# 3.4: Numeric Variables Analysis
cat("\n=== NUMERIC VARIABLES ANALYSIS ===\n")
numeric_vars <- names(data)[sapply(data, is.numeric)]
cat("Numeric variables found:", paste(numeric_vars, collapse = ", "), "\n")

# Distribution plots for numeric variables
par(mfrow = c(3, 3))
for(var in numeric_vars) {
  hist(data[[var]], 
       main = paste("Distribution of", var),
       xlab = var,
       col = "lightblue",
       breaks = 5)
}

# Boxplots to identify outliers # da war length(numeric_vars) drin
par(mfrow = c(3, 3))
for(var in numeric_vars) {
  boxplot(data[[var]], 
          main = var,
          col = "lightgreen")
}

# Outlier irrelevant für uns, da extrema Fälle enorm wichtig in Financial data

# Detailed statistics for numeric variables
cat("\nDetailed Statistics for Numeric Variables:\n")
for(var in numeric_vars) {
  cat(sprintf("\n--- %s ---\n", var))
  cat(sprintf("Mean: %.2f\n", mean(data[[var]], na.rm = TRUE)))
  cat(sprintf("Median: %.2f\n", median(data[[var]], na.rm = TRUE)))
  cat(sprintf("SD: %.2f\n", sd(data[[var]], na.rm = TRUE)))
  cat(sprintf("Min: %.2f\n", min(data[[var]], na.rm = TRUE)))
  cat(sprintf("Max: %.2f\n", max(data[[var]], na.rm = TRUE)))
  cat(sprintf("IQR: %.2f\n", IQR(data[[var]], na.rm = TRUE)))
  
  # Check for outliers using IQR method
  Q1 <- quantile(data[[var]], 0.25, na.rm = TRUE)
  Q3 <- quantile(data[[var]], 0.75, na.rm = TRUE)
  IQR_val <- Q3 - Q1
  outliers <- sum(data[[var]] < (Q1 - 1.5 * IQR_val) | 
                    data[[var]] > (Q3 + 1.5 * IQR_val), na.rm = TRUE)
  cat(sprintf("Number of outliers (IQR method): %d (%.2f%%)\n", 
              outliers, 100 * outliers / nrow(data)))
}

summary(data)

# PAY variables stay numeric:
# No factor(), no ordered = TRUE
# Reason: We would lose important info, since the distances are important,

glimpse(data)

# 3.5: Categorical Variables Analysis ### nachher wiederholen nach anpassung
cat("\n=== CATEGORICAL VARIABLES ANALYSIS ===\n")
categorical_vars <- names(data)[sapply(data, function(x) is.factor(x) | is.character(x))]
cat("Categorical variables found:", paste(categorical_vars, collapse = ", "), "\n")

# Frequency tables and bar plots
par(mfrow = c(2, 2))
for(var in categorical_vars) {
  cat(sprintf("\nFrequency table for %s:\n", var))
  print(table(data[[var]]))
  cat(sprintf("Proportions for %s:\n", var))
  print(prop.table(table(data[[var]])))
  
  # Bar plot
  barplot(table(data[[var]]), 
          main = paste("Distribution of", var),
          col = rainbow(length(unique(data[[var]]))))
}

# 3.6: Relationship between Predictors and Target
cat("\n=== RELATIONSHIP BETWEEN PREDICTORS AND TARGET ===\n")

# For numeric predictors vs binary target
par(mfrow = c(3, 3))
for(var in numeric_vars) {
  boxplot(data[[var]] ~ data[[target_var]], 
          main = paste(var, "by", target_var),
          xlab = target_var,
          ylab = var,
          col = c("lightblue", "coral"))
}

# For categorical predictors vs binary target
for(var in categorical_vars[categorical_vars != target_var]) {
  cat(sprintf("\nCross-tabulation: %s vs %s\n", var, target_var))
  print(table(data[[var]], data[[target_var]]))
  cat("\nProportions:\n")
  print(prop.table(table(data[[var]], data[[target_var]]), margin = 1))
}

# 3.7: Correlation Analysis (for numeric variables only)
if(length(numeric_vars) > 1) {
  cat("\n=== CORRELATION ANALYSIS ===\n")
  
  # Calculate correlation matrix
  cor_matrix <- cor(data[, numeric_vars], use = "complete.obs")
  print(round(cor_matrix, 3))
  
  # Visualize correlation matrix
  par(mfrow = c(1, 1))
  corrplot(cor_matrix, 
           method = "color",
           type = "upper",
           number.cex = 0.6,
           addCoef.col = "black",
           tl.col = "black",
           tl.srt = 45,
           title = "Correlation Matrix",
           mar = c(0, 0, 2, 0))
  
  # Identify highly correlated pairs
  cat("\nHighly correlated variable pairs (|r| > 0.7):\n")
  high_cor <- which(abs(cor_matrix) > 0.7 & abs(cor_matrix) < 1, arr.ind = TRUE)
  if(nrow(high_cor) > 0) {
    for(i in 1:nrow(high_cor)) {
      var1 <- rownames(cor_matrix)[high_cor[i, 1]]
      var2 <- colnames(cor_matrix)[high_cor[i, 2]]
      cor_val <- cor_matrix[high_cor[i, 1], high_cor[i, 2]]
      cat(sprintf("  %s <-> %s: %.3f\n", var1, var2, cor_val))
    }
    cat("\nNote: Consider removing one variable from highly correlated pairs\n")
  } else {
    cat("No highly correlated pairs found.\n")
  }
}

# 3.8: Pairwise Scatterplots (if not too many variables)
if(length(numeric_vars) <= 6 && length(numeric_vars) > 1) {
  cat("\n=== PAIRWISE RELATIONSHIPS ===\n")
  pairs(data[, numeric_vars], 
        main = "Scatterplot Matrix",
        col = as.numeric(data[[target_var]]) + 1,
        pch = 19)
}

cat("\n=== EDA COMPLETE ===\n")
cat("Review all plots and statistics before proceeding to modeling!\n\n")

glimpse(data)

cat("\n=== INVALID VALUE CHECK ===\n")

valid_list <- list(
  SEX = c(1, 2),
  EDUCATION = c(1, 2, 3, 4),
  MARRIAGE = c(1, 2, 3),
  DEFAULT = c(0, 1)
)

for (v in names(valid_list)) {
  invalid <- setdiff(unique(data[[v]]), valid_list[[v]])
  cat("\nVariable:", v, "Invalid values:", invalid, "\n")
}

# 0 zu "others = 3" bei Marriage und 0,5,6 bei Education zu "others = 4" machen

################################################################################
# SECTION 4: DATA PREPROCESSING    
################################################################################
# This section prepares data for machine learning models

cat("\n\n########## DATA PREPROCESSING ##########\n\n")

# 4.0: Data Type Conversions and Initial Cleaning
cat("=== DATA TYPE CONVERSIONS ===\n")
cat("Based on our EDA, we now convert variables to appropriate types for modeling\n\n")

# 1) Convert DEFAULT to factor with proper labels ------------------------------
# This is REQUIRED for classification models to work correctly
data$DEFAULT <- factor(data$DEFAULT, levels = c(0, 1), labels = c("No", "Yes"))

# quick check
unique(data$DEFAULT)

# 2) Convert SEX to factor with descriptive labels -----------------------------
data$SEX <- factor(data$SEX, levels = c(1, 2), labels = c("male", "female"))

# quick check
unique(data$SEX)
cat("Converted SEX to factor with labels: male, female\n")

# 3) Convert Education to factor -----------------------------------------------
data$EDUCATION <- ifelse(data$EDUCATION %in% c(0, 5, 6),
                         4,
                         data$EDUCATION)
# quick check
unique(data$EDUCATION)

# Convert to factor with correct labels
data$EDUCATION <- factor(data$EDUCATION,
                         levels = c(1, 2, 3, 4),
                         labels = c("Graduate School",
                                    "University",
                                    "High School",
                                    "Others"))

# quick check
unique(data$EDUCATION)

# 4) Convert MARRIAGE to factor ------------------------------------------------
data$MARRIAGE <- ifelse(data$MARRIAGE == 0,
                         3,
                         data$MARRIAGE)
# quick check
unique(data$MARRIAGE)

# Convert to factor with correct labels
data$MARRIAGE <- factor(data$MARRIAGE,
                         levels = c(1, 2, 3),
                         labels = c("married",
                                    "single",
                                    "others"))

# quick check
unique(data$MARRIAGE)

# Handle missing Embarked values

sum(is.na(data))

# no NA

cat("\nData type conversions complete!\n")
cat(sprintf("Final dataset dimensions: %d rows, %d columns\n\n", nrow(data), ncol(data)))

# 4.1: Handle Missing Values
cat("=== HANDLING MISSING VALUES ===\n")

# NACH den Konvertierungen in SECTION 4.0
numeric_vars <- names(data)[sapply(data, is.numeric)]
categorical_vars <- names(data)[sapply(data, function(x) is.factor(x) | is.character(x))]


# Check if there are any missing values
if(sum(is.na(data)) > 0) {
  cat("Missing values detected. Applying imputation...\n")
  
  # Option 1: Remove rows with missing values (if few)
  # data <- na.omit(data)
  # Not recommended here as Age has many missing values
  
  # Option 2: Impute numeric variables with median (grouped imputation is better!)
  # For Titanic, Age has missing values - let's impute by median within groups
  for(var in numeric_vars) {
    if(sum(is.na(data[[var]])) > 0) {
      # Simple median imputation
      median_val <- median(data[[var]], na.rm = TRUE)
      data[[var]][is.na(data[[var]])] <- median_val
      cat(sprintf("Imputed %s with median: %.2f\n", var, median_val))
      
      # ALTERNATIVE: Group-based imputation (better for Age)
      # For example, impute Age by median within Sex and Pclass groups
      # data <- data %>% 
      #   group_by(Sex, Pclass) %>%
      #   mutate(Age = ifelse(is.na(Age), median(Age, na.rm = TRUE), Age)) %>%
      #   ungroup()
    }
  }
  
  # Option 3: Impute categorical variables with mode
  for(var in categorical_vars) {
    if(sum(is.na(data[[var]])) > 0) {
      mode_val <- names(sort(table(data[[var]]), decreasing = TRUE))[1]
      data[[var]][is.na(data[[var]])] <- mode_val
      cat(sprintf("Imputed %s with mode: %s\n", var, mode_val))
    }
  }
} else {
  cat("No missing values found. Proceeding...\n")
}

# Verify no missing values remain
cat("Missing values after imputation:", sum(is.na(data)), "\n")

# 4.2: Handle Outliers (if necessary) => Not needed, outliers are very important info here

# check variables:

range(data$AGE)

# in the rest of the data, outliers carry important information and will thus not be cut

cat("\n=== OUTLIER HANDLING ===\n")
cat("  - Option 1: Winsorization (cap at certain percentiles)\n")
cat("  - Option 2: Transformation (log, sqrt)\n")
cat("  - Option 3: Remove extreme outliers (justify in report!)\n")

# Example: Winsorization (capping at 1st and 99th percentiles)
# for(var in numeric_vars) {
#   q01 <- quantile(data[[var]], 0.01, na.rm = TRUE)
#   q99 <- quantile(data[[var]], 0.99, na.rm = TRUE)
#   data[[var]][data[[var]] < q01] <- q01
#   data[[var]][data[[var]] > q99] <- q99
# }


# 4.3: Feature Engineering (if needed)
cat("\n=== FEATURE ENGINEERING ===\n")
cat("Creating new features from existing ones...\n")

## ============================================================
## FEATURE ENGINEERING – CREDIT CARD DEFAULT DATA    
## ============================================================

cat("\n=== FEATURE ENGINEERING FOR CREDIT DEFAULT DATA ===\n")

## 0) Definiere Variable-Gruppen (anpassen falls andere Namen)
delay_vars   <- c("PAY_SEPT", "PAY_AUG", "PAY_JUL",
                  "PAY_JUN", "PAY_MAY", "PAY_APR")

bill_vars    <- c("BILL_SEPT", "BILL_AUG", "BILL_JUL",
                  "BILL_JUN", "BILL_MAY", "BILL_APR")

pay_amt_vars <- c("PAY_SEPT_AMT", "PAY_AUG_AMT", "PAY_JUL_AMT",
                  "PAY_JUN_AMT", "PAY_MAY_AMT", "PAY_APR_AMT")

## Optional: Sanity-Check – existieren alle Spalten?
missing_delay   <- setdiff(delay_vars,   names(data))
missing_bill    <- setdiff(bill_vars,    names(data))
missing_pay_amt <- setdiff(pay_amt_vars, names(data))

if (length(missing_delay) > 0)   cat("WARN: Missing delay vars:",   missing_delay, "\n")
if (length(missing_bill) > 0)    cat("WARN: Missing bill vars:",    missing_bill, "\n")
if (length(missing_pay_amt) > 0) cat("WARN: Missing pay_amt vars:", missing_pay_amt, "\n")

## Sicherstellen, dass sie numerisch sind (sonst as.numeric())
data[delay_vars]   <- lapply(data[delay_vars],   as.numeric)
data[bill_vars]    <- lapply(data[bill_vars],    as.numeric)
data[pay_amt_vars] <- lapply(data[pay_amt_vars], as.numeric)


## ------------------------------------------------------------
## A) MaxDelay – maximaler Verzug in den letzten 6 Monaten
## ------------------------------------------------------------

data$MaxDelay <- apply(
  data[, delay_vars],
  1,
  function(x) {
    if (all(is.na(x))) NA_real_ else max(x, na.rm = TRUE)
  }
)

cat("Created feature: MaxDelay (max months in delay over last 6 months)\n")


## ------------------------------------------------------------
## B) AvgBill – durchschnittlicher Rechnungsbetrag
## ------------------------------------------------------------

data$AvgBill <- rowMeans(data[, bill_vars], na.rm = TRUE)

cat("Created feature: AvgBill (mean of bill statements over last 6 months)\n")


## ------------------------------------------------------------
## C) Payment Ratios – Zahlung / Rechnung (monatlich)
## ------------------------------------------------------------

# Einzelnes Ratio für September
data$PayRatio_SEPT <- data$PAY_SEPT_AMT / (data$BILL_SEPT + 1)

# Allgemeiner: Ratios für alle Monate in einer Schleife
for (i in seq_along(bill_vars)) {
  bill_var <- bill_vars[i]
  pay_var  <- pay_amt_vars[i]
  ratio_nm <- gsub("BILL_", "PayRatio_", bill_var)  # z.B. BILL_SEPT -> PayRatio_SEPT
  
  data[[ratio_nm]] <- data[[pay_var]] / (data[[bill_var]] + 1)
  data[[ratio_nm]] [is.na(data[[ratio_nm]])] <- 0
  
  # Inf / -Inf -> 0 (Sicherheitsnetz)
  data[[ratio_nm]][is.infinite(data[[ratio_nm]])] <- 0
}

cat("Created features: PayRatio_* (payment-to-bill ratios per month, +1 to avoid division by zero)\n")

## ------------------------------------------------------------
## D) Trends – Veränderung von April zu September
## ------------------------------------------------------------

# Rechnungs-Trend (Schuldenentwicklung)
data$BillTrend <- data$BILL_SEPT - data$BILL_APR

# Zahlungs-Trend (Zahlungsverhalten)
data$PayTrend  <- data$PAY_SEPT_AMT - data$PAY_APR_AMT

cat("Created features: BillTrend (BILL_SEPT - BILL_APR), PayTrend (PAY_SEPT_AMT - PAY_APR_AMT)\n")


## ------------------------------------------------------------
## E) TotalBill & TotalPay – Gesamtschulden & Gesamtzahlungen
## ------------------------------------------------------------

data$TotalBill <- rowSums(data[, bill_vars], na.rm = TRUE)
data$TotalPay  <- rowSums(data[, pay_amt_vars], na.rm = TRUE)

cat("Created features: TotalBill, TotalPay (row sums over 6 months)\n")

## ------------------------------------------------------------
## F) PaymentStd – Konsistenz der Zahlungen (Std-Abweichung)
## ------------------------------------------------------------

data$PaymentStd <- apply(
  data[, pay_amt_vars],
  1,
  function(x) {
    if (all(is.na(x))) NA_real_ else sd(x, na.rm = TRUE)
  }
)

cat("Created feature: PaymentStd (SD of payments over last 6 months)\n")


## ------------------------------------------------------------
## Zusammenfassung
## ------------------------------------------------------------

new_features <- c(
  "MaxDelay",
  "AvgBill",
  "PayRatio_SEPT",
  grep("^PayRatio_", names(data), value = TRUE),
  "BillTrend",
  "PayTrend",
  "TotalBill",
  "TotalPay",
  "PaymentStd"
)

cat("\nNewly engineered features:\n")
print(unique(new_features))
cat("\nFeature engineering completed.\n")


cat("Feature engineering complete!\n")

glimpse(data)

# 4.4: Verify and Encode Any Remaining Categorical Variables
cat("\n=== VERIFY CATEGORICAL VARIABLE ENCODING ===\n")

# Most caret functions handle factors automatically, but let's verify all categorical
# variables are properly encoded as factors (we did main conversions in 4.0)
cat("Checking if all categorical variables are properly encoded as factors...\n")
for(var in categorical_vars) {
  if(!is.factor(data[[var]])) {
    data[[var]] <- as.factor(data[[var]])
    cat(sprintf("Converted %s to factor\n", var))
  }
}

# Verify target variable is a factor with proper levels
if(!is.factor(data[[target_var]])) {
  data[[target_var]] <- as.factor(data[[target_var]])
  cat(sprintf("Converted target variable '%s' to factor\n", target_var))
}
cat(sprintf("\nTarget variable '%s' levels: %s\n", 
            target_var, 
            paste(levels(data[[target_var]]), collapse = ", ")))
cat("All categorical variables are properly encoded!\n")

# adjust numeric vars and predictors objects with new features

numeric_vars <- names(data)[sapply(data, is.numeric)]
numeric_predictors <- setdiff(numeric_vars, target_var)


# Last check if no NA

sum(is.na(data))

# Redo plot from line 300

# For categorical predictors vs binary target
for(var in categorical_vars[categorical_vars != target_var]) {
  cat(sprintf("\nCross-tabulation: %s vs %s\n", var, target_var))
  print(table(data[[var]], data[[target_var]]))
  cat("\nProportions:\n")
  print(prop.table(table(data[[var]], data[[target_var]]), margin = 1))
}


# 4.5: Create Training and Testing Sets

# 4.5.1: Create "new data"

cat("\n=== SPLITTING DATA INTO ARTIFICIAL NEW AND OLD DATA ===\n")

# Use stratified sampling to maintain class proportions
set.seed(123)  # For reproducibility
train_index_real <- createDataPartition(data[[target_var]], 
                                   p = 0.80,  # 80% for training
                                   list = FALSE,
                                   times = 1)

data_train <- data[train_index_real, ]
data_real <- data[-train_index_real, ]

cat(sprintf("Training set size: %d (%.1f%%)\n", 
            nrow(data_train), 
            100 * nrow(data_train) / nrow(data)))
cat(sprintf("Test set size: %d (%.1f%%)\n", 
            nrow(data_real), 
            100 * nrow(data_real) / nrow(data)))

# Verify class distribution is maintained
cat("\nClass distribution in training set:\n")
print(prop.table(table(data_train[[target_var]])))
cat("\nClass distribution in test set:\n")
print(prop.table(table(data_real[[target_var]])))

# 4.5.2: Take the rest as training and testing data

cat("\n=== SPLITTING DATA INTO TRAIN AND TEST SETS ===\n")

# Use stratified sampling to maintain class proportions
set.seed(123)  # For reproducibility
train_index <- createDataPartition(data_train[[target_var]], 
                                   p = 0.75,  # 75% for training
                                   list = FALSE,
                                   times = 1)

train_data <- data_train[train_index, ]
test_data <- data_train[-train_index, ]

cat(sprintf("Training set size: %d (%.1f%%)\n", 
            nrow(train_data), 
            100 * nrow(train_data) / nrow(data)))
cat(sprintf("Test set size: %d (%.1f%%)\n", 
            nrow(test_data), 
            100 * nrow(test_data) / nrow(data)))

# Verify class distribution is maintained
cat("\nClass distribution in training set:\n")
print(prop.table(table(train_data[[target_var]])))
cat("\nClass distribution in test set:\n")
print(prop.table(table(test_data[[target_var]])))

# 4.5.3: Feature Scaling
cat("\n=== FEATURE SCALING ===\n")
cat("Note: Some models (SVM, Neural Networks) require scaled features\n")
cat("Tree-based models (Random Forest, Gradient Boosting) don't require scaling\n")

# We'll create scaled versions for models that need it
# Identify numeric predictors (excluding target)
numeric_predictors <- setdiff(numeric_vars, target_var)

if(length(numeric_predictors) > 0) {
  # Create preprocessing object
  preproc <- preProcess(train_data[, numeric_predictors], 
                        method = c("center", "scale"))
  
  # Apply to both train and test (using train's parameters!)
  train_data_scaled <- train_data
  test_data_scaled <- test_data
  
  train_data_scaled[, numeric_predictors] <- predict(preproc, 
                                                     train_data[, numeric_predictors])
  test_data_scaled[, numeric_predictors] <- predict(preproc, 
                                                    test_data[, numeric_predictors])
  
  cat("Created scaled versions of data for SVM and Neural Networks\n")
} else {
  train_data_scaled <- train_data
  test_data_scaled <- test_data
  cat("No numeric predictors to scale\n")
}

cat("\n=== PREPROCESSING COMPLETE ===\n\n")

## ============================================================
## 4.6: Create train_small (10% stratified)
## ============================================================

# since we have many observations, we'll finetune models with a smaller dataset
# in order to make finetuning timewise feasible (taking 10% stratified of train)
# + we're adding class weights since our dataset is inbalanced (78% no vs 22% yes)

set.seed(123)

# 10% stratified subset from train_data for fast hyperparameter search
idx_small <- createDataPartition(train_data[[target_var]], p = 0.10, list = FALSE)
train_small <- train_data[idx_small, ]

cat(sprintf("train_small size: %d (%.1f%% of train_data)\n",
            nrow(train_small), 100*nrow(train_small)/nrow(train_data)))

cat("\nClass distribution in train_data:\n")
print(prop.table(table(train_data[[target_var]])))

cat("\nClass distribution in train_small:\n")
print(prop.table(table(train_small[[target_var]])))

## ------------------------------------------------------------
## Create scaled version of train_small (using preproc from train_data)
## ------------------------------------------------------------

train_small_scaled <- train_small

# Apply the SAME scaling fitted on train_data
train_small_scaled[, numeric_predictors] <- predict(preproc,
                                                    train_small[, numeric_predictors])

cat("Created train_small_scaled using preproc fitted on train_data.\n")


## ============================================================
## 4.7: Create class weights due to imbalance
## ============================================================

# base imbalance ratio (No/Yes) in TRAINING data
p_no  <- mean(train_data[[target_var]] == "No")
p_yes <- mean(train_data[[target_var]] == "Yes")
ratio <- p_no / p_yes  # should be around 0.78/0.22 = 3.55

w_base <- ratio

make_weights <- function(y, w_yes) {
  ifelse(y == "Yes", w_yes, 1)
}

w_train <- make_weights(train_data[[target_var]], w_base)
w_small <- make_weights(train_small[[target_var]], w_base)

cat(sprintf("\nUsing BASE class weight (Yes vs No=1): %.2f\n", w_base))


################################################################################
# SECTION 5: MODEL BUILDING    
################################################################################
# We'll build at least 4 different models as required
# All models will use cross-validation for robust performance estimation

cat("\n\n########## MODEL BUILDING ##########\n\n")

# We'll use 10-fold cross-validation for all models
ctrl <- trainControl(
  method = "cv",              # Cross-validation
  number = 10,                # 10 folds
  summaryFunction = twoClassSummary,  # For binary classification metrics
  classProbs = TRUE,          # Need class probabilities for ROC
  savePredictions = "final",  # Save predictions for analysis
  verboseIter = FALSE         # Set to TRUE to see progress
)

# Create formula (all predictors vs target)
# Adjust this if you want to exclude certain variables
formula <- as.formula(paste(target_var, "~ ."))

# Initialize results storage
# Create an empty data frame to store performance metrics for different models
model_results <- data.frame(
  Model = character(),               # Model name (e.g., Logistic Regression, Random Forest)
  AUC_Train = numeric(),        # Accuracy on the training dataset
  Accuracy_Test = numeric(),         # Accuracy on the test dataset
  Sensitivity_Test = numeric(),      # True positive rate on the test set
  Specificity_Test = numeric(),      # True negative rate on the test set
  AUC_Test = numeric(),              # Area Under the ROC Curve on the test set
  Training_Time = numeric(),         # Time taken to train the model (in seconds)
  Accuracy_Test_OptThr    = numeric(),
  Sensitivity_Test_OptThr = numeric(),
  Specificity_Test_OptThr = numeric(),
  Best_Threshold = numeric(),
  Cost_Train_USD = numeric(),
  Cost_Test_USD = numeric(),
  stringsAsFactors = FALSE           # Keep text columns as character, not factors
)

## ============================================================
## 5.2: COST-BASED THRESHOLD OPTIMIZATION (GLOBAL HELPERS)
## ============================================================

# calculation of average costs

# We approximate losses as EAD × LGD with a fixed LGD of 60% due to missing recovery data;
# statement balance is used as proxy for EAD

LGD <- 0.6   # Loss given Default, usually people are able to pay back 60% at defaulting of the outstanding
interest <- 0.02 # typical provision rate a credit card issuer takes on total transactions
customer_lifetime <- 10 # in steps of '6 months' since we have data for 6 months, 10 = 10*6 months = 5 years, approximate healthy customer lifetime
USD_per_NTDollar <- 1/32 # USD per Taiwanese dollar in 2005, approx.

# Costs per Default approx
Cost_per_Default <- mean(
  data$BILL_SEPT[data$DEFAULT == "Yes"] * LGD,
  na.rm = TRUE
)

Cost_per_Default

# in USD
Cost_per_Default_USD <- Cost_per_Default*USD_per_NTDollar

Cost_per_Default_USD

# Oppportunity Costs for not taking in a "healthy customer"
Opportunity_Cost_per_NonDefault <- mean(
  data$TotalPay[data$DEFAULT == "No"] * interest * customer_lifetime, na.rm = TRUE
)

Opportunity_Cost_per_NonDefault

Opportunity_Cost_per_NonDefault_USD <- Opportunity_Cost_per_NonDefault*USD_per_NTDollar

Opportunity_Cost_per_NonDefault_USD

# Case study costs in USD:
C_FN <- 910   # missed default (False Negative)
C_FP <- 219   # false alarm / rejected good customer (False Positive)

# Helper: compute FP/FN cost for a given threshold
threshold_search_cost <- function(prob_yes, y_true, C_FN = 910, C_FP = 219,
                                  thresholds = seq(0.01, 0.99, by = 0.01)) {
  y_true <- factor(y_true, levels = c("No", "Yes"))
  
  res <- lapply(thresholds, function(t) {
    pred <- factor(ifelse(prob_yes >= t, "Yes", "No"), levels = c("No", "Yes"))
    tab <- table(pred, y_true)
    
    FP <- ifelse("Yes" %in% rownames(tab) && "No"  %in% colnames(tab), tab["Yes","No"], 0)
    FN <- ifelse("No"  %in% rownames(tab) && "Yes" %in% colnames(tab), tab["No","Yes"], 0)
    
    cost <- C_FN * FN + C_FP * FP
    data.frame(threshold = t, cost = cost, FP = FP, FN = FN)
  })
  
  res <- do.call(rbind, res)
  best <- res[which.min(res$cost), ]
  list(best = best, curve = res)
}

# Helper: compute cost on a dataset given predicted labels + true labels
compute_cost_from_pred <- function(pred, y_true, C_FN = 910, C_FP = 219) {
  pred   <- factor(pred,   levels = c("No","Yes"))
  y_true <- factor(y_true, levels = c("No","Yes"))
  tab <- table(pred, y_true)
  
  FP <- ifelse("Yes" %in% rownames(tab) && "No"  %in% colnames(tab), tab["Yes","No"], 0)
  FN <- ifelse("No"  %in% rownames(tab) && "Yes" %in% colnames(tab), tab["No","Yes"], 0)
  
  C_FN * FN + C_FP * FP
}



################################################################################
# MODEL 1: LOGISTIC REGRESSION 
################################################################################

cat("\n=== MODEL 1: LOGISTIC REGRESSION ===\n")

# Train logistic regression with cross-validation
start_time <- Sys.time()
set.seed(123)

log_model <- train(               
  formula,                        # Model formula defining outcome ~ predictors
  data = train_data_scaled,       # Training dataset
  method = "glm",                  # Use generalized linear model (logistic regression)
  family = "binomial",             # Specify binomial family for binary classification
  trControl = ctrl,                # Cross-validation settings defined earlier
  metric = "ROC",                  # Optimize model performance based on ROC AUC
  weights = w_train                # weights new into it
)

end_time <- Sys.time()
log_time <- as.numeric(difftime(end_time, start_time, units = "secs"))

cat("Training complete!\n")
print(log_model)

# Variable importance
cat("\nVariable Importance (Logistic Regression):\n")
log_importance <- varImp(log_model)
print(log_importance)
plot(log_importance, main = "Variable Importance - Logistic Regression")

# Predictions on training set
log_pred_train <- predict(log_model, train_data_scaled)
log_pred_train_prob <- predict(log_model, train_data_scaled, type = "prob")

# Predictions on test set
log_pred_test <- predict(log_model, test_data_scaled)
log_pred_test_prob <- predict(log_model, test_data_scaled, type = "prob")

# Evaluate performance on test set
log_cm_test <- confusionMatrix(log_pred_test, test_data_scaled[[target_var]], positive = "Yes")
cat("\nTest Set Performance:\n")
print(log_cm_test)

# Calculate AUC
log_roc_test <- roc(test_data_scaled[[target_var]], log_pred_test_prob[, 2], levels = c("No", "Yes"))
log_auc_test <- auc(log_roc_test)
cat(sprintf("\nTest Set AUC: %.4f\n", log_auc_test))

# Store results
model_results <- rbind(model_results, data.frame(
  Model = "Logistic Regression",
  AUC_Train = max(log_model$results$ROC),
  Accuracy_Test = log_cm_test$overall["Accuracy"],
  Sensitivity_Test = log_cm_test$byClass["Sensitivity"],
  Specificity_Test = log_cm_test$byClass["Specificity"],
  AUC_Test = log_auc_test,
  Training_Time = log_time
))

## ------------------------------------------------------------
## COST-OPTIMAL THRESHOLD – LOGISTIC REGRESSION (weighted)
## ------------------------------------------------------------

# probabilities
log_prob_train_yes <- log_pred_train_prob[, "Yes"]
log_prob_test_yes  <- log_pred_test_prob[, "Yes"]

# threshold search on TRAIN
thr_log <- threshold_search_cost(
  prob_yes = log_prob_train_yes,
  y_true  = train_data_scaled[[target_var]],
  C_FN = C_FN, C_FP = C_FP
)
best_thr_log <- thr_log$best$threshold

# TRAIN cost
log_pred_train_thr <- factor(
  ifelse(log_prob_train_yes >= best_thr_log, "Yes", "No"),
  levels = c("No","Yes")
)
log_cost_train <- compute_cost_from_pred(
  log_pred_train_thr, train_data_scaled[[target_var]], C_FN, C_FP
)

# TEST cost
log_pred_test_thr <- factor(
  ifelse(log_prob_test_yes >= best_thr_log, "Yes", "No"),
  levels = c("No","Yes")
)

log_cm_test_thr <- confusionMatrix(
  log_pred_test_thr,
  test_data_scaled[[target_var]],
  positive = "Yes"
)

log_cost_test <- compute_cost_from_pred(
  log_pred_test_thr, test_data_scaled[[target_var]], C_FN, C_FP
)

# write to model_results
row <- which(model_results$Model == "Logistic Regression")
if(length(row) != 1) stop("Logistic Regression row not unique.")
model_results$Best_Threshold[row] <- best_thr_log
model_results$Cost_Train_USD[row] <- log_cost_train
model_results$Cost_Test_USD[row]  <- log_cost_test
model_results$Accuracy_Test_OptThr[row]    <- log_cm_test_thr$overall["Accuracy"]
model_results$Sensitivity_Test_OptThr[row] <- log_cm_test_thr$byClass["Sensitivity"]
model_results$Specificity_Test_OptThr[row] <- log_cm_test_thr$byClass["Specificity"]


################################################################################
# MODEL 2: REGULARIZED LOGISTIC REGRESSION 
################################################################################

cat("\n=== MODEL 2: LASSO LOGISTIC REGRESSION ===\n")

# Train Lasso with cross-validation
start_time <- Sys.time()
set.seed(123)

# Define parameter grid for lambda (regularization strength)
lasso_grid <- expand.grid(
  alpha = 1,  # 1 = Lasso, 0 = Ridge
  lambda = seq(0.0001, 0.1, length = 20)
)

lasso_model <- train(
  formula,
  data = train_data_scaled,
  method = "glmnet",
  family = "binomial",
  trControl = ctrl,
  tuneGrid = lasso_grid,
  metric = "ROC",
  weights = w_train
)

end_time <- Sys.time()
lasso_time <- as.numeric(difftime(end_time, start_time, units = "secs"))

cat("Training complete!\n")
print(lasso_model)

# Plot tuning results
plot(lasso_model, main = "Lasso Model Tuning")

# Best lambda
cat(sprintf("\nBest lambda: %.6f\n", lasso_model$bestTune$lambda))

# Variable importance
cat("\nVariable Importance (Lasso):\n")
lasso_importance <- varImp(lasso_model)
print(lasso_importance)
plot(lasso_importance, main = "Variable Importance - Lasso")

# Predictions
lasso_pred_train <- predict(lasso_model, train_data_scaled)
lasso_pred_train_prob <- predict(lasso_model, train_data_scaled, type = "prob")
lasso_pred_test <- predict(lasso_model, test_data_scaled)
lasso_pred_test_prob <- predict(lasso_model, test_data_scaled, type = "prob")

# Evaluate
lasso_cm_test <- confusionMatrix(lasso_pred_test, test_data_scaled[[target_var]], positive = "Yes")
cat("\nTest Set Performance:\n")
print(lasso_cm_test)

lasso_roc_test <- roc(test_data_scaled[[target_var]], lasso_pred_test_prob[, 2], levels = c("No", "Yes"))
lasso_auc_test <- auc(lasso_roc_test)
cat(sprintf("\nTest Set AUC: %.4f\n", lasso_auc_test))

# Store results
model_results <- rbind(model_results, data.frame(
  Model = "Lasso Regression",
  AUC_Train = max(lasso_model$results$ROC),
  Accuracy_Test = lasso_cm_test$overall["Accuracy"],
  Sensitivity_Test = lasso_cm_test$byClass["Sensitivity"],
  Specificity_Test = lasso_cm_test$byClass["Specificity"],
  AUC_Test = lasso_auc_test,
  Training_Time = lasso_time,
  Accuracy_Test_OptThr = NA_real_,
  Sensitivity_Test_OptThr = NA_real_,
  Specificity_Test_OptThr = NA_real_,
  Best_Threshold = NA_real_,
  Cost_Train_USD = NA_real_,
  Cost_Test_USD  = NA_real_,
  stringsAsFactors = FALSE
))

## ------------------------------------------------------------
## COST-OPTIMAL THRESHOLD – LASSO (weighted)
## ------------------------------------------------------------

lasso_prob_train_yes <- lasso_pred_train_prob[, "Yes"]
lasso_prob_test_yes  <- lasso_pred_test_prob[, "Yes"]

thr_lasso <- threshold_search_cost(
  lasso_prob_train_yes,
  train_data_scaled[[target_var]],
  C_FN, C_FP
)
best_thr_lasso <- thr_lasso$best$threshold

lasso_pred_train_thr <- factor(
  ifelse(lasso_prob_train_yes >= best_thr_lasso, "Yes", "No"),
  levels = c("No","Yes")
)
lasso_cost_train <- compute_cost_from_pred(
  lasso_pred_train_thr, train_data_scaled[[target_var]], C_FN, C_FP
)

lasso_pred_test_thr <- factor(
  ifelse(lasso_prob_test_yes >= best_thr_lasso, "Yes", "No"),
  levels = c("No","Yes")
)

lasso_cm_test_thr <- confusionMatrix(
  lasso_pred_test_thr,
  test_data_scaled[[target_var]],
  positive = "Yes"
)

lasso_cost_test <- compute_cost_from_pred(
  lasso_pred_test_thr, test_data_scaled[[target_var]], C_FN, C_FP
)

row <- which(model_results$Model == "Lasso Regression")
if(length(row) != 1) stop("Lasso row not unique.")
model_results$Best_Threshold[row] <- best_thr_lasso
model_results$Cost_Train_USD[row] <- lasso_cost_train
model_results$Cost_Test_USD[row]  <- lasso_cost_test
model_results$Accuracy_Test_OptThr[row]    <- lasso_cm_test_thr$overall["Accuracy"]
model_results$Sensitivity_Test_OptThr[row] <- lasso_cm_test_thr$byClass["Sensitivity"]
model_results$Specificity_Test_OptThr[row] <- lasso_cm_test_thr$byClass["Specificity"]


################################################################################
# MODEL 3: RIDGE LOGISTIC REGRESSION
################################################################################

cat("\n=== MODEL 3: RIDGE LOGISTIC REGRESSION ===\n")

start_time <- Sys.time()
set.seed(123)

# Grid für Ridge: alpha = 0
ridge_grid <- expand.grid(
  alpha = 0,                             # 0 = Ridge
  lambda = seq(0.00001, 1, length = 50)   # etwas breiterer Bereich als bei Lasso
)

ridge_model <- train(
  formula,
  data = train_data_scaled,  # wie bei Lasso: skalierte Daten!
  method = "glmnet",
  family = "binomial",
  trControl = ctrl,
  tuneGrid = ridge_grid,
  metric = "ROC",
  weights = w_train
)

end_time <- Sys.time()
ridge_time <- as.numeric(difftime(end_time, start_time, units = "secs"))

cat("Training complete!\n")
print(ridge_model)

# Plot tuning results
plot(ridge_model, main = "Ridge Model Tuning")

# Bestes Lambda
cat(sprintf("\nBest lambda (Ridge): %.6f\n", ridge_model$bestTune$lambda))

# Variable Importance
cat("\nVariable Importance (Ridge):\n")
ridge_importance <- varImp(ridge_model)
print(ridge_importance)
plot(ridge_importance, main = "Variable Importance - Ridge")

# Predictions
ridge_pred_train     <- predict(ridge_model, train_data_scaled)
ridge_pred_train_prob <- predict(ridge_model, train_data_scaled, type = "prob")
ridge_pred_test      <- predict(ridge_model, test_data_scaled)
ridge_pred_test_prob <- predict(ridge_model, test_data_scaled, type = "prob")

# Evaluate
ridge_cm_test <- confusionMatrix(ridge_pred_test,
                                 test_data_scaled[[target_var]],
                                 positive = "Yes")
cat("\nTest Set Performance (Ridge):\n")
print(ridge_cm_test)

ridge_roc_test <- roc(test_data_scaled[[target_var]],
                      ridge_pred_test_prob[, 2],
                      levels = c("No", "Yes"))
ridge_auc_test <- auc(ridge_roc_test)
cat(sprintf("\nTest Set AUC (Ridge): %.4f\n", ridge_auc_test))

# In Vergleichstabelle einfügen
model_results <- rbind(model_results, data.frame(
  Model            = "Ridge Regression",
  AUC_Train   = max(ridge_model$results$ROC),   # eigentlich: CV-ROC, nicht Accuracy
  Accuracy_Test    = ridge_cm_test$overall["Accuracy"],
  Sensitivity_Test = ridge_cm_test$byClass["Sensitivity"],
  Specificity_Test = ridge_cm_test$byClass["Specificity"],
  AUC_Test         = ridge_auc_test,
  Training_Time    = ridge_time,
  Accuracy_Test_OptThr = NA_real_,
  Sensitivity_Test_OptThr = NA_real_,
  Specificity_Test_OptThr = NA_real_,
  Best_Threshold = NA_real_,
  Cost_Train_USD = NA_real_,
  Cost_Test_USD  = NA_real_,
  
  stringsAsFactors = FALSE
))

## ------------------------------------------------------------
## COST-OPTIMAL THRESHOLD – RIDGE (weighted)
## ------------------------------------------------------------

ridge_prob_train_yes <- ridge_pred_train_prob[, "Yes"]
ridge_prob_test_yes  <- ridge_pred_test_prob[, "Yes"]

thr_ridge <- threshold_search_cost(
  ridge_prob_train_yes,
  train_data_scaled[[target_var]],
  C_FN, C_FP
)
best_thr_ridge <- thr_ridge$best$threshold

ridge_pred_train_thr <- factor(
  ifelse(ridge_prob_train_yes >= best_thr_ridge, "Yes", "No"),
  levels = c("No","Yes")
)
ridge_cost_train <- compute_cost_from_pred(
  ridge_pred_train_thr, train_data_scaled[[target_var]], C_FN, C_FP
)

ridge_pred_test_thr <- factor(
  ifelse(ridge_prob_test_yes >= best_thr_ridge, "Yes", "No"),
  levels = c("No","Yes")
)
ridge_cm_test_thr <- confusionMatrix(
  ridge_pred_test_thr,
  test_data_scaled[[target_var]],
  positive = "Yes"
)
ridge_cost_test <- compute_cost_from_pred(
  ridge_pred_test_thr, test_data_scaled[[target_var]], C_FN, C_FP
)

row <- which(model_results$Model == "Ridge Regression")
if(length(row) != 1) stop("Ridge row not unique.")
model_results$Best_Threshold[row] <- best_thr_ridge
model_results$Cost_Train_USD[row] <- ridge_cost_train
model_results$Cost_Test_USD[row]  <- ridge_cost_test
model_results$Accuracy_Test_OptThr[row]    <- ridge_cm_test_thr$overall["Accuracy"]
model_results$Sensitivity_Test_OptThr[row] <- ridge_cm_test_thr$byClass["Sensitivity"]
model_results$Specificity_Test_OptThr[row] <- ridge_cm_test_thr$byClass["Specificity"]


################################################################################
# MODEL 4: DECISION TREE
################################################################################

#    
cat("\n=== MODEL 4: DECISION TREE ===\n")

# Train decision tree with cross-validation
start_time <- Sys.time()
set.seed(123)

# Define parameter grid for complexity parameter
tree_grid <- expand.grid(
  cp = seq(0.001, 0.1, length = 20)  # Complexity parameter
)

tree_model <- train(
  formula,
  data = train_data,
  method = "rpart",
  trControl = ctrl,
  tuneGrid = tree_grid,
  metric = "ROC",
  weights = w_train
)

end_time <- Sys.time()
tree_time <- as.numeric(difftime(end_time, start_time, units = "secs"))

cat("Training complete!\n")
print(tree_model)

# Plot tuning results
plot(tree_model, main = "Decision Tree Tuning")

# Visualize the tree
library(rpart.plot)
rpart.plot(tree_model$finalModel, 
           main = "Decision Tree Structure",
           extra = 2,
           under = TRUE,
           faclen = 0)

# Variable importance
cat("\nVariable Importance (Decision Tree):\n")
tree_importance <- varImp(tree_model)
print(tree_importance)
plot(tree_importance, main = "Variable Importance - Decision Tree")

# Predictions
tree_pred_train <- predict(tree_model, train_data)
tree_pred_train_prob <- predict(tree_model, train_data, type = "prob")
tree_pred_test <- predict(tree_model, test_data)
tree_pred_test_prob <- predict(tree_model, test_data, type = "prob")

# Evaluate
tree_cm_test <- confusionMatrix(tree_pred_test, test_data[[target_var]], positive = "Yes")
cat("\nTest Set Performance:\n")
print(tree_cm_test)

tree_roc_test <- roc(test_data[[target_var]], tree_pred_test_prob[, 2], levels = c("No", "Yes"))
tree_auc_test <- auc(tree_roc_test)
cat(sprintf("\nTest Set AUC: %.4f\n", tree_auc_test))

# Store results
model_results <- rbind(model_results, data.frame(
  Model = "Decision Tree",
  AUC_Train = max(tree_model$results$ROC),
  Accuracy_Test = tree_cm_test$overall["Accuracy"],
  Sensitivity_Test = tree_cm_test$byClass["Sensitivity"],
  Specificity_Test = tree_cm_test$byClass["Specificity"],
  AUC_Test = tree_auc_test,
  Training_Time = tree_time,
  Accuracy_Test_OptThr = NA_real_,
  Sensitivity_Test_OptThr = NA_real_,
  Specificity_Test_OptThr = NA_real_,
  Best_Threshold = NA_real_,
  Cost_Train_USD = NA_real_,
  Cost_Test_USD  = NA_real_,
  
  stringsAsFactors = FALSE
))

## ------------------------------------------------------------
## COST-OPTIMAL THRESHOLD – DECISION TREE
## ------------------------------------------------------------

tree_prob_train_yes <- tree_pred_train_prob[, "Yes"]
tree_prob_test_yes  <- tree_pred_test_prob[, "Yes"]

thr_tree <- threshold_search_cost(
  tree_prob_train_yes,
  train_data[[target_var]],
  C_FN, C_FP
)
best_thr_tree <- thr_tree$best$threshold

tree_pred_train_thr <- factor(
  ifelse(tree_prob_train_yes >= best_thr_tree, "Yes", "No"),
  levels = c("No","Yes")
)
tree_cost_train <- compute_cost_from_pred(
  tree_pred_train_thr, train_data[[target_var]], C_FN, C_FP
)

tree_pred_test_thr <- factor(
  ifelse(tree_prob_test_yes >= best_thr_tree, "Yes", "No"),
  levels = c("No","Yes")
)
tree_cm_test_thr <- confusionMatrix(
  tree_pred_test_thr,
  test_data[[target_var]],
  positive = "Yes"
)
tree_cost_test <- compute_cost_from_pred(
  tree_pred_test_thr, test_data[[target_var]], C_FN, C_FP
)

row <- which(model_results$Model == "Decision Tree")
if(length(row) != 1) stop("Decision Tree row not unique.")
model_results$Best_Threshold[row] <- best_thr_tree
model_results$Cost_Train_USD[row] <- tree_cost_train
model_results$Cost_Test_USD[row]  <- tree_cost_test
model_results$Accuracy_Test_OptThr[row]    <- tree_cm_test_thr$overall["Accuracy"]
model_results$Sensitivity_Test_OptThr[row] <- tree_cm_test_thr$byClass["Sensitivity"]
model_results$Specificity_Test_OptThr[row] <- tree_cm_test_thr$byClass["Specificity"]


################################################################################
# MODEL 5: RANDOM FOREST (ranger) - train_small tuning + weighted final fit
################################################################################

cat("\n=== MODEL 5: RANDOM FOREST (ranger, weighted) ===\n")

# -------------------------------
# Model 5A: tune on train_small
# -------------------------------
set.seed(123)
start_time <- Sys.time()

x_small <- as.data.frame(train_small[, setdiff(names(train_small), target_var)])
y_small <- train_small[[target_var]]

n_features <- ncol(x_small)

rf_grid_small <- expand.grid(
  mtry = unique(round(c(sqrt(n_features), n_features/3, n_features/2))),
  splitrule = "gini",
  min.node.size = c(1, 20, 50, 100, 120, 135, 150, 165, 180, 200)
)

rf_tuned_small <- train(
  x = x_small,
  y = y_small,
  method = "ranger",
  trControl = ctrl,
  tuneGrid = rf_grid_small,
  metric = "ROC",
  importance = "impurity",
  num.trees = 500,
  weights = w_small
)

rf_tune_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

cat("train_small tuning complete. Best parameters:\n")
print(rf_tuned_small$bestTune)

print(plot(rf_tuned_small, main = "RF (ranger) Tuning – train_small"))

# -----------------------------------------
# MODEL 5B: FINAL FIT on train_data (once) + TEST evaluation
# -----------------------------------------
cat("\n--- MODEL 5B: Final RF fit on full train_data ---\n")

set.seed(123)
start_time <- Sys.time()

x_train <- as.data.frame(train_data[, setdiff(names(train_data), target_var)])
y_train <- train_data[[target_var]]

mtry_best <- rf_tuned_small$bestTune$mtry
node_best <- rf_tuned_small$bestTune$min.node.size

node_grid_full <- sort(unique(c(
  max(1, round(node_best * 0.8)),
  node_best,
  round(node_best * 1.2),
  round(node_best * 1.5),
  round(node_best * 2.0),
  round(node_best * 2.5),
  round(node_best * 3.0)
)))

rf_grid_full <- expand.grid(
  mtry = mtry_best,
  splitrule = "gini",
  min.node.size = node_grid_full
)

rf_model <- train(
  x = x_train,
  y = y_train,
  method = "ranger",
  trControl = ctrl,
  tuneGrid = rf_grid_full,
  metric = "ROC",
  importance = "impurity",
  num.trees = 500,
  weights = w_train
)

rf_fit_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
rf_time <- rf_tune_time + rf_fit_time

cat("Final RF training complete!\n")
print(rf_model)

print(plot(rf_model, main = "RF (ranger) Final Mini-Grid – full train"))

# Variable importance (same object names as your original block)
cat("\nVariable Importance (Random Forest - ranger):\n")
rf_importance <- varImp(rf_model)
print(rf_importance)
plot(rf_importance, main = "Variable Importance – Random Forest")

rf_pred_train <- predict(rf_model, train_data)
rf_pred_train_prob <- predict(rf_model, train_data, type = "prob")

rf_pred_test <- predict(rf_model, test_data)
rf_pred_test_prob <- predict(rf_model, test_data, type = "prob")

# Evaluate on test_data (threshold=0.5 for now; threshold policy comes later)
rf_cm_test <- confusionMatrix(rf_pred_test, test_data[[target_var]], positive = "Yes")
cat("\nTest Set Performance (threshold=0.5 default):\n")
print(rf_cm_test)

rf_roc_test <- roc(test_data[[target_var]],
                   rf_pred_test_prob[, "Yes"],
                   levels = c("No", "Yes"))

rf_auc_test <- auc(rf_roc_test)
cat(sprintf("\nTest Set AUC: %.4f\n", rf_auc_test))

# Store results (same structure; AUC_Train is CV ROC on full train_data)
model_results <- rbind(model_results, data.frame(
  Model = "Random Forest",
  AUC_Train = max(rf_model$results$ROC),
  Accuracy_Test = rf_cm_test$overall["Accuracy"],
  Sensitivity_Test = rf_cm_test$byClass["Sensitivity"],
  Specificity_Test = rf_cm_test$byClass["Specificity"],
  AUC_Test = rf_auc_test,
  Training_Time = rf_time,
  Accuracy_Test_OptThr = NA_real_,
  Sensitivity_Test_OptThr = NA_real_,
  Specificity_Test_OptThr = NA_real_,
  Best_Threshold = NA_real_,
  Cost_Train_USD = NA_real_,
  Cost_Test_USD  = NA_real_,
  
  stringsAsFactors = FALSE
))

## ------------------------------------------------------------
## COST-OPTIMAL THRESHOLD for RF (threshold chosen on TRAIN)
## ------------------------------------------------------------

# 1) choose threshold on training probabilities
rf_prob_train_yes <- rf_pred_train_prob[, "Yes"]
thr_rf <- threshold_search_cost(rf_prob_train_yes, train_data[[target_var]], C_FN, C_FP)
best_thr_rf <- thr_rf$best$threshold

# compute TRAIN cost at chosen threshold
rf_pred_train_thr <- factor(
  ifelse(rf_prob_train_yes >= best_thr_rf, "Yes", "No"),
  levels = c("No","Yes")
)

rf_cost_train <- compute_cost_from_pred(
  rf_pred_train_thr,
  train_data[[target_var]],
  C_FN,
  C_FP
)

cat(sprintf("\n[RF] Best threshold on train_data (min cost): %.2f\n", best_thr_rf))
print(thr_rf$best)

# optional plot (base R)
plot(thr_rf$curve$threshold, thr_rf$curve$cost, type="l",
     main="RF: Cost vs Threshold (chosen on train_data)",
     xlab="Threshold", ylab="Cost (USD)")

# 2) apply chosen threshold to test probabilities
rf_prob_test_yes <- rf_pred_test_prob[, "Yes"]
rf_pred_test_thr <- factor(ifelse(rf_prob_test_yes >= best_thr_rf, "Yes", "No"),
                           levels = c("No","Yes"))

rf_cm_test_thr <- confusionMatrix(rf_pred_test_thr, test_data[[target_var]], positive = "Yes")
cat("\n[RF] Test Performance (cost-optimized threshold):\n")
print(rf_cm_test_thr)

rf_cost_test <- compute_cost_from_pred(rf_pred_test_thr, test_data[[target_var]], C_FN, C_FP)
cat(sprintf("\n[RF] Test Cost (USD): %.2f\n", rf_cost_test))

rf_row <- which(model_results$Model == "Random Forest")

# 3) write into model_results row
if(length(rf_row) != 1) {
  stop(sprintf(
    "RF row not uniquely found in model_results. Found %d matches. Check Model names.",
    length(rf_row)
  ))
}

model_results$Best_Threshold[rf_row] <- best_thr_rf
model_results$Cost_Train_USD[rf_row] <- rf_cost_train
model_results$Cost_Test_USD[rf_row]  <- rf_cost_test
model_results$Accuracy_Test_OptThr[rf_row]    <- rf_cm_test_thr$overall["Accuracy"]
model_results$Sensitivity_Test_OptThr[rf_row] <- rf_cm_test_thr$byClass["Sensitivity"]
model_results$Specificity_Test_OptThr[rf_row] <- rf_cm_test_thr$byClass["Specificity"]


################################################################################
# MODEL 6: GRADIENT BOOSTING
################################################################################

cat("\n=== MODEL 6: GRADIENT BOOSTING ===\n")

################################################################################
# MODEL 6A: GRADIENT BOOSTING – tune on train_small
################################################################################

# Train Gradient Boosting with cross-validation
start_time <- Sys.time()
set.seed(123)

# Define parameter grid (simplified for speed)
gbm_grid_small <- expand.grid(
  n.trees = c(100, 300, 400),           # Number of trees
  interaction.depth = c(1, 2, 3, 4, 5), # Tree depth
  shrinkage = c(0.01, 0.03, 0.05, 0.07, 0.1),     # Learning rate
  n.minobsinnode = c(10)              # Minimum observations in terminal nodes
)

gbm_tuned_small <- train(
  formula,
  data = train_small,
  method = "gbm",
  trControl = ctrl,
  tuneGrid = gbm_grid_small,
  metric = "ROC",
  verbose = FALSE,
  weights = w_small
)

gbm_tune_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

cat("GBM train_small tuning complete. Best parameters:\n")
print(gbm_tuned_small$bestTune)

plot(gbm_tuned_small, main = "GBM Tuning – train_small")

################################################################################
# MODEL 6B: GRADIENT BOOSTING – final fit on full train_data
################################################################################

cat("\n=== MODEL 6B: GRADIENT BOOSTING (final fit) ===\n")

set.seed(123)
start_time <- Sys.time()

best_depth     <- gbm_tuned_small$bestTune$interaction.depth
best_shrinkage <- gbm_tuned_small$bestTune$shrinkage

gbm_grid_full <- expand.grid(
  n.trees = c(
    gbm_tuned_small$bestTune$n.trees,
    gbm_tuned_small$bestTune$n.trees * 1.5
  ),
  interaction.depth = best_depth,
  shrinkage = best_shrinkage,
  n.minobsinnode = 10
)

gbm_model <- train(
  formula,
  data = train_data,
  method = "gbm",
  trControl = ctrl,
  tuneGrid = gbm_grid_full,
  metric = "ROC",
  verbose = FALSE,
  weights = w_train
)

gbm_fit_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
gbm_time <- gbm_tune_time + gbm_fit_time

cat("Final GBM training complete!\n")
print(gbm_model)

plot(gbm_model, main = "GBM Final Mini-Grid – full train")


# Variable importance
cat("\nVariable Importance (Gradient Boosting):\n")
gbm_importance <- varImp(gbm_model)
print(gbm_importance)
plot(gbm_importance, main = "Variable Importance - Gradient Boosting")

# Predictions
gbm_pred_train <- predict(gbm_model, train_data)
gbm_pred_train_prob <- predict(gbm_model, train_data, type = "prob")
gbm_pred_test <- predict(gbm_model, test_data)
gbm_pred_test_prob <- predict(gbm_model, test_data, type = "prob")

# Evaluate
gbm_cm_test <- confusionMatrix(gbm_pred_test, test_data[[target_var]], positive = "Yes")
cat("\nTest Set Performance:\n")
print(gbm_cm_test)

gbm_roc_test <- roc(test_data[[target_var]], gbm_pred_test_prob[, 2], levels = c("No", "Yes"))
gbm_auc_test <- auc(gbm_roc_test)
cat(sprintf("\nTest Set AUC: %.4f\n", gbm_auc_test))

# Store results
model_results <- rbind(model_results, data.frame(
  Model = "Gradient Boosting",
  AUC_Train = max(gbm_model$results$ROC),
  Accuracy_Test = gbm_cm_test$overall["Accuracy"],
  Sensitivity_Test = gbm_cm_test$byClass["Sensitivity"],
  Specificity_Test = gbm_cm_test$byClass["Specificity"],
  AUC_Test = gbm_auc_test,
  Training_Time = gbm_time,
  Accuracy_Test_OptThr = NA_real_,
  Sensitivity_Test_OptThr = NA_real_,
  Specificity_Test_OptThr = NA_real_,
  Best_Threshold = NA_real_,
  Cost_Train_USD = NA_real_,
  Cost_Test_USD  = NA_real_,
  
  stringsAsFactors = FALSE
))

## ------------------------------------------------------------
## COST-OPTIMAL THRESHOLD – GRADIENT BOOSTING
## ------------------------------------------------------------

gbm_prob_train_yes <- gbm_pred_train_prob[, "Yes"]
gbm_prob_test_yes  <- gbm_pred_test_prob[, "Yes"]

thr_gbm <- threshold_search_cost(
  gbm_prob_train_yes,
  train_data[[target_var]],
  C_FN, C_FP
)
best_thr_gbm <- thr_gbm$best$threshold

gbm_pred_train_thr <- factor(
  ifelse(gbm_prob_train_yes >= best_thr_gbm, "Yes", "No"),
  levels = c("No","Yes")
)
gbm_cost_train <- compute_cost_from_pred(
  gbm_pred_train_thr, train_data[[target_var]], C_FN, C_FP
)

gbm_pred_test_thr <- factor(
  ifelse(gbm_prob_test_yes >= best_thr_gbm, "Yes", "No"),
  levels = c("No","Yes")
)
gbm_cm_test_thr <- confusionMatrix(
  gbm_pred_test_thr,
  test_data[[target_var]],
  positive = "Yes"
)
gbm_cost_test <- compute_cost_from_pred(
  gbm_pred_test_thr, test_data[[target_var]], C_FN, C_FP
)

row <- which(model_results$Model == "Gradient Boosting")
if(length(row) != 1) stop("GBM row not unique.")
model_results$Best_Threshold[row] <- best_thr_gbm
model_results$Cost_Train_USD[row] <- gbm_cost_train
model_results$Cost_Test_USD[row]  <- gbm_cost_test
model_results$Accuracy_Test_OptThr[row]    <- gbm_cm_test_thr$overall["Accuracy"]
model_results$Sensitivity_Test_OptThr[row] <- gbm_cm_test_thr$byClass["Sensitivity"]
model_results$Specificity_Test_OptThr[row] <- gbm_cm_test_thr$byClass["Specificity"]

#################### BONUS: GBM unweighted as comparison #######################

################################################################################
# MODEL 6.2: GRADIENT BOOSTING UNWEIGHTED
################################################################################

cat("\n=== MODEL 6: GRADIENT BOOSTING unweighted ===\n")

################################################################################
# MODEL 6AA: GRADIENT BOOSTING UNWEIGHTED – tune on train_small
################################################################################

# Train Gradient Boosting with cross-validation unweighted
start_time <- Sys.time()
set.seed(123)

# Define parameter grid (simplified for speed)
gbm_unweighted_grid_small <- expand.grid(
  n.trees = c(100, 150, 200, 300),           # Number of trees
  interaction.depth = c(1, 2, 3, 4, 5), # Tree depth
  shrinkage = c(0.01, 0.03, 0.05, 0.06, 0.08, 0.1),     # Learning rate
  n.minobsinnode = c(10)              # Minimum observations in terminal nodes
)

gbm_unweighted_tuned_small <- train(
  formula,
  data = train_small,
  method = "gbm",
  trControl = ctrl,
  tuneGrid = gbm_unweighted_grid_small,
  metric = "ROC",
  verbose = FALSE
)

gbm_unweighted_tune_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

cat("GBM train_small unweighted tuning complete. Best parameters:\n")
print(gbm_unweighted_tuned_small$bestTune)

plot(gbm_unweighted_tuned_small, main = "GBM unweighted Tuning – train_small")

################################################################################
# MODEL 6BB: GRADIENT BOOSTING UNWEIGHTED – final fit on full train_data
################################################################################

cat("\n=== MODEL 6B: GRADIENT BOOSTING unweighted (final fit) ===\n")

set.seed(123)
start_time <- Sys.time()

best_depth     <- gbm_tuned_small$bestTune$interaction.depth
best_shrinkage <- gbm_tuned_small$bestTune$shrinkage

gbm_grid_full <- expand.grid(
  n.trees = c(
    gbm_tuned_small$bestTune$n.trees,
    gbm_tuned_small$bestTune$n.trees * 1.5
  ),
  interaction.depth = best_depth,
  shrinkage = best_shrinkage,
  n.minobsinnode = 10
)

gbm_unweighted_model <- train(
  formula,
  data = train_data,
  method = "gbm",
  trControl = ctrl,
  tuneGrid = gbm_unweighted_grid_full,
  metric = "ROC",
  verbose = FALSE
)

gbm_unweighted_fit_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
gbm_unweighted_time <- gbm_unweighted_tune_time + gbm_unweighted_fit_time

cat("Final GBM unweighted training complete!\n")
print(gbm_unweighted_model)

plot(gbm_unweighted_model, main = "GBM Final unweighted Mini-Grid – full train")


# Variable importance
cat("\nVariable Importance (Gradient Boosting):\n")
gbm_unweighted_importance <- varImp(gbm_unweighted_model)
print(gbm_unweighted_importance)
plot(gbm_unweighted_importance, main = "Variable Importance - Gradient Boosting unweighted")

# Predictions
gbm_unweighted_pred_train <- predict(gbm_unweighted_model, train_data)
gbm_unweighted_pred_train_prob <- predict(gbm_unweighted_model, train_data, type = "prob")
gbm_unweighted_pred_test <- predict(gbm_unweighted_model, test_data)
gbm_unweighted_pred_test_prob <- predict(gbm_unweighted_model, test_data, type = "prob")

# Evaluate
gbm_unweighted_cm_test <- confusionMatrix(gbm_unweighted_pred_test, test_data[[target_var]], positive = "Yes")
cat("\nTest Set Performance:\n")
print(gbm_unweighted_cm_test)

gbm_unweighted_roc_test <- roc(test_data[[target_var]], gbm_unweighted_pred_test_prob[, 2], levels = c("No", "Yes"))
gbm_unweighted_auc_test <- auc(gbm_unweighted_roc_test)
cat(sprintf("\nTest Set AUC: %.4f\n", gbm_unweighted_auc_test))

# Store results
model_results <- rbind(model_results, data.frame(
  Model = "Gradient Boosting unweighted",
  AUC_Train = max(gbm_unweighted_model$results$ROC),
  Accuracy_Test = gbm_unweighted_cm_test$overall["Accuracy"],
  Sensitivity_Test = gbm_unweighted_cm_test$byClass["Sensitivity"],
  Specificity_Test = gbm_unweighted_cm_test$byClass["Specificity"],
  AUC_Test = gbm_unweighted_auc_test,
  Training_Time = gbm_unweighted_time,
  Accuracy_Test_OptThr = NA_real_,
  Sensitivity_Test_OptThr = NA_real_,
  Specificity_Test_OptThr = NA_real_,
  Best_Threshold = NA_real_,
  Cost_Train_USD = NA_real_,
  Cost_Test_USD  = NA_real_,
  
  stringsAsFactors = FALSE
))

## ------------------------------------------------------------
## COST-OPTIMAL THRESHOLD – GRADIENT BOOSTING
## ------------------------------------------------------------

gbm_unweighted_prob_train_yes <- gbm_unweighted_pred_train_prob[, "Yes"]
gbm_unweighted_prob_test_yes  <- gbm_unweighted_pred_test_prob[, "Yes"]

thr_gbm_unweighted <- threshold_search_cost(
  gbm_unweighted_prob_train_yes,
  train_data[[target_var]],
  C_FN, C_FP
)
best_thr_gbm_unweighted <- thr_gbm_unweighted$best$threshold

gbm_unweighted_pred_train_thr <- factor(
  ifelse(gbm_unweighted_prob_train_yes >= best_thr_gbm_unweighted, "Yes", "No"),
  levels = c("No","Yes")
)
gbm_unweighted_cost_train <- compute_cost_from_pred(
  gbm_unweighted_pred_train_thr, train_data[[target_var]], C_FN, C_FP
)

gbm_unweighted_pred_test_thr <- factor(
  ifelse(gbm_unweighted_prob_test_yes >= best_thr_gbm_unweighted, "Yes", "No"),
  levels = c("No","Yes")
)
gbm_unweighted_cm_test_thr <- confusionMatrix(
  gbm_unweighted_pred_test_thr,
  test_data[[target_var]],
  positive = "Yes"
)
gbm_unweighted_cost_test <- compute_cost_from_pred(
  gbm_unweighted_pred_test_thr, test_data[[target_var]], C_FN, C_FP
)

row <- which(model_results$Model == "Gradient Boosting unweighted")
if(length(row) != 1) stop("GBM row not unique.")
model_results$Best_Threshold[row] <- best_thr_gbm_unweighted
model_results$Cost_Train_USD[row] <- gbm_unweighted_cost_train
model_results$Cost_Test_USD[row]  <- gbm_unweighted_cost_test
model_results$Accuracy_Test_OptThr[row]    <- gbm_unweighted_cm_test_thr$overall["Accuracy"]
model_results$Sensitivity_Test_OptThr[row] <- gbm_unweighted_cm_test_thr$byClass["Sensitivity"]
model_results$Specificity_Test_OptThr[row] <- gbm_unweighted_cm_test_thr$byClass["Specificity"]





################################################################################
# MODEL 7: SUPPORT VECTOR MACHINE (SVM) - TAKEN OUT DUE TO INCOMPATIBILITY 
# WITH WEIGHTS APPROACH - would need an around-engineering, not feasible timewise
################################################################################

# cat("\n=== MODEL 7: SUPPORT VECTOR MACHINE (SVM) ===\n")
# 
# # Train SVM with cross-validation (using scaled data!)
# start_time <- Sys.time()
# set.seed(123)
# 
# svm_grid <- expand.grid(
#   sigma = c(0.002, 0.005, 0.01),
#   C = c(1, 2, 4)
# )
# 
# # Use radial basis function (RBF) kernel
# svm_model <- train(
#   formula,
#   data = train_data_scaled,  # Use scaled data!
#   method = "svmRadial",
#   trControl = ctrl,
#   metric = "ROC",
#   verbose = FALSE,
#   tuneGrid = svm_grid
# )
# 
# end_time <- Sys.time()
# svm_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
# 
# cat("Training complete!\n")
# print(svm_model)
# 
# # Predictions (use scaled test data!)
# svm_pred_train <- predict(svm_model, train_data_scaled)
# svm_pred_train_prob <- predict(svm_model, train_data_scaled, type = "prob")
# svm_pred_test <- predict(svm_model, test_data_scaled)
# svm_pred_test_prob <- predict(svm_model, test_data_scaled, type = "prob")
# 
# # Evaluate
# svm_cm_test <- confusionMatrix(svm_pred_test, test_data_scaled[[target_var]], positive = "Yes")
# cat("\nTest Set Performance:\n")
# print(svm_cm_test)
# 
# svm_roc_test <- roc(test_data_scaled[[target_var]], svm_pred_test_prob[, 2], levels = c("No", "Yes"))
# svm_auc_test <- auc(svm_roc_test)
# cat(sprintf("\nTest Set AUC: %.4f\n", svm_auc_test))
# 
# # Store results
# model_results <- rbind(model_results, data.frame(
#   Model = "SVM (RBF)",
#   AUC_Train = max(svm_model$results$ROC),
#   Accuracy_Test = svm_cm_test$overall["Accuracy"],
#   Sensitivity_Test = svm_cm_test$byClass["Sensitivity"],
#   Specificity_Test = svm_cm_test$byClass["Specificity"],
#   AUC_Test = svm_auc_test,
#   Training_Time = svm_time
# ))

################################################################################
# SECTION 6: MODEL COMPARISON AND SELECTION 
################################################################################

cat("\n\n########## MODEL COMPARISON ##########\n\n")

# Display results table
cat("=== MODEL PERFORMANCE COMPARISON ===\n")
print(model_results)

# Sort by relevant metrics

cat("\nModels ranked by Test AUC:\n")
print(model_results[order(-model_results$AUC_Test), ])

cat("\nModels ranked by Test Cost (USD):\n")
print(model_results[order(model_results$Cost_Test_USD), ])


# Visualize model comparison
par(mfrow = c(2,2))

# # 1. Test Accuracy Comparison
# barplot(model_results$Accuracy_Test,
#         names.arg = model_results$Model,
#         las = 2,
#         col = rainbow(nrow(model_results)),
#         main = "Test Set Accuracy by Model",
#         ylab = "Accuracy",
#         ylim = c(0, 1))

# 1. Test AUC Comparison
barplot(as.numeric(model_results$AUC_Test),
        names.arg = model_results$Model,
        las = 2,
        col = "darkorange",
        main = "Test Set AUC by Model",
        ylab = "AUC",
        ylim = c(0, 1))

# # 2. Test Cost Comparison
barplot(model_results$Cost_Test_USD,
        names.arg = model_results$Model,
        las = 2,
        col = "steelblue",
        main = "Test Set Cost by Model (USD)",
        ylab = "")
mtext(
  "Cost (USD)",
  side = 2,
  line = 4    # <-- größer = weiter nach links
)

# # 3. Training Time Comparison
# barplot(model_results$Training_Time,
#         names.arg = model_results$Model,
#         las = 2,
#         col = rainbow(nrow(model_results)),
#         main = "Training Time by Model",
#         ylab = "Time (seconds)")

# ROC Curves Comparison
par(mfrow = c(1, 1))

plot(log_roc_test, col = 1, main = "ROC Curves Comparison", lwd = 2, legacy.axes = TRUE)
plot(lasso_roc_test, col = 2, add = TRUE, lwd = 2)
plot(ridge_roc_test, col = 3, add = TRUE, lwd = 2)
plot(tree_roc_test, col = 4, add = TRUE, lwd = 2)
plot(rf_roc_test, col = 5, add = TRUE, lwd = 2)
plot(gbm_roc_test, col = 6, add = TRUE, lwd = 2)
# plot(svm_roc_test, col = 7, add = TRUE, lwd = 2)

legend("bottomright", 
       legend = model_results$Model,
       col = 1:6, lwd = 2)

# ROC Curves Comparison GBM weighted vs unweighted
par(mfrow = c(1, 1))

plot(gbm_roc_test, col = 1, main = "ROC Curves Comparison", lwd = 2, legacy.axes = TRUE)
plot(gbm_unweighted_roc_test, col = 2, add = TRUE, lwd = 2)

legend("bottomright", 
       legend = c("Gradient Boosting", "Gradient Boosting unweighted"),
       col = 1:2, lwd = 2)

# Select best model based on costs test

# best_model_idx <- which.min(model_results$Cost_Test_USD)
# best_model_name <- model_results$Model[best_model_idx]
# cat(sprintf("\n=== BEST MODEL: %s ===\n", best_model_name))
# # cat(sprintf("Test Accuracy: %.4f\n", model_results$Accuracy_Test[best_model_idx]))
# cat(sprintf("Test Costs: %.4f\n", model_results$Cost_Test_USD[best_model_idx]))
# cat(sprintf("Test Sensitivity: %.4f\n", model_results$Sensitivity_Test_OptThr[best_model_idx]))
# cat(sprintf("Test Specificity: %.4f\n", model_results$Specificity_Test_OptThr[best_model_idx]))
# cat(sprintf("Test AUC: %.4f\n", model_results$AUC_Test[best_model_idx]))

best_model_idx <- which.min(model_results$Cost_Test_USD)
best_model_name <- model_results$Model[best_model_idx]

cat(sprintf("\n=== COST-OPTIMAL MODEL: %s ===\n", best_model_name))
cat(sprintf("Test AUC: %.4f\n", model_results$AUC_Test[best_model_idx]))
cat(sprintf("Test Cost (USD): %.2f\n", model_results$Cost_Test_USD[best_model_idx]))
cat(sprintf("Chosen Threshold: %.2f\n", model_results$Best_Threshold[best_model_idx]))



################################################################################
# SECTION 7: FINAL PREDICTIONS ON NEW DATA 
################################################################################

cat("\n\n########## FINAL PREDICTIONS ##########\n\n")

cat("=== GENERATING PREDICTIONS FOR NEW DATA ===\n")

# For your final project, you'll load your actual test data here

# Store true labels for evaluation (before removing DEFAULT)
y_real <- data_real[[target_var]]

# For this template, we'll use our test_data as a demonstration
new_data <- data_real

# get DEFAULT out
new_data <- new_data %>% 
  select(-DEFAULT)

# Kopie erstellen
new_data_scaled <- new_data

# Nur numerische Prädiktoren skalieren – mit preproc aus TRAIN
new_data_scaled[, numeric_predictors] <- predict(preproc,
                                                 new_data[, numeric_predictors])


# Apply the same preprocessing steps you used on training data!
# 1. Handle missing values (same imputation)
# 2. Feature engineering (same transformations)
# 3. Scaling (if needed, using training parameters)

# Make predictions using the best model
# Adjust the model object name based on your best model
# Options: log_model, lasso_model, tree_model, rf_model, gbm_model, svm_model

cat(sprintf("Using %s for final predictions...\n", best_model_name))

# Select the best model object
if(best_model_name == "Logistic Regression") {
  final_model <- log_model
  final_data <- new_data_scaled
} else if(best_model_name == "Lasso Regression") {
  final_model <- lasso_model
  final_data <- new_data_scaled
} else if(best_model_name == "Ridge Regression") {
  final_model <- ridge_model
  final_data <- new_data_scaled
} else if(best_model_name == "Decision Tree") {
  final_model <- tree_model
  final_data <- new_data
} else if(best_model_name == "Random Forest") {
  final_model <- rf_model
  final_data <- new_data
} else if(best_model_name == "Gradient Boosting") {
  final_model <- gbm_model
  final_data <- new_data
} else if(best_model_name == "Gradient Boosting unweighted") {
  final_model <- gbm_unweighted_model
  final_data <- new_data
} else if(best_model_name == "SVM (RBF)") {
  final_model <- svm_model
  final_data <- new_data_scaled  # SVM needs scaled data!
}

# 1) Wahrscheinlichkeiten holen
final_predictions_prob <- predict(final_model, final_data, type = "prob")
final_prob_yes <- final_predictions_prob[, "Yes"]

# 2) Kostenoptimalen Threshold holen
final_threshold <- model_results$Best_Threshold[
  model_results$Model == best_model_name
]

# Safety check
stopifnot(length(final_threshold) == 1)

# 3) Klassifikation mit DEINEM Threshold
final_predictions <- factor(
  ifelse(final_prob_yes >= final_threshold, "Yes", "No"),
  levels = c("No","Yes")
)

# Create output dataframe
# Include any ID variable if present in your data
output_predictions <- data.frame(
  # ID = new_data$ID,  # Include if you have an ID column
  Predicted_Class = final_predictions,
  Probability_No = final_predictions_prob[, 1],
  Probability_Yes = final_predictions_prob[, 2]
)

# Display first few predictions
cat("\nFirst 10 predictions:\n")
print(head(output_predictions, 10))

# Summary of predictions
cat("\nPrediction Summary:\n")
print(table(output_predictions$Predicted_Class))
cat("\nPrediction Proportions:\n")
print(prop.table(table(output_predictions$Predicted_Class)))

### performance with "real data" on cost minimization

# Confusion matrix with stored label

final_cm <- confusionMatrix(
  final_predictions,
  y_real,
  positive = "Yes"
)

print(final_cm)

# Final cost on new data

final_cost <- compute_cost_from_pred(
  pred   = final_predictions,
  y_true = y_real,
  C_FN   = C_FN,
  C_FP   = C_FP
)

cat(sprintf(
  "\nFINAL DEPLOYMENT COST on new data (USD): %.2f\n",
  final_cost
))

# Final AUC

final_roc <- roc(
  y_real,
  final_prob_yes,
  levels = c("No", "Yes")
)

final_auc <- auc(final_roc)

cat(sprintf("Final AUC on new data: %.4f\n", final_auc))

# Output Final Summary

final_summary <- data.frame(
  Model        = best_model_name,
  Threshold    = final_threshold,
  AUC          = final_auc,
  Accuracy     = final_cm$overall["Accuracy"],
  Sensitivity  = final_cm$byClass["Sensitivity"],
  Specificity  = final_cm$byClass["Specificity"],
  Cost_USD     = final_cost
)

print(final_summary)


### Cost vs Threshold Plots ####################################################

########### Helper COST CURVES FUNCTION

make_cost_curves <- function(prob_train_yes, y_train, prob_test_yes, y_test,
                             C_FN, C_FP,
                             thresholds = seq(0.01, 0.99, by = 0.01)) {
  thr_train <- threshold_search_cost(prob_train_yes, y_train, C_FN, C_FP, thresholds)
  thr_test  <- threshold_search_cost(prob_test_yes,  y_test,  C_FN, C_FP, thresholds)
  
  curve_train <- thr_train$curve
  curve_train$Split <- "Train"
  
  curve_test <- thr_test$curve
  curve_test$Split <- "Test"
  
  curve <- rbind(curve_train, curve_test)
  
  list(
    best_train = thr_train$best,
    best_test  = thr_test$best,
    curve      = curve
  )
}

make_cost_curves <- function(prob_train_yes, y_train, prob_test_yes, y_test,
                             C_FN, C_FP,
                             thresholds = seq(0.01, 0.99, by = 0.01),
                             scale_to = 1e6) {
  
  thr_train <- threshold_search_cost(prob_train_yes, y_train, C_FN, C_FP, thresholds)
  thr_test  <- threshold_search_cost(prob_test_yes,  y_test,  C_FN, C_FP, thresholds)
  
  n_train <- length(y_train)
  n_test  <- length(y_test)
  
  # Curves
  curve_train <- thr_train$curve
  curve_train$Split <- "Train"
  curve_train$cost  <- curve_train$cost * (scale_to / n_train)
  
  curve_test <- thr_test$curve
  curve_test$Split <- "Test"
  curve_test$cost  <- curve_test$cost * (scale_to / n_test)
  
  curve <- rbind(curve_train, curve_test)
  
  # Best points (optional but nice for tables/annotations)
  best_train <- thr_train$best
  best_train$cost <- best_train$cost * (scale_to / n_train)
  
  best_test <- thr_test$best
  best_test$cost <- best_test$cost * (scale_to / n_test)
  
  list(
    best_train = best_train,
    best_test  = best_test,
    curve      = curve
  )
}


### RF cost vs threshold

rf_curves <- make_cost_curves(
  prob_train_yes = rf_pred_train_prob[, "Yes"],
  y_train        = train_data[[target_var]],
  prob_test_yes  = rf_pred_test_prob[, "Yes"],
  y_test         = test_data[[target_var]],
  C_FN = C_FN, C_FP = C_FP
)

ggplot(rf_curves$curve, aes(x = threshold, y = cost, linetype = Split)) +
  geom_line(linewidth = 0.8) +
  geom_vline(xintercept = rf_curves$best_train$threshold, linetype = "dashed") +
  geom_vline(xintercept = rf_curves$best_test$threshold,  linetype = "dotted") +
  annotate("text", x = rf_curves$best_train$threshold, y = max(rf_curves$curve$cost),
           label = sprintf("Train best: %.2f", rf_curves$best_train$threshold),
           hjust = -0.1, vjust = 1.2, size = 3) +
  annotate("text", x = rf_curves$best_test$threshold, y = max(rf_curves$curve$cost),
           label = sprintf("Test best: %.2f", rf_curves$best_test$threshold),
           hjust = -0.1, vjust = 2.4, size = 3) +
  labs(
    title = "Cost vs Threshold — Random Forest",
    subtitle = "Train curve (dashed best) vs Test curve (dotted best)",
    x = "Decision Threshold",
    y = "Cost (USD per 1,000,000 customers)",
    linetype = ""
  ) +
  scale_y_continuous(
    labels = scales::dollar_format(big.mark = ",", accuracy = 1)
  ) +
  theme_minimal()

### GBM cost vs threshold

gbm_curves <- make_cost_curves(
  prob_train_yes = gbm_pred_train_prob[, "Yes"],
  y_train        = train_data[[target_var]],
  prob_test_yes  = gbm_pred_test_prob[, "Yes"],
  y_test         = test_data[[target_var]],
  C_FN = C_FN, C_FP = C_FP
)

ggplot(gbm_curves$curve, aes(x = threshold, y = cost, linetype = Split)) +
  geom_line(linewidth = 0.8) +
  geom_vline(xintercept = gbm_curves$best_train$threshold, linetype = "dashed") +
  geom_vline(xintercept = gbm_curves$best_test$threshold,  linetype = "dotted") +
  annotate("text", x = gbm_curves$best_train$threshold, y = max(gbm_curves$curve$cost),
           label = sprintf("Train best: %.2f", gbm_curves$best_train$threshold),
           hjust = -0.1, vjust = 1.2, size = 3) +
  annotate("text", x = gbm_curves$best_test$threshold, y = max(gbm_curves$curve$cost),
           label = sprintf("Test best: %.2f", gbm_curves$best_test$threshold),
           hjust = -0.1, vjust = 2.4, size = 3) +
  labs(
    title = "Cost vs Threshold — GBM (weighted)",
    subtitle = "Train curve (dashed best) vs Test curve (dotted best)",
    x = "Decision Threshold",
    y = "Cost (USD per 1,000,000 customers)",
    linetype = ""
  ) +
  scale_y_continuous(
    labels = scales::dollar_format(big.mark = ",", accuracy = 1)
  ) +
  theme_minimal()

### GBM unweighted cost vs threshold

gbm_unweighted_curves <- make_cost_curves(
  prob_train_yes = gbm_unweighted_pred_train_prob[, "Yes"],
  y_train        = train_data[[target_var]],
  prob_test_yes  = gbm_unweighted_pred_test_prob[, "Yes"],
  y_test         = test_data[[target_var]],
  C_FN = C_FN, C_FP = C_FP
)

ggplot(gbm_unweighted_curves$curve, aes(x = threshold, y = cost, linetype = Split)) +
  geom_line(linewidth = 0.8) +
  geom_vline(xintercept = gbm_unweighted_curves$best_train$threshold, linetype = "dashed") +
  geom_vline(xintercept = gbm_unweighted_curves$best_test$threshold,  linetype = "dotted") +
  annotate("text", x = gbm_unweighted_curves$best_train$threshold, y = max(gbm_unweighted_curves$curve$cost),
           label = sprintf("Train best: %.2f", gbm_unweighted_curves$best_train$threshold),
           hjust = -0.1, vjust = 1.2, size = 3) +
  annotate("text", x = gbm_unweighted_curves$best_test$threshold, y = max(gbm_unweighted_curves$curve$cost),
           label = sprintf("Test best: %.2f", gbm_unweighted_curves$best_test$threshold),
           hjust = -0.1, vjust = 2.4, size = 3) +
  labs(
    title = "Cost vs Threshold — GBM (unweighted)",
    subtitle = "Train curve (dashed best) vs Test curve (dotted best)",
    x = "Decision Threshold",
    y = "Cost (USD per 1,000,000 customers)",
    linetype = ""
  ) +
  scale_y_continuous(
    labels = scales::dollar_format(big.mark = ",", accuracy = 1)
  ) +
  theme_minimal()

### log baseline cost vs threshold

log_curves <- make_cost_curves(
  prob_train_yes = log_pred_train_prob[, "Yes"],
  y_train        = train_data[[target_var]],
  prob_test_yes  = log_pred_test_prob[, "Yes"],
  y_test         = test_data[[target_var]],
  C_FN = C_FN, C_FP = C_FP
)

ggplot(log_curves$curve, aes(x = threshold, y = cost, linetype = Split)) +
  geom_line(linewidth = 0.8) +
  geom_vline(xintercept = log_curves$best_train$threshold, linetype = "dashed") +
  geom_vline(xintercept = log_curves$best_test$threshold,  linetype = "dotted") +
  annotate("text", x = log_curves$best_train$threshold, y = max(log_curves$curve$cost),
           label = sprintf("Train best: %.2f", log_curves$best_train$threshold),
           hjust = -0.1, vjust = 1.2, size = 3) +
  annotate("text", x = log_curves$best_test$threshold, y = max(log_curves$curve$cost),
           label = sprintf("Test best: %.2f", log_curves$best_test$threshold),
           hjust = -0.1, vjust = 2.4, size = 3) +
  labs(
    title = "Cost vs Threshold — Log Linear Regression",
    subtitle = "Train curve (dashed best) vs Test curve (dotted best)",
    x = "Decision Threshold",
    y = "Cost (USD per 1,000,000 customers)",
    linetype = ""
  ) +
  scale_y_continuous(
    labels = scales::dollar_format(big.mark = ",", accuracy = 1)
  ) +
  theme_minimal()


#### Bonus: Near Zero Variance Check:

nzv_info <- nearZeroVar(train_data, saveMetrics = TRUE)
nzv_info


#### Bonus: Model Comparison paper ready:

library(reshape2)

# Long format for ggplot

model_results_sorted <- model_results %>%
  arrange(desc(AUC_Test))

results_long <- model_results_sorted %>%
  select(Model, Accuracy_Test, AUC_Test, Sensitivity_Test, Specificity_Test) %>%
  melt(id.vars = "Model",
       variable.name = "Metric",
       value.name   = "Value")

ggplot(results_long, aes(x = Model, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  coord_flip() +
  ylim(0, 1) +
  labs(
    title = "Model Performance Comparison",
    y = "Score",
    x = ""
  ) +
  theme_minimal()



#### Bonus: Threshold Plot for best model:

# Beispiel: nimm best_model_name und passende ROC-Objekt
best_roc <- switch(
  best_model_name,
  "Logistic Regression" = log_roc_test,
  "Lasso Regression"    = lasso_roc_test,
  "Ridge Regression"    = ridge_roc_test,
  "Decision Tree"       = tree_roc_test,
  "Random Forest"       = rf_roc_test,
  "Gradient Boosting"   = gbm_roc_test
#  "SVM (RBF)"           = svm_roc_test
)

coords_df <- coords(best_roc,
                    x = "all",
                    ret = c("threshold", "sensitivity", "specificity"),
                    transpose = FALSE)


ggplot(coords_df, aes(x = threshold)) +
  geom_line(aes(y = sensitivity, color = "Sensitivity"), linewidth = 0.7) +
  geom_line(aes(y = specificity, color = "Specificity"), linewidth = 0.7, linetype = "dashed") +
  scale_color_manual(values = c("Sensitivity" = "blue", 
                                "Specificity" = "red")) +
  labs(
    title = paste("Sensitivity/Specificity vs Threshold -", best_model_name),
    y = "Metric Value",
    x = "Decision Threshold",
    color = "Metric"
  ) +
  theme_minimal()


########### am schluss!! AUC train anpassen oder durch max ROC ersetzen, oder löschen

## Making useful tables

model_results_relevant <- model_results %>% 
  select(Model, AUC_Test, Best_Threshold, Accuracy_Test_OptThr, Sensitivity_Test_OptThr, Specificity_Test_OptThr, Cost_Train_USD, Cost_Test_USD)
