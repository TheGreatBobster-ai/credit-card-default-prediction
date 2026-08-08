# Cost-Sensitive Credit Card Default Prediction

My first machine-learning project: an end-to-end classification pipeline for predicting credit card default using six supervised learning algorithms, with model selection based not only on predictive performance but also on **asymmetric business costs and cost-sensitive threshold optimization**.

> **Key Result:** Gradient Boosting achieved the lowest expected business cost on the unseen test set. For a hypothetical portfolio of one million customers, the model reduces expected costs from **$202.18M without customer selection to $102.47M — a reduction of approximately 49%**. It also reduces costs by **7.23% ($8.0M) relative to Logistic Regression**, despite Random Forest achieving a slightly higher ROC-AUC.

**Key Overview:** For a hypothetical portfolio of **one million customers** 2005 Taiwan, all models substantially reduce expected costs relative to accepting every applicant. Gradient Boosting produces the largest economic benefit.

| Model | Cost Reduction | Additional Reduction vs. Logistic Regression |
|---|---:|---:|
| Logistic Regression | $91.72M | — |
| Lasso Regression | $91.97M | $0.27M |
| Ridge Regression | $91.88M | $0.17M |
| Decision Tree | $96.83M | $5.11M |
| Random Forest | $92.84M | $1.12M |
| **Gradient Boosting** | **$99.71M** | **$7.99M** |

*Expected cost reductions per one million customers under the project's business-cost assumptions.*

---

## Project Overview

Credit card default prediction is a binary classification problem with highly asymmetric economic consequences. Missing a future default can lead to substantial financial losses, whereas incorrectly rejecting a creditworthy customer primarily results in lost business opportunities.

This creates an important distinction between **predictive accuracy and economic performance**: the model with the highest ROC-AUC is not necessarily the model that leads to the best lending decisions.

This project therefore compares six machine-learning models under both conventional classification metrics and an explicit business-cost framework. Classification thresholds are optimized based on expected economic costs rather than using the conventional 0.5 cutoff.

The final models are evaluated on an **unseen holdout dataset** that was not used for model or threshold selection.

---

## Models

- Logistic Regression
- Lasso Regression
- Ridge Regression
- Decision Tree
- Random Forest
- Gradient Boosting

---

## Workflow

- Data cleaning and preprocessing
- Exploratory data analysis
- Feature engineering
- Stratified train / validation / holdout split
- Handling class imbalance through class weighting
- Hyperparameter tuning using cross-validation
- Model evaluation (ROC-AUC, Accuracy, Sensitivity, Specificity)
- Cost-sensitive threshold optimization
- Final evaluation on an unseen holdout dataset

---

## Feature Engineering

Several additional features were created to better capture repayment behaviour, including:

- Payment-to-bill ratios
- Total bill amount
- Total payment amount
- Maximum historical payment delay
- Bill trend over time
- Payment variability

---

## Business Impact

The cost-sensitive evaluation illustrates why predictive performance alone is insufficient for model selection.

Assuming a hypothetical portfolio of **one million customers**, expected costs on the unseen test set are:

| Model | Test AUC | Optimal Threshold | Expected Cost | Cost Reduction vs. No Selection | Improvement vs. Logistic Regression |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.748 | 0.53 | $110.47M | 45% | — |
| Lasso Regression | 0.748 | 0.53 | $110.20M | 45% | 0.23% |
| Ridge Regression | 0.747 | 0.52 | $110.30M | 45% | 0.15% |
| Decision Tree | 0.767 | 0.43 | $105.36M | 48% | 4.63% |
| Random Forest | **0.795** | 0.38 | $109.34M | 46% | 1.02% |
| **Gradient Boosting** | 0.793 | 0.45 | **$102.47M** | **49%** | **7.23%** |

Without customer selection, expected costs are approximately **$202.18M**.

The comparison demonstrates the central result of the project: **the model with the highest ROC-AUC is not necessarily the model with the lowest economic cost**. Random Forest achieves the highest test AUC (0.795), but Gradient Boosting produces substantially lower expected costs after threshold optimization.

Compared with the Logistic Regression baseline, Gradient Boosting reduces expected costs by approximately **$8.0M per one million customers**, corresponding to an additional **7.23% cost reduction**.

> Costs represent expected economic consequences of a single hypothetical credit-approval round under the assumptions defined in the project and should not be interpreted as realized historical savings.

---

## Key Findings

- **Gradient Boosting achieved the lowest expected test-set cost ($102.47M)**, reducing expected costs by approximately **49% compared with accepting all customers** under the assumed business-cost framework.
- **The highest-AUC model was not the economically optimal model:** Random Forest achieved the highest test AUC (0.795), while Gradient Boosting (0.793) produced substantially lower expected costs.
- Cost-sensitive threshold optimization improved decision-making beyond the conventional 0.5 classification threshold.
- Compared with Logistic Regression, Gradient Boosting reduced expected costs by approximately **$8.0M per one million customers (7.23%)**.
- Threshold stability provided additional information about model generalization.
- Recent payment behaviour was substantially more informative than demographic variables.

---

## Results

### ROC Comparison

![ROC Comparison](images/roc-comparison.png)

### Cost-Sensitive Threshold Optimization

![Threshold Optimization](images/threshold-optimization-weighted.png)

### Cost-Sensitive Threshold Optimization - Unweighted Comparison

![Threshold Optimization](images/threshold-optimization-unweighted.png)

### Variable Importance

![Variable Importance](images/variable-importance.png)



## Full Report

A detailed description of the methodology, experiments, results, and discussion is available in:

📄 **[credit_card_default_report.pdf](report/credit_card_default_report.pdf)**

---


## Repository Structure

```
.
├── data/
│   └── data_creditcard.xlsx
│
├── R/
│   └── main.r
│
├── report/
│   └── credit_card_default_report.pdf
│
├── .gitignore/
├── license/
└── README.md
```

---

## Dataset

The project uses the **Default of Credit Card Clients** dataset introduced by Yeh & Lien (2009), containing information on 30,000 Taiwanese credit card clients.

Reference:

> Yeh, I.-C., & Lien, C.-H. (2009). *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients*. Expert Systems with Applications.

---


## Technologies

- R
- caret
- glmnet
- ranger
- gbm
- ggplot2
- pROC
- tidyverseg
- readxl

## Lessons Learned

This project highlighted that optimizing purely for ROC-AUC does not necessarily lead to economically optimal decisions. Incorporating business costs into threshold selection can substantially improve real-world decision making despite similar predictive performance.

---

## Author

Robert Puselja