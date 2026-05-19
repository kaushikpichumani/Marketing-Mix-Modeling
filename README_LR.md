# Classical Marketing Mix Modeling (MMM) using Regression

## Overview

This project implements a classical Marketing Mix Modeling (MMM) framework using Regression-based techniques to measure the impact of marketing channels on sales and optimize media budget allocation.

The model captures important real-world marketing effects such as:

- Carryover effects using Adstock transformation
- Diminishing returns using Saturation curves
- Channel contribution analysis
- ROI estimation
- Budget optimization

The implementation uses Python, Scikit-learn, and optimization techniques to build an end-to-end MMM pipeline.

---

# Business Problem

Marketing teams invest across multiple advertising channels such as:

- Google Ads
- Facebook Ads
- TikTok
- Snapchat
- Affiliate Marketing

The challenge is to answer:

- Which channels drive the most sales?
- Which channels provide the best ROI?
- How should future marketing budgets be allocated?
- How do lag and saturation effects influence performance?

This project solves these problems using regression-based Marketing Mix Modeling.

---

# Dataset Description

The dataset contains weekly marketing spend and sales data.

## Features

| Column | Description |
|---|---|
| Date | Weekly timestamp |
| Sales | Target sales variable |
| adwords_spending | Google Ads spend |
| facebookads_spending | Facebook Ads spend |
| awin_spending | Affiliate marketing spend |
| tiktok_spending | TikTok Ads spend |
| snapchat_spending | Snapchat Ads spend |

---

# Key Concepts

## 1. Adstock Transformation

Advertising impact does not disappear immediately after spending.

Adstock models the carryover effect of advertising over time.

### Formula

:contentReference[oaicite:0]{index=0}

Where:
- `λ` = decay factor
- Higher values indicate longer-lasting impact

---

## 2. Saturation Transformation

Marketing channels experience diminishing returns.

Beyond a certain spend level, additional investment produces lower incremental gains.

### Formula

:contentReference[oaicite:1]{index=1}

Where:
- `α` controls curve steepness
- `θ` controls saturation midpoint

---

# Project Workflow

```text
Raw Media Spend
        ↓
Adstock Transformation
        ↓
Saturation Transformation
        ↓
Feature Engineering
        ↓
Regression Model Training
        ↓
Prediction & Evaluation
        ↓
ROI Analysis
        ↓
Budget Optimization
```

---

# Tech Stack

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| Pandas | Data Processing |
| NumPy | Numerical Computation |
| Scikit-learn | Regression Modeling |
| SciPy | Optimization |
| Matplotlib | Visualization |
| Seaborn | Statistical Visualization |

---

# Project Structure

```text
project/
│
├── data/
│   └── data.csv
│
├── notebooks/
│   └── mmm.ipynb
│
├── models/
│   └── ridge_mmm_model.pkl
│
├── outputs/
│   ├── plots/
│   └── reports/
│
├── README.md
└── requirements.txt
```

---

# Installation

## Clone Repository

```bash
git clone <repository-url>
cd classical-mmm-regression
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy joblib
```

---

# Data Preprocessing

The following preprocessing steps were applied:

- Date conversion
- Missing value handling
- Black Friday feature engineering
- Adstock transformation
- Saturation transformation
- Train-test split
- Feature scaling

---

# Model Training

A Ridge Regression model was trained using transformed media variables.

## Why Ridge Regression?

Ridge Regression helps:

- Reduce overfitting
- Handle multicollinearity
- Stabilize channel coefficients

---

# Model Evaluation

The model was evaluated using:

- RMSE (Root Mean Squared Error)
- R² Score
- Actual vs Predicted Sales comparison

---

# Channel Contribution Analysis

Channel contribution was estimated using regression coefficients.

The model helps identify:

- Most impactful channels
- Contribution trends over time
- Relative marketing effectiveness

---

# ROI Analysis

ROI was calculated as:

:contentReference[oaicite:2]{index=2}

Channels with higher ROI indicate better profitability.

---

# Budget Optimization

An optimization engine was implemented using `scipy.optimize`.

## Objective

Find the optimal spend allocation across marketing channels under a fixed budget constraint.

### Constraint

:contentReference[oaicite:3]{index=3}

The optimizer:
- Tests multiple spend combinations
- Predicts resulting sales
- Maximizes expected sales

---

# Example Optimization Output

| Channel | Optimal Spend |
|---|---|
| Google Ads | 250,000 |
| Facebook Ads | 180,000 |
| TikTok | 70,000 |

---

# Model Persistence

The trained model and scaler were saved using `joblib`.

## Save Model

```python
joblib.dump(model, 'ridge_mmm_model.pkl')
```

## Load Model

```python
loaded_model = joblib.load('ridge_mmm_model.pkl')
```

---

# Visualizations Included

- Correlation heatmap
- Sales trend over time
- Actual vs Predicted sales
- Channel contribution trends
- ROI comparison charts
- Feature importance plots

---

# Key Learnings

This project demonstrates:

- Classical Marketing Mix Modeling
- Media attribution analysis
- Lag and saturation modeling
- Regression-based forecasting
- ROI estimation
- Marketing budget optimization

---

# Future Improvements

Potential enhancements:

- Bayesian MMM implementation
- Time-series cross-validation
- SHAP explainability
- Hyperparameter tuning
- Geo-level MMM
- Multi-touch attribution integration
- Automated retraining pipeline

---
