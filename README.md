# Marketing Mix Modeling (MMM) using LightweightMMM

## Overview

This project implements a complete Marketing Mix Modeling (MMM) pipeline using the `LightweightMMM` library. The goal is to measure the contribution of different marketing channels toward sales and optimize media spending for maximum return on investment (ROI).

The dataset used in this project is synthetically generated to emulate real-world marketing campaign data from an online retailer.

The project covers:

- Data exploration and preprocessing
- Bayesian Marketing Mix Modeling
- Media contribution analysis
- ROI estimation
- Prediction evaluation
- Budget optimization
- Model persistence and reuse

---

# Business Objective

The primary objective of this project is to:

- Understand how different marketing channels contribute to sales
- Measure channel effectiveness and ROI
- Capture marketing effects such as:
  - Saturation
  - Carryover effects
  - Lagged effects
  - Seasonality
  - Trends
- Optimize marketing spend allocation across channels

---

# Dataset Description

The dataset contains weekly marketing and sales data spanning three years.

## Features

| Column | Description |
|---|---|
| Date | Week starting date |
| Sales | Total sales amount (Target Variable) |
| adwords_spending | Google AdWords spend |
| facebookads_spending | Facebook Ads spend |
| awin_spending | Awin affiliate marketing spend |
| tiktok_spending | TikTok advertising spend |
| snapchat_spending | Snapchat advertising spend |

---

# Project Workflow

## Task 1 — Import Required Libraries

The following libraries were used:

```python
import numpy as np
import pandas as pd
import jax.numpy as jnp
import numpyro

from lightweight_mmm import lightweight_mmm
from lightweight_mmm import optimize_media
from lightweight_mmm import plot
from lightweight_mmm import preprocessing
from lightweight_mmm import utils

import matplotlib.pyplot as plt
import seaborn as sns
```

---

# Task 2 — Load the Dataset

The dataset was loaded into a Pandas DataFrame.

```python
df = pd.read_csv("data.csv")
df.head()
```

---

# Task 3 — Exploratory Data Analysis (EDA)

Basic validation and exploratory checks were performed:

## Checks Performed

- Missing value detection
- Outlier inspection
- Dataset statistical summary
- Correlation analysis
- Sales trend visualization

## Correlation Matrix

A correlation matrix was generated to understand relationships between media channels and sales.

```python
corr_matrix = df.corr(numeric_only=True)
```

## Heatmap Visualization

```python
sns.heatmap(corr_matrix, annot=True)
```

## Sales Trend Over Time

```python
plt.plot(df["Date"], df["Sales"])
```

---

# Task 4 — Data Preprocessing

The dataset was prepared for modeling using the following steps.

## Preprocessing Steps

### 1. Convert Date Column

```python
df["Date"] = pd.to_datetime(df["Date"])
```

### 2. Create Black Friday Feature

A binary feature was added to identify Black Friday weeks.

```python
df["black_friday"] = ...
```

### 3. Handle Missing Values

```python
df.fillna(0, inplace=True)
```

### 4. Train-Test Split

- 80% Training Data
- 20% Testing Data

### 5. Scaling

Custom scaling was applied using the mean value of each variable.

```python
scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
```

---

# Task 5 — Train the Marketing Mix Model

The MMM model was trained using `LightweightMMM`.

## Model Type

The following model types are supported:

- `carryover`
- `adstock`
- `hill_adstock`

This project used:

```python
mmm = lightweight_mmm.LightweightMMM(model_name="hill_adstock")
```

---

# Model Features Captured

The model accounts for:

- Trend
- Seasonality
- Lagged effects
- Saturation effects
- Media contribution
- External variables

---

# Model Training

```python
mmm.fit(
    media=media_data,
    extra_features=extra_features,
    media_prior=costs,
    target=target,
    number_warmup=1000,
    number_samples=1000,
    number_chains=2
)
```

## Important Parameters

| Parameter | Description |
|---|---|
| media | Marketing channel spending |
| extra_features | Additional variables |
| media_prior | Channel costs |
| target | Sales variable |
| number_warmup | Warm-up iterations |
| number_samples | Posterior samples |
| number_chains | Parallel MCMC chains |

---

# Task 6 — Convergence Check

Model convergence was evaluated using:

- Number of divergences
- Stability of sampling chains

```python
mmm.print_summary()
```

## Expected Result

- Ideally: `0 divergences`
- Acceptable: `< 30 divergences`

---

# Task 7 — Model Evaluation

Predictions were generated using the trained model.

## Prediction Workflow

```python
predictions = mmm.predict(...)
mean_predictions = predictions.mean(axis=0)
```

## RMSE Calculation

Root Mean Square Error (RMSE) was used to evaluate model accuracy.

```python
rmse = np.sqrt(np.mean((actual - predicted) ** 2))
```

---

# Task 8 — Prediction Quality Analysis

Predictions were compared against actual sales values.

## Analysis Included

- Mean predictions
- Prediction uncertainty intervals
- Standard deviation bands
- Actual vs predicted visualization

## Visualization

Plots included:

- Actual sales
- Predicted sales
- Confidence intervals (`±2 std deviation`)

---

# Task 9 — Parameter Estimation Analysis

Posterior distributions were analyzed to understand parameter learning.

## Visualizations

### Prior vs Posterior Distributions

This helps assess:

- Impact of prior assumptions
- Learning from observed data

### Baseline Media Contributions

This shows:

- Contribution of each marketing channel
- Contribution changes over time

---

# Task 10 — Media Effect & ROI Analysis

Media effectiveness and ROI were evaluated.

## Metrics Computed

### Media Effect

Represents the percentage contribution of each channel toward sales.

### ROI

Measures profitability of each marketing channel.

```python
media_effect = ...
roi = ...
```

## Insights

Channels with:

- Higher media effect
- Higher ROI

should receive more investment.

---

# Task 11 — Media Budget Optimization

Marketing spend allocation was optimized under a fixed budget constraint.

## Optimization Inputs

- Budget
- Time horizon
- Historical performance
- Channel constraints

## Optimization Goal

Maximize expected sales return.

```python
solution = optimize_media.find_optimal_budgets(...)
```

---

# Task 12 — Model Persistence

The trained model was saved and reloaded for future use.

## Save Model

```python
utils.save_model(mmm, "mmm_model")
```

## Load Model

```python
loaded_model = utils.load_model("mmm_model")
```

## Validate Reloaded Model

Predictions were generated again to confirm successful loading.

---

# Key Concepts in MMM

## Saturation

Increasing spend eventually produces diminishing returns.

## Lag Effect

Marketing impact may continue across multiple future periods.

## Carryover Effect

Advertising impact persists after the campaign ends.

## Seasonality

Recurring patterns such as holidays or festive periods.

---

# Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| JAX | Accelerated numerical computing |
| NumPyro | Bayesian inference |
| LightweightMMM | Marketing mix modeling |
| Matplotlib | Visualization |
| Seaborn | Statistical visualization |

---

# Example Project Structure

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
│   └── saved_model/
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
cd marketing-mix-modeling
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run the Notebook

```bash
jupyter notebook
```

Open:

```text
mmm.ipynb
```

---

# Key Learnings

This project demonstrates:

- Bayesian Marketing Mix Modeling
- Media attribution
- ROI estimation
- Marketing budget optimization
- Probabilistic forecasting
- Uncertainty quantification

---

# Future Improvements

Potential enhancements include:

- Geo-level MMM
- Multi-touch attribution integration
- Hyperparameter tuning
- Automated retraining pipeline
- Real-time dashboard integration
- Bayesian hierarchical models

---

# References

- LightweightMMM Documentation
- Google Bayesian MMM Research
- NumPyro Documentation
- JAX Documentation

