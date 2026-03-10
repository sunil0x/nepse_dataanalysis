# NEPSE Data Analysis

Stock market analysis and prediction of companies listed on the Nepal Stock Exchange (NEPSE) using Python, machine learning, and deep learning.

**Repository:** https://github.com/sunil0x/nepse_dataanalysis

---

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Project Steps](#project-steps)
- [Prediction Model](#prediction-model)
- [Quality Score System](#quality-score-system)
- [Investment Simulation](#investment-simulation)
- [Results](#results)
- [How to Run](#how-to-run)
- [Requirements](#requirements)
- [Team](#team)

---

## Project Overview

This is an advanced Python data science project that analyzes historical stock data from the Nepal Stock Exchange (NEPSE). The project covers the full data science pipeline — from raw data collection and exploratory analysis, to deep learning-based price forecasting, stock quality scoring, and investment simulation.

The goal is to understand market behavior, predict next-day closing prices, and evaluate which stocks are worth investing in based on quantitative criteria.

---

## Project Structure

```
nepse_dataanalysis/
│
├── data-companywise/        # Historical stock data per company (CSV files)
├── notebooks/               # Jupyter notebooks for each stage of analysis
├── results/
│   └── plots/               # Output charts and visualizations
├── src/                     # Python source scripts
└── README.md
```

---

## Dataset

Historical NEPSE stock data collected per company with the following features:

| Column    | Description                        |
|-----------|------------------------------------|
| `date`    | Trading date                       |
| `open`    | Opening price of the day           |
| `high`    | Highest price of the day           |
| `low`     | Lowest price of the day            |
| `close`   | Closing price of the day           |
| `volume`  | Number of shares traded            |

Data is stored company-wise inside the `data-companywise/` folder, with one CSV file per company.

---

## Project Steps

### Step 1 — Data Collection

- Historical stock data imported from NEPSE sources
- Each company stored as a separate CSV file
- Data covers multiple years of daily trading records

### Step 2 — Data Preprocessing

- Parsing and sorting dates chronologically
- Handling missing values and outliers
- Normalizing/scaling features for model input

### Step 3 — Exploratory Data Analysis (EDA)

- Visualizing price trends over time
- Analyzing volume patterns and volatility
- Correlation analysis between features
- Identifying high-performing and volatile stocks

### Step 4 — Feature Engineering

Key features engineered from raw data:

| Feature               | Description                              |
|-----------------------|------------------------------------------|
| `prev_close`          | Previous day's closing price             |
| `price_change_pct`    | Percentage change from previous close    |
| `ma_7`                | 7-day moving average                     |
| `ma_20`               | 20-day moving average                    |
| `price_momentum`      | Short-term directional trend indicator   |

All features are shifted by 1 day to prevent data leakage — only past information is used to predict the future.

### Step 5 — Train / Test Split

- Chronological 80/20 split (not random)
- Training set: first 80% of dates
- Test set: last 20% of dates
- This simulates a realistic forecasting scenario

---

## Prediction Model

### LSTM (Long Short-Term Memory)

A deep learning model designed for time-series forecasting.

- **Input:** Last 60 days of closing prices (sequence)
- **Output:** Next day's predicted closing price
- **Architecture:** LSTM layers with dropout for regularization

```
Input (60 days) → LSTM → Dropout → LSTM → Dropout → Dense → Predicted Close
```

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| RMSE   | Root Mean Squared Error — average prediction error in price units |
| MAPE   | Mean Absolute Percentage Error — average % deviation from actual price |

Lower values of both metrics indicate a better model.

---

## Quality Score System

Each stock is assigned a quality score from **0 to 100** based on fundamental and technical factors.

| Factor           | Weight | Description                                      |
|------------------|--------|--------------------------------------------------|
| ROE              | 30%    | Return on Equity — profitability indicator        |
| P/E Ratio        | 20%    | Price-to-Earnings — valuation indicator           |
| Price Momentum   | 25%    | Recent price trend strength                      |
| Volatility       | 25%    | Inverse of price volatility (lower is better)    |

A higher score indicates a fundamentally stronger and more stable stock. Scores above 70 are considered investment-grade.

---

## Investment Simulation

### Buy Strategy

A trade signal is triggered when both conditions are met:

- Predicted price is **at least 2% higher** than the current price
- Stock Quality Score is **above 70**

### Performance Metrics

| Metric           | Description                                                   |
|------------------|---------------------------------------------------------------|
| Total Return     | Overall profit/loss percentage over the simulation period     |
| Sharpe Ratio     | Return per unit of risk (higher is better, >1 is good)       |
| Maximum Drawdown | Largest peak-to-trough loss — measures downside risk          |

---

## Results

Predicted vs actual close price plots and performance results are saved in `results/plots/`.

---

## How to Run

**1. Clone the repository**

```bash
git clone https://github.com/sunil0x/nepse_dataanalysis.git
cd nepse_dataanalysis
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Run the notebooks in order**

Open Jupyter and run the notebooks inside the `notebooks/` folder sequentially:

```bash
jupyter notebook
```

| Notebook | Purpose |
|----------|---------|
| `01_data_exploration.ipynb`  | Load data, EDA, visualizations |
| `02_feature_engineering.ipynb` | Create features, handle leakage |
| `03_lstm_model.ipynb`        | Train and evaluate LSTM model |
| `04_quality_score.ipynb`     | Compute quality scores per stock |
| `05_simulation.ipynb`        | Run investment simulation |

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
tensorflow
keras
jupyter
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## Team

Advanced Python Project — NEPSE Data Analysis Team

**Repository:** https://github.com/sunil0x/nepse_dataanalysis
