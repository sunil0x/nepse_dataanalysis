# 🚀 NEPSE Data Analysis

> Stock market analysis and prediction of companies listed on the Nepal Stock Exchange (NEPSE) using Python, machine learning, and deep learning.

**Repository:** [https://github.com/sunil0x/nepse_dataanalysis](https://github.com/sunil0x/nepse_dataanalysis)

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Project Steps](#project-steps)
- [Prediction Model](#prediction-model)
- [Quality Score System](#quality-score-system)
- [Portfolio Optimization](#portfolio-optimization)
- [Investment Simulation](#investment-simulation)
- [Results](#results)
- [How to Run](#how-to-run)
- [Requirements](#requirements)
- [Contributors](#contributors)

---

## 📌 Project Overview

This is an advanced Python data science project that analyzes historical stock data from the Nepal Stock Exchange (NEPSE). The project covers the full data science pipeline — from raw data collection and exploratory analysis, to deep learning-based price forecasting, dynamic stock quality scoring, Random Forest prediction, and portfolio optimization.

The goal is to understand market behavior, predict next-day closing prices, evaluate which stocks are worth investing in based on quantitative criteria, and construct an optimal portfolio from the highest-quality stocks.

---

## 🏗️ Project Structure

```
nepse_dataanalysis/
│
├── data-companywise/
│   ├── raw/                         # Raw stock data per company (CSV files)
│   └── processed_data/              # Cleaned and feature-engineered CSVs
├── notebooks/
│   └── Data Preprocessing and Feature Engineering/
│       ├── <company_name>.ipynb     # Per-company preprocessing & feature engineering
│       └── portfolio_optimization.ipynb
├── model/
│   ├── lstm_model.ipynb             # LSTM price prediction model
│   └── random_forest_model.ipynb   # Random Forest QS prediction model
├── results/
│   ├── plots/                       # Output charts and visualizations
│   └── company_scores.csv           # Aggregated quality scores per company
├── src/                             # Python source scripts
├── README.md
└── requirements.txt
```

---

## 📊 Dataset

Historical NEPSE stock data collected per company with the following features:

| Column         | Description                                      |
|----------------|--------------------------------------------------|
| `date`         | Trading date                                     |
| `open`         | Opening price of the day                         |
| `high`         | Highest price of the day                         |
| `low`          | Lowest price of the day                          |
| `close`        | Closing price of the day                         |
| `volume`       | Number of shares traded                          |
| `eps`          | Earnings Per Share (annual, mapped to each day)  |
| `pe-ratio`     | Price-to-Earnings ratio (annual)                 |
| `pb-ratio`     | Price-to-Book ratio (annual)                     |
| `roe`          | Return on Equity (annual)                        |
| `net-margin`   | Net Profit Margin (annual)                       |
| `dividend`     | Dividend per share (annual)                      |
| `debt-equity`  | Debt-to-Equity ratio (annual)                    |

> Fundamental columns (EPS, ROE, PE, etc.) are published annually by NEPSE companies and are mapped to every trading day of that fiscal year. For 2026, where annual reports are not yet published, the most recent known fundamentals (2025) are carried forward.

---

## 🔬 Project Steps

### 1️⃣ Data Collection

- Historical stock data imported from NEPSE sources
- Each company stored as a separate CSV file
- Data covers 2022–2026 daily trading records
- Fundamental data merged from annual financial reports

### 2️⃣ Data Preprocessing

- Parsing and sorting dates chronologically
- Handling missing values and outliers
- Carrying forward 2025 fundamentals for 2026 rows where annual data is unavailable
- Normalizing/scaling features for model input

### 3️⃣ Exploratory Data Analysis (EDA)

- Visualizing price trends over time
- Analyzing volume patterns
- Correlation analysis between fundamental features and quality score
- Identifying high-performing stocks across years

### 4️⃣ Feature Engineering

Key features engineered from raw data:

| Feature            | Description                                                        |
|--------------------|--------------------------------------------------------------------|
| `eps_growth`       | Year-over-year percentage change in EPS, mapped to daily rows      |
| `pe_dynamic`       | PE ratio recalculated daily: `close / eps`                         |
| `inv_pe_dynamic`   | Inverse of dynamic PE (higher = better value)                      |
| `inv_pb`           | Inverse of annual PB ratio (higher = better value)                 |
| `inv_de`           | Inverse of debt-equity ratio (higher = safer)                      |
| `return_1d`        | 1-day price return                                                 |
| `return_5d`        | 5-day price return                                                 |
| `return_20d`       | 20-day price return                                                |
| `ma_ratio`         | Ratio of 7-day MA to 21-day MA (above 1 = uptrend)                |
| `qs_lag1/2/3`      | Previous 1, 2, 3 days Quality Score                               |

> All features are shifted by 1 day to prevent data leakage — only past information is used to predict the future.

### 5️⃣ Train / Test Split

- Chronological split: train on 2022–2024, test on 2025
- This simulates a realistic forecasting scenario
- 2026 data used for live prediction only

---

## 🤖 Prediction Model

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

## ⭐ Quality Score System

Each stock is assigned a **daily Quality Score between 0 and 1** based on fundamental and market factors. Unlike a purely annual score, the Quality Score updates every trading day because the PE ratio is recalculated using the live closing price.

### Why Dynamic PE?

```
PE ratio = Close Price / EPS

Close price changes daily → PE changes daily → Quality Score changes daily
```

This means the Quality Score reflects both fundamental strength and current market valuation simultaneously.

### Feature Categories and Weights

| Category       | Weight | Features Used                              |
|----------------|--------|--------------------------------------------|
| Profitability  | 50%    | ROE, Net Margin, EPS Growth                |
| Valuation      | 30%    | Inverse Dynamic PE, Inverse PB Ratio       |
| Safety         | 20%    | Dividend, Inverse Debt-Equity Ratio        |

### Quality Score Formula

```python
quality_score = (
    scaled[profitability].mean(axis=1) * 0.50 +
    scaled[valuation].mean(axis=1)     * 0.30 +
    scaled[safety].mean(axis=1)        * 0.20
)
```

All features are scaled using MinMaxScaler fitted on 2022–2025 training data. The same scaler is applied (without refitting) when scoring 2026 data.

### Signal Thresholds

| Quality Score | Signal  | Recommendation |
|---------------|---------|----------------|
| ≥ 0.6         | 🟢 Bullish | Buy            |
| 0.4 – 0.6     | 🟡 Neutral | Hold           |
| ≤ 0.4         | 🔴 Bearish | Sell           |

### 2026 Prediction Approach

Since 2026 annual fundamentals are not yet published, the following carry-forward approach is used:

- All annual fundamentals (ROE, net margin, dividend, debt-equity, PB ratio, EPS) are carried forward from 2025
- The dynamic PE is recalculated daily using actual 2026 close prices and the 2025 EPS
- Quality Score is then computed normally using the same formula and scaler

```
2026 QS = f(2026 close price, 2025 fundamentals carried forward)
```

### Random Forest Model

A Random Forest Regressor is trained to predict the next day's Quality Score using recent QS history and market signals.

**Features used:**
- `qs_lag1`, `qs_lag2`, `qs_lag3` — past QS values
- `return_1d`, `return_5d`, `return_20d` — price momentum
- `ma_ratio` — trend indicator (MA7 / MA21)

**Training:** 2022–2024 daily data (~700 rows)
**Testing:** 2025 daily data (~225 rows)

**Evaluation Metrics:**

| Metric               | Description                                          |
|----------------------|------------------------------------------------------|
| RMSE                 | Prediction error in QS units                         |
| R²                   | How well the model explains QS variance              |
| Directional Accuracy | Did the model correctly predict QS going up or down? |

### Company Score Aggregation

After computing daily Quality Scores for each company, the average QS is calculated and saved to a central file:

```python
avg_score = df['quality_score'].mean().round(4)
signal = 'Bullish' if avg_score >= 0.6 else ('Bearish' if avg_score <= 0.4 else 'Neutral')

# Result saved to:
# results/company_scores.csv
```

**Format of `company_scores.csv`:**

| company | avg_score | signal  |
|---------|-----------|---------|
| NABIL   | 0.6821    | Bullish |
| EBL     | 0.5134    | Neutral |
| SCB     | 0.3892    | Bearish |

### Visualizations

| Plot                          | Description                                              |
|-------------------------------|----------------------------------------------------------|
| QS Trend with Signal Zones    | Daily QS with Bullish/Neutral/Bearish bands              |
| QS vs Close Price             | Dual-axis chart showing QS and price together            |
| Yearly Average QS             | Bar chart of annual average quality score                |
| Signal Count Per Year         | Stacked bar of Bullish/Neutral/Bearish days per year     |
| Correlation Heatmap           | Correlation between all features and quality score       |
| Feature Importance            | RF model feature importance ranking                      |

---

## 💼 Portfolio Optimization

### Goal

After computing Quality Scores for all NEPSE companies, the next step is to identify the best combination of stocks to hold — maximizing return while managing risk. Only stocks with a Bullish signal are considered as candidates.

### Step 1 — Filter Investment-Grade Stocks

Only stocks with a Bullish signal (avg_score ≥ 0.6) from `company_scores.csv` are considered:

```python
df_scores = pd.read_csv('results/company_scores.csv')
candidates = df_scores[df_scores['signal'] == 'Bullish']
```

### Step 2 — Compute Historical Returns

For each candidate company, calculate daily returns from historical close prices:

```python
returns = close_prices[candidates['company']].pct_change().dropna()
```

### Step 3 — Mean-Variance Optimization (Markowitz)

Use Modern Portfolio Theory to find the optimal portfolio weights:

- **Objective:** Maximize the Sharpe Ratio (return per unit of risk)
- **Constraints:** Weights sum to 1, no short selling (weights ≥ 0)
- **Inputs:** Expected returns, covariance matrix of daily returns

```python
from scipy.optimize import minimize

def neg_sharpe(weights, returns, risk_free=0.05/252):
    port_return = np.dot(weights, returns.mean()) * 252
    port_vol    = np.sqrt(weights @ returns.cov() @ weights * 252)
    return -(port_return - risk_free) / port_vol

result = minimize(neg_sharpe, x0=equal_weights, constraints=..., bounds=...)
optimal_weights = result.x
```

### Step 4 — Portfolio Metrics

| Metric            | Description                                                      |
|-------------------|------------------------------------------------------------------|
| Expected Return   | Annualized portfolio return based on historical data             |
| Portfolio Risk    | Annualized standard deviation of portfolio returns               |
| Sharpe Ratio      | Return per unit of risk (higher is better, target > 1)           |
| Max Drawdown      | Largest peak-to-trough loss over the evaluation period           |

### Step 5 — Efficient Frontier Visualization

Plot the risk-return tradeoff across thousands of random portfolio simulations, highlighting:

- The **Minimum Variance Portfolio** (lowest risk)
- The **Maximum Sharpe Portfolio** (best risk-adjusted return)
- The **Optimal Portfolio** selected by the model

### Output

```
results/
├── company_scores.csv          # Average QS and signal per company
├── portfolio_weights.csv       # Optimal allocation per company
├── efficient_frontier.png      # Risk-return scatter plot
└── portfolio_performance.png   # Cumulative return over time
```

**Format of `portfolio_weights.csv`:**

| company | weight | avg_score | signal  |
|---------|--------|-----------|---------|
| NABIL   | 0.3412 | 0.6821    | Bullish |
| EBL     | 0.2890 | 0.6340    | Bullish |
| SCB     | 0.3698 | 0.6105    | Bullish |

---

## 💰 Investment Simulation

### Buy Strategy

A trade signal is triggered when **both** conditions are met:

- Predicted price is **at least 2% higher** than the current price (from LSTM)
- Stock Quality Score is **above 0.6** (Bullish signal)

### Performance Metrics

| Metric           | Description                                                     |
|------------------|-----------------------------------------------------------------|
| Total Return     | Overall profit/loss percentage over the simulation period       |
| Sharpe Ratio     | Return per unit of risk (higher is better, > 1 is good)        |
| Maximum Drawdown | Largest peak-to-trough loss — measures downside risk            |

---

## 📈 Results

Predicted vs actual close price plots, quality score trends, and portfolio performance results are saved in `results/plots/`. Aggregated company scores are saved in `results/company_scores.csv`.

---

## 🚀 How to Run

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
| `notebooks/Data Preprocessing and Feature Engineering/<company_name>.ipynb` | Per-company data preprocessing & feature engineering |
| `notebooks/Data Preprocessing and Feature Engineering/portfolio_optimization.ipynb` | Filter Bullish stocks and optimize portfolio weights |
| `model/lstm_model.ipynb` | Train and evaluate LSTM price prediction model |
| `model/random_forest_model.ipynb` | Train RF model to predict next-day QS |

---

## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
tensorflow
keras
jupyter
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## 👥 Contributors

This project was developed as a group project for the Advanced Python course.

| Contributor | GitHub | Role |
|-------------|--------|------|
| Sunil | [@sunil0x](https://github.com/sunil0x) | Project Lead  |
| Contributor 2 |  []()) |  |
| Contributor 3 |  []()) |  |

> 📌 **Note:** The repository shows 2 contributors. If you are a co-contributor, feel free to update this section with your name and GitHub handle.

---

<p align="center">
  Made with ❤️ in Nepal | Advanced Python Project — NEPSE Data Analysis Team
</p>
