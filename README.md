## 🚀 Team Project: NEPSE Data Analysis

📊 Stock market analysis of Nepal (NEPSE) using Python and ML.
A simple data science project focused on analyzing and predicting stocks listed on the Nepal Stock Exchange (NEPSE).

🔗 Project Repository:
https://github.com/sunil0x/nepse_dataanalysis

---

## 📌 Project Overview
This project uses historical NEPSE data to understand market behavior and forecast future prices.  
It combines deep learning prediction with a scoring system to evaluate stock quality and simulate investment decisions.

---

## 🏗️ Project Steps

### 1️⃣ Data Collection
- Historical stock data imported/scraped
- Features used:
  - Open
  - High
  - Low
  - Close
  - Volume

### 2️⃣ Data Analysis
- Visualizing trends
- Understanding price movement and volatility
- Basic statistical exploration

### 3️⃣ Prediction Model (LSTM)
- Deep learning model for time-series forecasting
- Uses last **60 days** of data
- Predicts **next day closing price**

**Evaluation Metrics**
- RMSE
- MAPE

---

## 📊 Quality Score
Stocks are rated from **0–100** using fundamental factors:

| Factor | Weight |
|-------|--------|
| ROE | 30% |
| P/E Ratio | 20% |
| Price Momentum | 25% |
| Volatility | 25% |

Higher score indicates stronger overall stock quality.

---

## 💰 Investment Simulation

### Strategy
Buy when:
- Predicted price is **2% higher** than current price
- Quality Score is **above 70**

### Metrics Measured
- Total Return
- Sharpe Ratio
- Maximum Drawdown

---

## 🚀 How to Run

```bash
git clone https://github.com/yourusername/nepse-analysis.git
pip install -r requirements.txt
