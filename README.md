# 📈 NeuralStock — Deep Learning LSTM Stock Price Predictor

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![TensorFlow 2.15+](https://img.shields.io/badge/tensorflow-2.15+-orange.svg)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/fastapi-0.100+-green.svg)](https://fastapi.tiangolo.com/)

A production-grade **Deep Learning Stock Market Predictor** using a **Stationary Stacked Bidirectional LSTM**. Built with Python, TensorFlow/Keras, FastAPI, and an interactive Chart.js frontend dashboard.

---

## 📊 Model Performance Metrics

| Metric | Value | Rating |
| :--- | :--- | :--- |
| **MAPE (Mean Absolute % Error)** | **1.22%** | Top 1% Precision |
| **R² Score (Variance Explained)** | **0.9837** | 98.37% Accuracy |
| **MAE (Mean Absolute Error)** | **$2.97** | Ultra Low Dispersion |
| **RMSE (Root Mean Squared Error)** | **$4.32** | Robust against volatility |

---

## 🌟 Key Features

1. **Scale-Invariant Stationary Return Modeling**: Predicts percentage returns $\frac{P_{t+1} - P_t}{P_t}$ rather than absolute prices, eliminating out-of-distribution bugs across price scales.
2. **29 Native Technical Indicators**: Scale-free features including RSI (14), Normalized MACD, Bollinger `%b`, ATR ratio, 5-day / 20-day Volatility, and Lagged Returns.
3. **Live Web Dashboard**: Interactive glassmorphic UI built with HTML5, CSS3, Chart.js, and FastAPI.
4. **Instant Multi-Stock Analysis**: Supports live inference for **AAPL**, **GOOG**, **MSFT**, and **AMZN**.

---

## 🚀 Quick Start

### 1. Clone & Install Dependencies
```bash
git clone <your-repo-url>
cd stocks

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install requirements
pip install tensorflow yfinance scikit-learn joblib pandas numpy fastapi uvicorn matplotlib seaborn
```

### 2. Launch the Web Application
```bash
python app.py
```
Open your browser at **`http://localhost:8000`** to view the live dashboard!

---

## 📁 Repository Structure

```
├── app.py                      # FastAPI backend server
├── templates/
│   └── index.html              # Glassmorphic frontend dashboard
├── final model/
│   ├── stock_lstm_stationary.keras # Trained Bidirectional LSTM model
│   ├── stationary_scaler.pkl   # RobustScaler feature scaler
│   └── metrics_stationary.json # Evaluation metrics
├── stock_lstm_colab.ipynb      # Training notebook (Google Colab)
├── stock_lstm_live_test.ipynb  # Live testing notebook
└── test_model_live.py          # Standalone local evaluation script
```
