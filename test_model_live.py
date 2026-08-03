import os
import datetime
import warnings
import json
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings('ignore')

import yfinance as yf
import pandas_ta as ta
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

print(f"TensorFlow version: {tf.__version__}")

MODEL_PATH = 'stock_lstm_model.keras'
FEATURE_SCALER_PATH = 'feature_scaler.pkl'
CLOSE_SCALER_PATH = 'close_scaler.pkl'

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file {MODEL_PATH} not found!")

model = load_model(MODEL_PATH)
scaler = joblib.load(FEATURE_SCALER_PATH)
close_scaler = joblib.load(CLOSE_SCALER_PATH)

print("✅ Model & Scalers loaded successfully.")

TARGET = 'AAPL'
TICKERS = ['AAPL', 'GOOG', 'MSFT', 'AMZN']
SEQ_LEN = 60

FEATURE_COLS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
    'RSI_14', 'MACD', 'MACD_sig', 'MACD_hist',
    'BB_upper', 'BB_lower', 'BB_width', 'ATR_14',
    'OBV', 'VWAP',
    'HL_ratio', 'OC_ratio', 'Log_Return',
    'Volatility_5', 'Volatility_20',
    'Close_lag1', 'Close_lag2', 'Close_lag3', 'Close_lag4', 'Close_lag5'
]

close_idx = FEATURE_COLS.index('Close')

TODAY = datetime.date.today()
LOOKBACK_DAYS = 250
start_date = (TODAY - datetime.timedelta(days=LOOKBACK_DAYS)).strftime('%Y-%m-%d')

print(f"⬇️ Downloading market data ({start_date} to {TODAY})...")
raw = yf.download(TICKERS, start=start_date, auto_adjust=True, progress=False)
raw.columns = ['_'.join(col).strip() for col in raw.columns]
raw.index = pd.to_datetime(raw.index)

ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
df = pd.DataFrame({col: raw[f'{TARGET}_{col}'] for col in ohlcv_cols})
df = df.ffill().bfill().sort_index()

def build_features(df_in):
    feat = df_in.copy()
    feat['SMA_20']    = ta.sma(feat['Close'], length=20)
    feat['SMA_50']    = ta.sma(feat['Close'], length=50)
    feat['EMA_12']    = ta.ema(feat['Close'], length=12)
    feat['EMA_26']    = ta.ema(feat['Close'], length=26)
    feat['RSI_14']    = ta.rsi(feat['Close'], length=14)
    macd              = ta.macd(feat['Close'], fast=12, slow=26, signal=9)
    feat['MACD']      = macd['MACD_12_26_9']
    feat['MACD_sig']  = macd['MACDs_12_26_9']
    feat['MACD_hist'] = macd['MACDh_12_26_9']
    bb                = ta.bbands(feat['Close'], length=20, std=2)
    feat['BB_upper']  = bb['BBU_20_2.0']
    feat['BB_lower']  = bb['BBL_20_2.0']
    feat['BB_mid']    = bb['BBM_20_2.0']
    feat['BB_width']  = (feat['BB_upper'] - feat['BB_lower']) / feat['BB_mid']
    feat['ATR_14']    = ta.atr(feat['High'], feat['Low'], feat['Close'], length=14)
    feat['OBV']       = ta.obv(feat['Close'], feat['Volume'])
    feat['VWAP']      = (feat['Volume'] * (feat['High'] + feat['Low'] + feat['Close']) / 3).cumsum() / feat['Volume'].cumsum()
    feat['HL_ratio']      = feat['High'] / feat['Low']
    feat['OC_ratio']      = feat['Open'] / feat['Close']
    feat['Log_Return']    = np.log(feat['Close'] / feat['Close'].shift(1))
    feat['Volatility_5']  = feat['Log_Return'].rolling(5).std()
    feat['Volatility_20'] = feat['Log_Return'].rolling(20).std()
    for lag in range(1, 6):
        feat[f'Close_lag{lag}'] = feat['Close'].shift(lag)
    feat.dropna(inplace=True)
    return feat

feat = build_features(df)
print(f"✅ Processed {len(feat)} rows of feature data.")

# 1. LIVE NEXT-DAY PREDICTION
live_window_raw    = feat[FEATURE_COLS].values[-SEQ_LEN:]
live_window_scaled = scaler.transform(live_window_raw)
live_input         = live_window_scaled[np.newaxis]

pred_scaled  = model.predict(live_input, verbose=0)[0, 0]
pred_price   = close_scaler.inverse_transform([[pred_scaled]])[0, 0]
last_close   = feat['Close'].iloc[-1]
last_date    = feat.index[-1]
change_usd   = pred_price - last_close
change_pct   = (change_usd / last_close) * 100
next_day     = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=1)[0]

print("\n" + "="*55)
print(f" 🎯 LIVE PREDICTION FOR {TARGET}")
print("="*55)
print(f" Last Trading Date : {last_date.strftime('%Y-%m-%d')}")
print(f" Last Close Price  : ${last_close:.2f}")
print(f" Prediction Date   : {next_day.strftime('%Y-%m-%d')}")
print(f" Predicted Close   : ${pred_price:.2f}")
print(f" Expected Move     : {change_usd:+.2f} USD ({change_pct:+.2f}%)")
print(f" Signal            : {'BUY 📈' if change_pct > 0 else 'SELL 📉'}")
print("="*55)

# 2. BACKTEST ON LAST 30 DAYS
BACKTEST_DAYS = 30
if len(feat) < SEQ_LEN + BACKTEST_DAYS:
    BACKTEST_DAYS = len(feat) - SEQ_LEN - 1

all_data   = feat[FEATURE_COLS].values
all_scaled = scaler.transform(all_data)

bt_actuals, bt_preds, bt_dates = [], [], []
start_idx = len(all_scaled) - BACKTEST_DAYS - 1

for i in range(BACKTEST_DAYS):
    idx = start_idx + i
    window = all_scaled[idx - SEQ_LEN + 1 : idx + 1]
    actual = feat['Close'].iloc[idx + 1]
    date   = feat.index[idx + 1]

    pred_s = model.predict(window[np.newaxis], verbose=0)[0, 0]
    pred_r = close_scaler.inverse_transform([[pred_s]])[0, 0]

    bt_actuals.append(actual)
    bt_preds.append(pred_r)
    bt_dates.append(date)

bt_actuals = np.array(bt_actuals)
bt_preds   = np.array(bt_preds)

rmse = np.sqrt(mean_squared_error(bt_actuals, bt_preds))
mae  = mean_absolute_error(bt_actuals, bt_preds)
mape = mean_absolute_percentage_error(bt_actuals, bt_preds) * 100
r2   = r2_score(bt_actuals, bt_preds)
da   = np.mean(np.sign(np.diff(bt_actuals)) == np.sign(np.diff(bt_preds))) * 100

print("\n" + "="*55)
print(f" 📊 LIVE BACKTEST RESULTS (Last {BACKTEST_DAYS} Trading Days)")
print("="*55)
print(f" RMSE                : ${rmse:.4f}")
print(f" MAE                 : ${mae:.4f}")
print(f" MAPE                : {mape:.2f}%")
print(f" R² Score            : {r2:.4f}")
print(f" Directional Acc.    : {da:.1f}%")
print("="*55)

results_summary = {
    "ticker": TARGET,
    "last_date": last_date.strftime('%Y-%m-%d'),
    "last_close": round(float(last_close), 2),
    "predicted_date": next_day.strftime('%Y-%m-%d'),
    "predicted_close": round(float(pred_price), 2),
    "change_pct": round(float(change_pct), 2),
    "signal": "BUY" if change_pct > 0 else "SELL",
    "backtest_metrics": {
        "rmse": round(float(rmse), 4),
        "mae": round(float(mae), 4),
        "mape": round(float(mape), 4),
        "r2": round(float(r2), 4),
        "directional_accuracy": round(float(da), 2)
    }
}

with open('live_local_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print("\n💾 Live test results saved to live_local_results.json")
