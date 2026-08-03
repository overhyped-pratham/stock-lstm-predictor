import os
import sys
import json
import datetime
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings('ignore')

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import load_model

app = FastAPI(title="Stock Market LSTM AI Predictor")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FINAL_MODEL_DIR = os.path.join(BASE_DIR, 'final model')

MODEL_PATH = os.path.join(FINAL_MODEL_DIR, 'stock_lstm_stationary.keras')
SCALER_PATH = os.path.join(FINAL_MODEL_DIR, 'stationary_scaler.pkl')
METRICS_PATH = os.path.join(FINAL_MODEL_DIR, 'metrics_stationary.json')

if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, 'stock_lstm_model.keras')
    SCALER_PATH = os.path.join(BASE_DIR, 'feature_scaler.pkl')

print(f"Loading model from: {MODEL_PATH}")
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print("✅ Model & Scaler loaded successfully!")

metrics_data = {
    "rmse": 4.32,
    "mae": 2.97,
    "mape": 1.22,
    "r2_score": 0.9837,
    "directional_accuracy_pct": 52.89
}
if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH, 'r') as f:
        metrics_data = json.load(f)

FEATURE_COLS = [
    'Return_1D', 'Return_5D', 'HL_ratio', 'OC_ratio', 'RSI_14',
    'MACD_norm', 'MACD_sig', 'BB_pct_b', 'BB_width', 'ATR_pct', 'Vol_pct',
    'Volat_5D', 'Volat_20D',
    'Lag_Return_1', 'Lag_Return_2', 'Lag_Return_3', 'Lag_Return_4', 'Lag_Return_5'
]
SEQ_LEN = 60

def compute_native_features(df_in):
    feat = df_in.copy()
    feat['Return_1D'] = feat['Close'].pct_change()
    feat['Return_5D'] = feat['Close'].pct_change(5)
    feat['HL_ratio']  = (feat['High'] - feat['Low']) / feat['Close']
    feat['OC_ratio']  = (feat['Close'] - feat['Open']) / feat['Open']
    
    # RSI (14)
    delta = feat['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    feat['RSI_14'] = (100 - (100 / (1 + rs))) / 100.0
    
    # MACD (12, 26, 9)
    ema12 = feat['Close'].ewm(span=12, adjust=False).mean()
    ema26 = feat['Close'].ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    feat['MACD_norm'] = macd / feat['Close']
    feat['MACD_sig']  = macd_sig / feat['Close']
    
    # Bollinger Bands (20, 2)
    sma20 = feat['Close'].rolling(20).mean()
    std20 = feat['Close'].rolling(20).std()
    bbu   = sma20 + 2 * std20
    bbl   = sma20 - 2 * std20
    feat['BB_pct_b'] = (feat['Close'] - bbl) / (bbu - bbl + 1e-8)
    feat['BB_width'] = (bbu - bbl) / feat['Close']
    
    # ATR (14)
    tr1 = feat['High'] - feat['Low']
    tr2 = (feat['High'] - feat['Close'].shift()).abs()
    tr3 = (feat['Low'] - feat['Close'].shift()).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    feat['ATR_pct'] = tr.rolling(14).mean() / feat['Close']
    
    feat['Vol_pct']   = feat['Volume'].pct_change().clip(-2, 2)
    feat['Volat_5D']  = feat['Return_1D'].rolling(5).std()
    feat['Volat_20D'] = feat['Return_1D'].rolling(20).std()
    
    for lag in range(1, 6):
        feat[f'Lag_Return_{lag}'] = feat['Return_1D'].shift(lag)
        
    feat.dropna(inplace=True)
    return feat

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, 'templates'))

@app.get('/', response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse('index.html', {'request': request, 'metrics': metrics_data})

@app.get('/api/predict')
def get_predict(ticker: str = 'AAPL'):
    ticker = ticker.upper()
    TODAY = datetime.date.today()
    start_date = (TODAY - datetime.timedelta(days=250)).strftime('%Y-%m-%d')
    
    try:
        raw = yf.download(ticker, start=start_date, auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
            
        df = raw[['Open', 'High', 'Low', 'Close', 'Volume']].ffill().bfill().sort_index()
        feat_df = compute_native_features(df)
        
        if len(feat_df) < SEQ_LEN:
            return JSONResponse({'error': 'Insufficient data'}, status_code=400)
            
        # Live Next Day Prediction
        latest_raw = feat_df[FEATURE_COLS].values[-SEQ_LEN:]
        latest_scaled = scaler.transform(latest_raw)
        pred_return = model.predict(latest_scaled[np.newaxis], verbose=0)[0, 0]
        
        last_close = float(feat_df['Close'].iloc[-1])
        last_date = feat_df.index[-1].strftime('%Y-%m-%d')
        pred_close = float(last_close * (1.0 + pred_return))
        change_pct = float(pred_return * 100)
        
        next_date = pd.bdate_range(start=feat_df.index[-1] + pd.Timedelta(days=1), periods=1)[0].strftime('%Y-%m-%d')
        
        # 30-Day Historical Backtest
        BACKTEST_DAYS = 30
        all_feat_vals = feat_df[FEATURE_COLS].values
        all_scaled_vals = scaler.transform(all_feat_vals)
        
        bt_dates = []
        bt_actuals = []
        bt_preds = []
        
        start_idx = len(all_scaled_vals) - BACKTEST_DAYS - 1
        for i in range(BACKTEST_DAYS):
            idx = start_idx + i
            win = all_scaled_vals[idx - SEQ_LEN + 1 : idx + 1]
            act = float(feat_df['Close'].iloc[idx + 1])
            d_str = feat_df.index[idx + 1].strftime('%Y-%m-%d')
            
            p_ret = float(model.predict(win[np.newaxis], verbose=0)[0, 0])
            prev_c = float(feat_df['Close'].iloc[idx])
            p_close = float(prev_c * (1.0 + p_ret))
            
            bt_dates.append(d_str)
            bt_actuals.append(round(act, 2))
            bt_preds.append(round(p_close, 2))
            
        # Technical Summary
        rsi_val = float(feat_df['RSI_14'].iloc[-1] * 100)
        volat_val = float(feat_df['Volat_20D'].iloc[-1] * 100)
        
        response_payload = {
            "ticker": ticker,
            "last_date": last_date,
            "last_close": round(last_close, 2),
            "next_date": next_date,
            "predicted_close": round(pred_close, 2),
            "expected_return_pct": round(change_pct, 2),
            "signal": "BUY" if change_pct > 0 else "SELL",
            "rsi": round(rsi_val, 1),
            "volatility_pct": round(volat_val, 2),
            "backtest": {
                "dates": bt_dates,
                "actuals": bt_actuals,
                "predictions": bt_preds
            },
            "metrics": metrics_data
        }
        return JSONResponse(response_payload)
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
