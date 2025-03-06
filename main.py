from fastapi import FastAPI, HTTPException
import pandas as pd
import yfinance as yf
import ta
import joblib
import xgboost as xgb
import numpy as np
import os
import logging
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from typing import Optional
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from transformers import AutoTokenizer, AutoModel
import torch
import requests
from io import BytesIO
import keras

# Initialize FastAPI
app = FastAPI()
logging.basicConfig(level=logging.INFO)

DATA_COLUMNS = [
    'SMA_20', 'SMA_50', 'EMA_20', 'RSI_14', 'MACD', 'MACD_Signal',
    'BB_high', 'BB_low', 'ATR', 'OBV', 'Stoch_%K', 'Stoch_%D'
]
SEQ_LENGTH = 30  
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
finbert = AutoModel.from_pretrained("yiyanghkust/finbert-tone")

def fetch_stock_data(symbol: str, years: int = 10) -> pd.DataFrame:
    """Fetch historical stock data with dynamic end date"""
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=years*365)).strftime('%Y-%m-%d')
    
    try:
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if df.empty:
            raise ValueError(f"No data found for {symbol}")
            
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        return df.reset_index()
    except Exception as e:
        logging.error(f"Error fetching data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add comprehensive technical indicators"""
    print("adding indicators")
    df.columns = df.columns.droplevel(1)
    print("df columns:", df.columns)
    print("df shape:", df.shape)
    df['Close'] = df['Close'].astype(float)
    print('df[Close]: ', df['Close'])

    # Simple Moving Averages
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    print('df[SMA_20]: ', df['SMA_20'])
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)

    # Exponential Moving Average
    df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)

    # Relative Strength Index
    df['RSI_14'] = ta.momentum.rsi(df['Close'], window=14)

    # MACD
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()  # Ensure it's a 1D Series
    df['MACD_Signal'] = macd.macd_signal()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['Close'])
    df['BB_high'] = bb.bollinger_hband()
    df['BB_low'] = bb.bollinger_lband()

    # Average True Range
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])

    # On-Balance Volume (Ensure 'Volume' is float)
    df['Volume'] = df['Volume'].astype(float)
    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])

    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=14)
    df['Stoch_%K'] = stoch.stoch()
    df['Stoch_%D'] = stoch.stoch_signal()

    # Drop NaN values (some indicators return NaN for first few rows)
    df = df.dropna()
    print('df: ', df)

    return df

def create_sequences(data: np.ndarray, targets: np.ndarray, seq_length: int) -> tuple:
    """Create time sequences for LSTM training"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(targets[i+seq_length])
    return np.array(X), np.array(y)

def get_finbert_sentiment(text: str) -> float:
    """Get financial sentiment score using FinBERT"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = finbert(**inputs)
    embeddings = torch.mean(outputs.last_hidden_state, dim=1)
    return embeddings.detach().numpy()[0][0]

@app.get("/train/")
async def train_model(symbol: str, test_size: float = 0.2):
    try:
        df = fetch_stock_data(symbol)
        print('df:---------------------------- ', df)
        df = add_technical_indicators(df)
        print('df: ', df)

        features = df[DATA_COLUMNS]
        print('features: ', features)
        target = df['Close'].values.reshape(-1, 1)
        
        split_idx = int(len(features) * (1 - test_size))
        X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
        y_train, y_test = target[:split_idx], target[split_idx:]
        
        feature_scaler = MinMaxScaler()
        X_train_scaled = feature_scaler.fit_transform(X_train)
        X_test_scaled = feature_scaler.transform(X_test)
        
        target_scaler = MinMaxScaler()
        y_train_scaled = target_scaler.fit_transform(y_train)
        y_test_scaled = target_scaler.transform(y_test)
        
        joblib.dump(feature_scaler, f"{MODEL_DIR}/feature_scaler_{symbol}.pkl")
        joblib.dump(target_scaler, f"{MODEL_DIR}/target_scaler_{symbol}.pkl")
        
        xgb_model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.01,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=20,
            eval_metric='rmse'
        )
        xgb_model.fit(
            X_train_scaled, 
            y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False
        )
        joblib.dump(xgb_model, f"{MODEL_DIR}/xgb_{symbol}.pkl")
        
        X_seq, y_seq = create_sequences(X_train_scaled, y_train_scaled, SEQ_LENGTH)
        lstm_model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True, input_shape=(SEQ_LENGTH, len(DATA_COLUMNS)))),
            Dropout(0.3),
            LSTM(64),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        
        lstm_model.compile(optimizer='adam', loss='mse')
        early_stop = EarlyStopping(monitor='val_loss', patience=10)
        checkpoint = ModelCheckpoint(
            f"{MODEL_DIR}/lstm_{symbol}.h5",
            save_best_only=True,
            monitor='val_loss'
        )
        
        lstm_model.fit(
            X_seq, y_seq,
            epochs=100,
            batch_size=32,
            validation_split=0.1,
            callbacks=[early_stop, checkpoint],
            verbose=1
        )
        
        return {"message": f"Models trained for {symbol}", "status": "success"}
    except Exception as e:
        logging.error(f"Training error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict/")
async def predict(symbol: str, days: int = 1):
    try:
        df = fetch_stock_data(symbol)
        df = add_technical_indicators(df)
        print('df: ', df)
        
        feature_scaler = joblib.load(f"{MODEL_DIR}/feature_scaler_{symbol}.pkl")
        print('MODEL_DIR: ', MODEL_DIR)
        print('feature_scaler: ', feature_scaler)
        target_scaler = joblib.load(f"{MODEL_DIR}/target_scaler_{symbol}.pkl")
        xgb_model = joblib.load(f"{MODEL_DIR}/xgb_{symbol}.pkl")
        lstm_model = load_model(f"{MODEL_DIR}/lstm_{symbol}.h5", custom_objects={'mse':keras.losses.MeanSquaredError()})
        print('lstm_model: ', lstm_model)
        
        latest_data = feature_scaler.transform(df[DATA_COLUMNS].iloc[-SEQ_LENGTH:])
        
        xgb_pred = xgb_model.predict(latest_data[-1].reshape(1, -1))[0]
        print('xgb_pred: ', xgb_pred)
        
        lstm_input = latest_data.reshape(1, SEQ_LENGTH, len(DATA_COLUMNS))
        print('lstm_input: ', lstm_input)
        lstm_pred_scaled = lstm_model.predict(lstm_input)[0][0]
        print('lstm_pred_scaled: ', lstm_pred_scaled)
        lstm_pred = target_scaler.inverse_transform([[lstm_pred_scaled]])[0][0]
        print('lstm_pred: ', lstm_pred)
        
        news = requests.get(f"https://newsapi.org/v2/everything?q={symbol}&apiKey=YOUR_KEY").json()
        sentiment_scores = [get_finbert_sentiment(article['title']) for article in news.get('articles', [])[:5]]
        avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
        
        final_pred = (0.6 * lstm_pred + 0.4 * xgb_pred) * (1 + 0.1 * avg_sentiment)
        print('final_pred: ', final_pred)
        
        return {
            "symbol": float(symbol),
            "lstm_prediction": float(lstm_pred),
            "xgb_prediction": float(xgb_pred),
            "sentiment_impact": float(avg_sentiment),
            "final_prediction": float(final_pred)
        }
        
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sentiment/")
async def get_sentiment(symbol: str):
    try:
        news = requests.get(f"https://newsapi.org/v2/everything?q={symbol}&apiKey=09a69dd907da4ac3801e3d48bac54d66").json()
        scores = [get_finbert_sentiment(article['title']) for article in news.get('articles', [])[:5]]
        return {"symbol": symbol, "average_sentiment": np.mean(scores) if scores else 0}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))