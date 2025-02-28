import pandas as pd
import numpy as np
import ta.momentum
import ta.trend
import yfinance as yf
import ta, joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

def fetch_data(symbol="^NSEI", start="2020-01-01", end="2024-12-31"):
    df = yf.download(symbol, start=start, end=end, interval="id")
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.dropna(inplace=True)
    return df


def add_indicators(df):
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
    df['RSI_14'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = ta.trend.macd(df['Close'])
    df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
    df.dropna(inplace=True)
    return df

df = fetch_data()
df = add_indicators(df)


X = df[['SMA_20', 'SMA_50', 'EMA_20', 'RSI_14', 'MACD', 'MACD_Signal']]
Y = df['Close'].shift(-1)

X_train, X_test, y_train, y_test = train_test_split(X[:-1], y[:-1], test_size=0.2, random_state=42)

model = XGBRegressor(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)

joblib.dump(model, "stock_model.pkl")
