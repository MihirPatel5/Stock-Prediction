from fastapi import FastAPI, HTTPException
import pandas as pd
import yfinance as yf
import ta
import joblib
import xgboost as xgb
from tensorflow import keras
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI()

DATA_COLUMNS = ['SMA_20', 'SMA_50', 'EMA_20', 'RSI_14', 'MACD', 'MACD_Signal']


def fetch_stock_data(symbol="^NSEI", start="2015-01-01", end="2024-01-01"):
    try:
        df = yf.download(symbol, start=start, end=end, interval="1d",auto_adjust=False)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        print("Original Columns:", df.columns)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)  # Flatten MultiIndex if present
        print(df.head())
        print("Flattened Columns:", df.columns)
        # Check if expected columns exist
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(status_code=500, detail=f"Missing columns in data: {missing_columns}")

        df = df[required_columns]  # Select only required columns
        df = df.reset_index()  # Ensure Date is a column

        logging.info(f"Fetched {len(df)} rows of stock data for {symbol}")
        return df
    except Exception as e:
        logging.error(f"Error fetching stock data: {e}")
        raise HTTPException(status_code=500, detail="Error fetching stock data")


def add_indicators(df):
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    print('df: ', df['SMA_20'])
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    print('df[SMA_50]: ', df['SMA_50'])
    df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
    df['RSI_14'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = ta.trend.macd(df['Close'])
    df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
    df.dropna(inplace=True)
    return df


def train_xgboost(df, stock_symbol):
    try:
        X = df[DATA_COLUMNS][:-1]
        print('X: ', X)
        y = df['Close'].shift(-1).dropna()
        print('y: ', y)

        if len(X) != len(y):
            raise ValueError("Mismatch in input and target sizes for XGBoost")

        xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)
        print('xgb_model: ', xgb_model)
        xgb_model.fit(X, y)
        joblib.dump(xgb_model, f"xgb_model_{stock_symbol}.pkl")
        logging.info(f"XGBoost model trained and saved for {stock_symbol}")

    except Exception as e:
        logging.error(f"Error training XGBoost model: {e}")
        raise HTTPException(status_code=500, detail="Error training XGBoost model")


def train_lstm(df, stock_symbol):
    try:
        X = df[DATA_COLUMNS].values[:-1]
        y = df['Close'].shift(-1).dropna().values

        if len(X) != len(y):
            raise ValueError("Mismatch in input and target sizes for LSTM")

        X = X.reshape((X.shape[0], 1, X.shape[1]))

        lstm_model = keras.Sequential([
            keras.layers.LSTM(50, return_sequences=True, input_shape=(1, X.shape[2])),
            keras.layers.LSTM(50),
            keras.layers.Dense(1)
        ])

        lstm_model.compile(optimizer='adam', loss='mean_squared_error')
        lstm_model.fit(X, y, epochs=50, batch_size=32, verbose=0)
        lstm_model.save(f"lstm_model_{stock_symbol}.h5")
        logging.info(f"LSTM model trained and saved for {stock_symbol}")

    except Exception as e:
        logging.error(f"Error training LSTM model: {e}")
        raise HTTPException(status_code=500, detail="Error training LSTM model")


@app.get("/train/")
def train_stock_model(symbol: str):
    df = fetch_stock_data(symbol)
    df = add_indicators(df)
    print(df.head())
    train_xgboost(df, symbol)
    train_lstm(df, symbol)
    return {"message": f"Models trained for {symbol}"}

