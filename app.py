import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

st.title("📈 Live Stock Predictions with Buy/Sell Signals")

symbol = st.text_input("Enter Stock Symbol (e.g., RELIANCE.NS):", "RELIANCE.NS")


@st.cache_data
def fetch_live_data(symbol):
    df = yf.download(symbol, period="7d", interval="1h")
    return df


@st.cache_data
def get_predictions(symbol):
    try:
        response = requests.get(f"http://127.0.0.1:8000/predict/?symbol={symbol}").json()
        return response.get("predictions", {})
    except Exception:
        return {}


@st.cache_data
def get_sentiment(symbol):
    try:
        response = requests.get(f"http://127.0.0.1:8000/sentiment/?symbol={symbol}").json()
        return response.get("sentiment", 0)
    except Exception:
        return 0


df = fetch_live_data(symbol)
predictions = get_predictions(symbol)
sentiment = get_sentiment(symbol)


df["Signal"] = df["Close"].diff().apply(lambda x: "BUY" if x > 0 else "SELL")

fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Actual Price"))
fig.add_trace(go.Scatter(x=df.index[-1:], y=[predictions.get("xgb", 0)], mode="markers", marker=dict(color="red", size=10), name="XGB Prediction"))
fig.add_trace(go.Scatter(x=df.index[-1:], y=[predictions.get("lstm", 0)], mode="markers", marker=dict(color="blue", size=10), name="LSTM Prediction"))

st.plotly_chart(fig)

st.write(f"📊 Sentiment Score: {sentiment}")





