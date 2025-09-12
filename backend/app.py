from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
import os, time

load_dotenv()

app = Flask(__name__)
FRONTEND_URL = os.getenv("CORS_ORIGIN")

# Configure CORS with frontend URL
CORS(app, origins=[FRONTEND_URL])

# Load the trained model once
model_path = "stock_model_multihorizon_keras.keras"
model = tf.keras.models.load_model(model_path)

# Portfolio Simulation Variables
portfolio = {}
balance = 10000  # Starting balance
transaction_history = []

# --- Caching Layer ---
CACHE = {}
CACHE_TTL = 60  # seconds

def cached_fetch(ticker, period="200d", interval="1d"):
    """Fetch stock data with caching and retry logic"""
    key = f"{ticker}_{period}_{interval}"
    now = time.time()

    # ✅ return from cache if fresh
    if key in CACHE and now - CACHE[key]["time"] < CACHE_TTL:
        return CACHE[key]["data"]

    # Retry wrapper for yfinance
    retries, delay = 3, 2
    for i in range(retries):
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval)
            if df.empty:
                raise ValueError("No data received")

            CACHE[key] = {"data": df, "time": now}
            return df
        except Exception as e:
            print(f"⚠️ Error fetching {ticker}: {e}, retry {i+1}/{retries}")
            time.sleep(delay)
            delay *= 2

    return None


# --- Helper to prepare stock input for model ---
def get_stock_input(ticker, period="200d", interval="1d"):
    df = cached_fetch(ticker, period, interval)
    if df is None or len(df) < 100:
        return None, None

    data = df["Close"].values[-100:].reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    return np.array([data_scaled]), scaler


# --- Prediction Route ---
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        ticker = data.get("ticker", "^GSPC")

        X_input, scaler = get_stock_input(ticker)
        if X_input is None or scaler is None:
            return jsonify({"error": f"Invalid ticker '{ticker}' or insufficient data"}), 400

        y_pred = model.predict(X_input)
        predictions = {
            "1_day": round(float(scaler.inverse_transform(y_pred)[0][0]), 2),
            "5_days": round(float(scaler.inverse_transform(y_pred)[0][1]), 2),
            "1_month": round(float(scaler.inverse_transform(y_pred)[0][2]), 2),
            "1_year": round(float(scaler.inverse_transform(y_pred)[0][3]), 2)
        }

        print(f"✅ Predictions for {ticker}: {predictions}")
        return jsonify(predictions)

    except Exception as e:
        print("❌ Prediction Error:", str(e))
        return jsonify({"error": "Prediction failed due to internal server error"}), 500


# --- Simulation Route ---
@app.route('/simulate', methods=['POST'])
def simulate():
    try:
        data = request.json
        ticker = data.get("ticker", "AAPL")
        interval = data.get("interval", "1d")

        df = cached_fetch(ticker, period=interval, interval="1h")
        if df is None or "Close" not in df.columns:
            return jsonify({"error": "Invalid stock ticker or no data available"}), 400

        timestamps = df.index.strftime('%Y-%m-%d %H:%M:%S').tolist()
        prices = df["Close"].tolist()

        return jsonify({
            "ticker": ticker,
            "timestamps": timestamps,
            "prices": prices
        })

    except Exception as e:
        print("❌ Simulation Error:", str(e))
        return jsonify({"error": "Simulation failed due to internal server error"}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))


