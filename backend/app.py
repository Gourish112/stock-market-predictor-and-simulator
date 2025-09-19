from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import numpy as np
import requests
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
import os
import time

load_dotenv()

app = Flask(__name__)
FRONTEND_URL = os.getenv("CORS_ORIGIN")
API_KEY = os.getenv("TWELVE_API_KEY")

# Configure CORS for both Flask and Flask-SocketIO
CORS(app, origins=FRONTEND_URL)
socketio = SocketIO(app, cors_allowed_origins=FRONTEND_URL)

# Load model
model_path = "stock_model_multihorizon_keras.keras"
model = tf.keras.models.load_model(model_path)

# --- Caching Layer ---
CACHE = {}
CACHE_TTL = 60  # seconds

def cached_fetch(symbol, interval="1day", outputsize=100):
    """Fetch stock data from Twelve Data with caching"""
    key = f"{symbol}_{interval}_{outputsize}"
    now = time.time()

    # Return cached result if fresh
    if key in CACHE and now - CACHE[key]["time"] < CACHE_TTL:
        return CACHE[key]["data"]

    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize={outputsize}&apikey={API_KEY}"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if "values" not in data:
            print("❌ Error from Twelve Data:", data)
            return None

        df = data["values"][::-1]  # reverse chronological order
        CACHE[key] = {"data": df, "time": now}
        return df
    except Exception as e:
        print("⚠️ Fetch error:", e)
        return None


def get_stock_input(symbol):
    df = cached_fetch(symbol)
    if not df or len(df) < 100:
        return None, None

    # Extract close prices
    closes = [float(item["close"]) for item in df[-100:]]
    data = np.array(closes).reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    return np.array([data_scaled]), scaler


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        symbol = data.get("ticker", "AAPL")  # Default to Apple

        X_input, scaler = get_stock_input(symbol)
        if X_input is None or scaler is None:
            return jsonify({"error": f"Invalid symbol '{symbol}' or insufficient data"}), 400

        y_pred = model.predict(X_input)
        predictions = {
            "1_day": round(float(scaler.inverse_transform(y_pred)[0][0]), 2),
            "5_days": round(float(scaler.inverse_transform(y_pred)[0][1]), 2),
            "1_month": round(float(scaler.inverse_transform(y_pred)[0][2]), 2),
            "1_year": round(float(scaler.inverse_transform(y_pred)[0][3]), 2)
        }

        print(f"✅ Predictions for {symbol}: {predictions}")
        return jsonify(predictions)

    except Exception as e:
        print("❌ Prediction Error:", str(e))
        return jsonify({"error": "Prediction failed"}), 500


@app.route('/simulate', methods=['POST'])
def simulate():
    try:
        data = request.json
        symbol = data.get("ticker", "AAPL")
        # Ensure the interval is supported by the API (e.g., '1h', not '3mo')
        interval = data.get("interval", "1h")

        df = cached_fetch(symbol, interval=interval, outputsize=200)
        if not df:
            return jsonify({"error": "No data available"}), 400

        timestamps = [item["datetime"] for item in df]
        prices = [float(item["close"]) for item in df]

        return jsonify({
            "ticker": symbol,
            "timestamps": timestamps,
            "prices": prices
        })

    except Exception as e:
        print("❌ Simulation Error:", str(e))
        return jsonify({"error": "Simulation failed"}), 500


# --- WebSocket Events ---
@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

# You can add more socketio event handlers here if needed.
# For example, to handle custom events from the client:
# @socketio.on('my_event')
# def handle_my_custom_event(json):
#     print('received json: ' + str(json))
#     emit('my_response', json)

if __name__ == '__main__':
    # Use socketio.run() instead of app.run() to start the server
    socketio.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
