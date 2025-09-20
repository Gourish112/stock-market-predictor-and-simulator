from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
import numpy as np
import requests, os, time, threading
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
FRONTEND_URL = os.getenv("CORS_ORIGIN")
API_KEY = os.getenv("TWELVE_API_KEY")

# Configure CORS
CORS(app, origins=[FRONTEND_URL])

# Add SocketIO
socketio = SocketIO(app, cors_allowed_origins="*",async_mode="threading")

# Load model
model_path = "stock_model_multihorizon_keras.keras"
model = tf.keras.models.load_model(model_path)

# --- Caching Layer ---
CACHE = {}
CACHE_TTL = 60  # seconds

def cached_fetch(symbol, interval="1day", outputsize=100):
    key = f"{symbol}_{interval}_{outputsize}"
    now = time.time()
    if key in CACHE and now - CACHE[key]["time"] < CACHE_TTL:
        return CACHE[key]["data"]

    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize={outputsize}&apikey={API_KEY}"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if "values" not in data:
            print("❌ Error from Twelve Data:", data)
            return None
        df = data["values"][::-1]
        CACHE[key] = {"data": df, "time": now}
        return df
    except Exception as e:
        print("⚠️ Fetch error:", e)
        return None


def get_stock_input(symbol):
    df = cached_fetch(symbol)
    if not df or len(df) < 100:
        return None, None
    closes = [float(item["close"]) for item in df[-100:]]
    data = np.array(closes).reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    return np.array([data_scaled]), scaler


# --------- REST Endpoints ---------
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        symbol = data.get("ticker", "AAPL")
        X_input, scaler = get_stock_input(symbol)
        if X_input is None:
            return jsonify({"error": f"Invalid symbol '{symbol}' or insufficient data"}), 400
        y_pred = model.predict(X_input)
        predictions = {
            "1_day": round(float(scaler.inverse_transform(y_pred)[0][0]), 2),
            "5_days": round(float(scaler.inverse_transform(y_pred)[0][1]), 2),
            "1_month": round(float(scaler.inverse_transform(y_pred)[0][2]), 2),
            "1_year": round(float(scaler.inverse_transform(y_pred)[0][3]), 2)
        }
        return jsonify(predictions)
    except Exception as e:
        print("❌ Prediction Error:", str(e))
        return jsonify({"error": "Prediction failed"}), 500

INTERVAL_MAP = {
    "real-time": {"interval": "1min", "outputsize": 30},   # ~30 minutes
    "1d": {"interval": "5min", "outputsize": 78},         # 5-min candles → 288 in 24h
    "5d": {"interval": "30min", "outputsize": 200},        # 5 days with half-hour candles
    "1mo": {"interval": "1day", "outputsize": 30},         # ~1 month daily
    "3mo": {"interval": "1day", "outputsize": 90},         # ~3 months
    "6mo": {"interval": "1day", "outputsize": 180},        # ~6 months
    "1y": {"interval": "1day", "outputsize": 365},         # 1 year
}

@app.route('/simulate', methods=['POST'])
def simulate():
    try:
        data = request.json
        symbol = data.get("ticker", "AAPL")
        frontend_interval = data.get("interval", "1d")

        mapping = INTERVAL_MAP.get(frontend_interval, {"interval": "1day", "outputsize": 30})
        interval = mapping["interval"]
        outputsize = mapping["outputsize"]

        df = cached_fetch(symbol, interval=interval, outputsize=outputsize)
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



# --------- SOCKET.IO ---------
clients = {}

@socketio.on("subscribeStock")
def handle_subscribe(symbol):
    print(f"📡 Client subscribed to {symbol}")
    join_room(symbol)
    clients[request.sid] = symbol

    # send initial data
    df = cached_fetch(symbol, interval="1min", outputsize=30)
    if df:
        timestamps = [item["datetime"] for item in df]
        prices = [float(item["close"]) for item in df]
        emit("stockData", {
            "ticker": symbol,
            "timestamps": timestamps,
            "prices": prices
        }, room=request.sid)


@socketio.on("unsubscribeStock")
def handle_unsubscribe(symbol):
    print(f"❌ Client unsubscribed from {symbol}")
    leave_room(symbol)
    if request.sid in clients:
        del clients[request.sid]


# background job to push updates
def push_updates():
    while True:
        for sid, symbol in list(clients.items()):
            df = cached_fetch(symbol, interval="1min", outputsize=1)
            if df:
                latest = df[-1]
                emit("stockUpdate", {
                    "ticker": symbol,
                    "price": float(latest["close"]),
                    "timestamp": latest["datetime"]
                }, room=symbol)
        time.sleep(10)  # fetch every 10s


threading.Thread(target=push_updates, daemon=True).start()

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
