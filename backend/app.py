import os
import json
import random
import time
import requests
import numpy as np
import redis
from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit
from tensorflow.keras.models import load_model

# -----------------------
# Flask + SocketIO setup
# -----------------------
app = Flask(__name__)
app.config["SECRET_KEY"] = "your_secret"
socketio = SocketIO(app, cors_allowed_origins="*")

# -----------------------
# Redis cache setup
# -----------------------
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
r = redis.from_url(redis_url)

# -----------------------
# Twelve Data API Setup
# -----------------------
TWELVE_API_KEY = os.getenv("TWELVE_API_KEY")  # <- set this in Render environment
TWELVE_BASE_URL = "https://api.twelvedata.com"

# -----------------------
# Load ML model
# -----------------------
MODEL_PATH = "stock_model_multihorizon_keras.keras"
model = load_model(MODEL_PATH)

# -----------------------
# Globals for simulation
# -----------------------
active_subscriptions = {}  # {ticker: { "clients": set(), "price": float, "interval": None }}

# -----------------------
# Helpers
# -----------------------
def get_realistic_price_movement(base_price):
    """Generate small random fluctuation for simulation"""
    volatility = 0.002
    change = base_price * volatility * (random.random() * 2 - 1)
    return round(base_price + change, 2)

def fetch_and_cache_stock_data(ticker):
    """Fetch stock data from Twelve Data with Redis caching"""
    cache_key = f"stock:{ticker}"
    cached = r.get(cache_key)
    if cached:
        return json.loads(cached)

    try:
        url = f"{TWELVE_BASE_URL}/quote?symbol={ticker}&apikey={TWELVE_API_KEY}"
        resp = requests.get(url, timeout=5)
        data = resp.json()

        if "price" not in data:
            raise Exception(data.get("message", "Invalid response"))

        current_price = float(data["price"])

        payload = {
            "basePrice": current_price,
            "latestPrice": current_price,
            "meta": {
                "currency": data.get("currency", "USD"),
                "exchange": data.get("exchange", "Unknown"),
                "symbol": data.get("symbol", ticker)
            },
        }

        # Cache for 60 seconds
        r.setex(cache_key, 60, json.dumps(payload))
        return payload
    except Exception as e:
        print(f"⚠️ Error fetching {ticker}: {e}")
        return {"basePrice": 100, "latestPrice": 100, "meta": {"currency": "USD"}}

def generate_market_news(ticker):
    companies = {
        "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Google",
        "AMZN": "Amazon", "META": "Meta", "TSLA": "Tesla", "NFLX": "Netflix"
    }
    events = [
        "announced new product line", "reported quarterly earnings",
        "CEO made a statement", "unveiled strategic partnership",
        "faces regulatory challenges", "stock upgraded by analysts",
        "plans expansion into new markets", "reported higher than expected revenue"
    ]
    return {
        "headline": f"{companies.get(ticker, ticker)} {random.choice(events)}",
        "source": random.choice(["Bloomberg", "CNBC", "Reuters"]),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sentiment": random.choice(["positive", "neutral", "negative"])
    }

# -----------------------
# Socket.IO Handlers
# -----------------------
@socketio.on("connect")
def handle_connect():
    print(f"Client connected: {request.sid}")

@socketio.on("disconnect")
def handle_disconnect():
    print(f"Client disconnected: {request.sid}")
    for ticker, sub in list(active_subscriptions.items()):
        sub["clients"].discard(request.sid)
        if not sub["clients"] and sub.get("interval"):
            socketio.stop_timer(sub["interval"])
            del active_subscriptions[ticker]

@socketio.on("subscribeStock")
def handle_subscribe_stock(ticker):
    if not ticker:
        emit("error", {"message": "Invalid ticker"})
        return

    print(f"Client {request.sid} subscribed to {ticker}")
    if ticker not in active_subscriptions:
        stock_data = fetch_and_cache_stock_data(ticker)
        active_subscriptions[ticker] = {
            "clients": set(),
            "price": stock_data["latestPrice"],
        }

        def send_updates():
            while True:
                price = get_realistic_price_movement(active_subscriptions[ticker]["price"])
                active_subscriptions[ticker]["price"] = price
                news_item = generate_market_news(ticker) if random.random() < 0.05 else None
                socketio.emit("stockUpdate", {
                    "ticker": ticker, "price": price,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "news": news_item
                }, room=list(active_subscriptions[ticker]["clients"]))
                socketio.sleep(1)

        active_subscriptions[ticker]["interval"] = socketio.start_background_task(send_updates)

    active_subscriptions[ticker]["clients"].add(request.sid)
    emit("stockData", {
        "ticker": ticker,
        "price": active_subscriptions[ticker]["price"],
        "currency": "USD",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    })

@socketio.on("unsubscribeStock")
def handle_unsubscribe_stock(ticker):
    if ticker in active_subscriptions:
        active_subscriptions[ticker]["clients"].discard(request.sid)
        if not active_subscriptions[ticker]["clients"]:
            print(f"Stopping {ticker} updates")
            del active_subscriptions[ticker]

@socketio.on("subscribeMarketNews")
def handle_market_news():
    def send_news():
        while True:
            random_ticker = random.choice(["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA"])
            socketio.emit("marketNews", generate_market_news(random_ticker), room=request.sid)
            socketio.sleep(10)
    socketio.start_background_task(send_news)

# -----------------------
# Flask API Routes
# -----------------------
@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.json
    seq = np.array(data.get("sequence", [])).reshape(1, -1, 1)
    preds = model.predict(seq)
    return jsonify({"prediction": preds.tolist()})

@app.route("/api/market-overview")
def market_overview():
    indices = {"AAPL": "Apple", "MSFT": "Microsoft", "RELIANCE.NSE": "Reliance"}
    result = {}
    for idx, name in indices.items():
        stock = fetch_and_cache_stock_data(idx)
        result[idx] = {
            "name": name,
            "price": stock["latestPrice"],
            "change": round(random.uniform(-1, 1), 2),
            "percentChange": round(random.uniform(-1.5, 1.5), 2)
        }
    return jsonify(result)

# -----------------------
# Entry Point
# -----------------------
if __name__ == "__main__":
    import eventlet
    import eventlet.wsgi
    eventlet.monkey_patch()

    socketio.run(app, host="0.0.0.0", port=5000)






