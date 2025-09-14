from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import numpy as np
import requests
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
import os, time
from functools import lru_cache
import threading
import pandas as pd
import datetime

load_dotenv()

app = Flask(__name__)
FRONTEND_URL = os.getenv("CORS_ORIGIN", "*")
API_KEY = os.getenv("TWELVE_DATA_KEY")

CORS(app, origins=FRONTEND_URL)
socketio = SocketIO(app, cors_allowed_origins=FRONTEND_URL)

# --- Real-Time Stock Data with Single Thread Optimization ---
active_subscriptions = {}
stock_cache = {}
lock = threading.Lock()
update_thread = None
update_thread_running = False

def get_realistic_price_movement(base_price):
    """Generate a realistic price movement."""
    volatility = 0.002
    change = base_price * volatility * (np.random.rand() * 2 - 1)
    return base_price + change

@lru_cache(maxsize=100)
def fetch_and_cache_stock_data(ticker):
    """Fetch initial stock data from Twelve Data and cache it."""
    try:
        url = f"https://api.twelvedata.com/quote?symbol={ticker}&apikey={API_KEY}"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if not data or "status" not in data or data["status"] != "ok":
            print(f"Error fetching data for {ticker}: {data.get('message', 'Unknown error')}")
            return None

        stock_data = {
            "basePrice": float(data.get("close", 100)),
            "latestPrice": float(data.get("close", 100)),
            "meta": {
                "currency": data.get("currency", "USD"),
                "exchangeName": data.get("exchange", "Unknown"),
                "name": data.get("name", ticker)
            }
        }
        return stock_data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def background_stock_updater():
    """Single thread to update all subscribed stocks."""
    global update_thread_running
    print("Starting background stock updater thread.")
    
    with app.app_context():
        while update_thread_running:
            with lock:
                tickers_to_update = list(active_subscriptions.keys())
            
            for ticker in tickers_to_update:
                with lock:
                    if ticker not in active_subscriptions:
                        continue
                    
                    # Fetch and update current price for the ticker
                    current_price = fetch_and_cache_stock_data(ticker)
                    if not current_price:
                        continue
                    
                    current_price = get_realistic_price_movement(current_price["latestPrice"])
                    
                    include_news = np.random.rand() < 0.05
                    news_item = generate_market_news(ticker) if include_news else None
                    
                    socketio.emit("stockUpdate", {
                        "ticker": ticker,
                        "price": round(current_price, 2),
                        "timestamp": datetime.datetime.now().isoformat(),
                        "volume": int(np.random.rand() * 10000),
                        "news": news_item
                    })
            time.sleep(1) # Wait 1 second before next update loop

@socketio.on("connect")
def on_connect():
    global update_thread, update_thread_running
    print(f"Client connected: {request.sid}")
    
    with lock:
        if not update_thread_running:
            update_thread_running = True
            update_thread = threading.Thread(target=background_stock_updater)
            update_thread.daemon = True
            update_thread.start()

@socketio.on("disconnect")
def on_disconnect():
    print(f"Client disconnected: {request.sid}")
    with lock:
        for ticker in list(active_subscriptions.keys()):
            if request.sid in active_subscriptions[ticker]:
                active_subscriptions[ticker].remove(request.sid)
                if not active_subscriptions[ticker]:
                    del active_subscriptions[ticker]

@socketio.on("subscribeStock")
def on_subscribe_stock(ticker):
    if not isinstance(ticker, str):
        return emit("error", {"message": "Invalid ticker provided"})

    print(f"Client {request.sid} subscribed to {ticker}")
    
    with lock:
        if ticker not in active_subscriptions:
            active_subscriptions[ticker] = set()
        active_subscriptions[ticker].add(request.sid)

    stock_data = fetch_and_cache_stock_data(ticker)
    if stock_data:
        emit("stockData", {
            "ticker": ticker,
            "price": round(stock_data["latestPrice"], 2),
            "currency": stock_data["meta"].get("currency", "USD"),
            "exchange": stock_data["meta"].get("exchangeName", "Unknown"),
            "timestamp": datetime.datetime.now().isoformat()
        })
    else:
        emit("error", {"message": f"Failed to fetch initial data for {ticker}"})

@socketio.on("unsubscribeStock")
def on_unsubscribe_stock(ticker):
    print(f"Client {request.sid} unsubscribed from {ticker}")
    with lock:
        if ticker in active_subscriptions and request.sid in active_subscriptions[ticker]:
            active_subscriptions[ticker].remove(request.sid)
            if not active_subscriptions[ticker]:
                del active_subscriptions[ticker]

# --- Other endpoints (unchanged from previous version) ---
# ... All other routes from the previous version remain the same ...
# For brevity, these routes are omitted but should be included in your file.
@lru_cache(maxsize=100)
def cached_fetch_twelve_data(symbol, interval="1day", outputsize=100):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize={outputsize}&apikey={API_KEY}"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if "values" not in data or data["status"] != "ok":
            return None
        return data["values"][::-1]
    except Exception as e:
        return None

def get_stock_input(symbol):
    df = cached_fetch_twelve_data(symbol)
    if not df or len(df) < 100:
        return None, None
    closes = [float(item["close"]) for item in df[-100:]]
    data = np.array(closes).reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    return np.array([data_scaled]), scaler

try:
    model_path = "stock_model_multihorizon_keras.keras"
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    model = None

@app.route('/api/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Prediction model not loaded"}), 500
    try:
        data = request.json
        symbol = data.get("ticker", "AAPL")
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
        return jsonify(predictions)
    except Exception as e:
        return jsonify({"error": "Prediction failed"}), 500

def get_historical_data_twelve(ticker, interval, exchange="US"):
    try:
        interval_map = {"1d": "1min", "5d": "5min", "1mo": "1h", "3mo": "1h", "6mo": "1day", "1y": "1day", "5y": "1week"}
        output_map = {"1d": 390, "5d": 390, "1mo": 420, "3mo": 180, "6mo": 180, "1y": 252, "5y": 260}
        
        if interval not in interval_map:
            return {"error": "Invalid interval"}
        url_base = f"https://api.twelvedata.com/time_series?symbol={ticker}&interval={interval_map[interval]}&outputsize={output_map[interval]}&apikey={API_KEY}"
        resp_base = requests.get(url_base, timeout=10).json()
        if "values" not in resp_base or resp_base["status"] != "ok":
            return {"error": resp_base.get("message", "No data available")}

        df = pd.DataFrame(resp_base["values"]).astype(float)
        df["datetime"] = [datetime.datetime.fromtimestamp(int(t)) for t in df["datetime"]]
        df = df.set_index("datetime")
        df["PctChange"] = df["close"].pct_change() * 100
        df["Volatility"] = df["PctChange"].rolling(window=20).std()
        
        info_url = f"https://api.twelvedata.com/quote?symbol={ticker}&apikey={API_KEY}"
        info_resp = requests.get(info_url, timeout=5).json()
        
        result = {
            "ticker": ticker,
            "name": info_resp.get("name", ticker),
            "timestamps": df.index.strftime("%Y-%m-%d %H:%M:%S").tolist(),
            "prices": df["close"].tolist(),
            "volumes": df["volume"].tolist(),
            "sma_20": [], "sma_50": [], "rsi": [], # Free tier might not have these
            "volatility": df["Volatility"].tolist(),
            "pct_change": df["PctChange"].tolist()
        }
        return result
    except Exception as e:
        return {"error": f"Failed to fetch data: {str(e)}"}

@app.route('/simulate', methods=['POST'])
def simulate():
    try:
        data = request.json
        ticker = data.get("ticker", "AAPL")
        interval = data.get("interval", "1d")
        exchange = data.get("exchange", "US")
        result = get_historical_data_twelve(ticker, interval, exchange)
        if "error" in result:
            return jsonify(result), 400
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": "Simulation failed"}), 500

@app.route('/search-stocks', methods=['GET'])
def search_stocks():
    try:
        query = request.args.get('query', '')
        if not query or len(query) < 2:
            return jsonify({"results": []})
        url = f"https://api.twelvedata.com/symbol_search?symbol={query}&apikey={API_KEY}"
        resp = requests.get(url, timeout=5).json()
        if "data" not in resp:
            return jsonify({"results": []})
        results = []
        for item in resp["data"]:
            results.append({
                "symbol": item.get("symbol"),
                "name": item.get("instrument_name", item.get("symbol")),
                "exchange": item.get("exchange")
            })
        return jsonify({"results": results[:10]})
    except Exception as e:
        return jsonify({"error": "Search failed"}), 500

@app.route('/stock-info', methods=['GET'])
def stock_info():
    try:
        ticker = request.args.get('ticker', 'AAPL')
        url = f"https://api.twelvedata.com/quote?symbol={ticker}&apikey={API_KEY}"
        info_resp = requests.get(url, timeout=5).json()
        if "status" not in info_resp or info_resp["status"] != "ok":
            return jsonify({"error": "Stock not found"}), 404
        info = {
            "name": info_resp.get("name", ticker),
            "symbol": info_resp.get("symbol"),
            "currency": info_resp.get("currency", "USD"),
            "exchange": info_resp.get("exchange", "Unknown"),
            "country": info_resp.get("country", "Unknown")
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({"error": "Failed to get stock info"}), 500

@app.route('/market-summary', methods=['GET'])
def market_summary():
    try:
        indices = ["^GSPC", "^DJI", "^IXIC"]
        results = {}
        for idx in indices:
            try:
                url = f"https://api.twelvedata.com/quote?symbol={idx}&apikey={API_KEY}"
                resp = requests.get(url, timeout=5).json()
                if "status" not in resp or resp["status"] != "ok":
                    continue
                results[idx] = {
                    "name": resp.get("name", idx),
                    "price": float(resp.get("close", 0)),
                    "change": float(resp.get("change", 0)),
                    "percentChange": float(resp.get("percent_change", 0))
                }
            except Exception as e:
                continue
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": "Failed to get market summary"}), 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    socketio.run(app, debug=True, host="0.0.0.0", port=port)
