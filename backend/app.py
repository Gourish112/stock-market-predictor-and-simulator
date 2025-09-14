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

# Configure CORS for both Flask and Flask-SocketIO
CORS(app, origins=FRONTEND_URL)
socketio = SocketIO(app, cors_allowed_origins=FRONTEND_URL)

# --- Real-Time Stock Data ---
active_subscriptions = {}
stock_cache = {}
lock = threading.Lock()

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

def generate_market_news(ticker=None):
    """Generate a random news item."""
    companies = {"AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Google", "AMZN": "Amazon"}
    events = ["announced new product line", "reported quarterly earnings", "CEO made a statement about future plans"]
    
    if ticker is None:
        ticker = np.random.choice(list(companies.keys()))
        
    company_name = companies.get(ticker, ticker)
    event = np.random.choice(events)
    
    return {
        "headline": f"{company_name} {event}",
        "source": np.random.choice(["Bloomberg", "CNBC"]),
        "timestamp": datetime.datetime.now().isoformat(),
        "sentiment": np.random.choice(["positive", "neutral", "negative"])
    }

def start_stock_interval(ticker):
    """Start the real-time price update loop for a given ticker."""
    with lock:
        if ticker not in active_subscriptions or "thread" in active_subscriptions[ticker]:
            return
        
        stock_data = fetch_and_cache_stock_data(ticker)
        if not stock_data:
            return

        def update_prices():
            current_price = stock_data["latestPrice"]
            while True:
                with lock:
                    if ticker not in active_subscriptions or not active_subscriptions[ticker]["subscribers"]:
                        print(f"Stopping interval for {ticker}")
                        if ticker in active_subscriptions:
                            del active_subscriptions[ticker]
                        return

                current_price = get_realistic_price_movement(current_price)
                include_news = np.random.rand() < 0.05
                news_item = generate_market_news(ticker) if include_news else None
                
                socketio.emit("stockUpdate", {
                    "ticker": ticker,
                    "price": round(current_price, 2),
                    "timestamp": datetime.datetime.now().isoformat(),
                    "volume": int(np.random.rand() * 10000),
                    "news": news_item
                })
                
                time.sleep(1)

        thread = threading.Thread(target=update_prices)
        thread.daemon = True
        thread.start()
        active_subscriptions[ticker]["thread"] = thread

@socketio.on("connect")
def on_connect():
    print(f"Client connected: {request.sid}")

@socketio.on("disconnect")
def on_disconnect():
    print(f"Client disconnected: {request.sid}")
    with lock:
        for ticker in list(active_subscriptions.keys()):
            if request.sid in active_subscriptions[ticker]["subscribers"]:
                active_subscriptions[ticker]["subscribers"].remove(request.sid)

@socketio.on("subscribeStock")
def on_subscribe_stock(ticker):
    if not isinstance(ticker, str):
        return emit("error", {"message": "Invalid ticker provided"})

    print(f"Client {request.sid} subscribed to {ticker}")
    
    with lock:
        if ticker not in active_subscriptions:
            active_subscriptions[ticker] = {"subscribers": set(), "thread": None}
        active_subscriptions[ticker]["subscribers"].add(request.sid)

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
    
    start_stock_interval(ticker)

@socketio.on("unsubscribeStock")
def on_unsubscribe_stock(ticker):
    print(f"Client {request.sid} unsubscribed from {ticker}")
    with lock:
        if ticker in active_subscriptions and request.sid in active_subscriptions[ticker]["subscribers"]:
            active_subscriptions[ticker]["subscribers"].remove(request.sid)

@socketio.on("subscribeMarketNews")
def on_subscribe_market_news():
    print(f"Client {request.sid} subscribed to market news")
    
    def news_loop():
        while True:
            socketio.emit("marketNews", generate_market_news(), room=request.sid)
            time.sleep(10)
    
    thread = threading.Thread(target=news_loop)
    thread.daemon = True
    thread.start()

# --- Stock Prediction & Historical Data ---
try:
    model_path = "stock_model_multihorizon_keras.keras"
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    print(f"Failed to load the model: {e}")
    model = None

@lru_cache(maxsize=100)
def cached_fetch_twelve_data(symbol, interval="1day", outputsize=100):
    """Fetch stock data from Twelve Data with caching."""
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize={outputsize}&apikey={API_KEY}"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if "values" not in data or data["status"] != "ok":
            return None
        return data["values"][::-1] # Reverse the data to be chronological
    except Exception as e:
        print(f"⚠️ Twelve Data Fetch error: {e}")
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
        print(f"✅ Predictions for {symbol}: {predictions}")
        return jsonify(predictions)
    except Exception as e:
        print(f"❌ Prediction Error: {str(e)}")
        return jsonify({"error": "Prediction failed"}), 500

def get_historical_data_twelve(ticker, interval, exchange="US"):
    """Fetch historical data with indicators from Twelve Data."""
    try:
        interval_map = {"1d": "1min", "5d": "5min", "1mo": "1h", "3mo": "1h", "6mo": "1day", "1y": "1day", "5y": "1week"}
        output_map = {"1d": 390, "5d": 390, "1mo": 420, "3mo": 180, "6mo": 180, "1y": 252, "5y": 260}
        
        if interval not in interval_map:
            return {"error": "Invalid interval"}

        # Combine requests for OHLCV and indicators
        url_base = f"https://api.twelvedata.com/time_series?symbol={ticker}&interval={interval_map[interval]}&outputsize={output_map[interval]}&apikey={API_KEY}"
        
        # Twelve Data requires separate calls for indicators on the free tier
        url_sma20 = f"{url_base}&indicator=SMA&time_period=20"
        url_sma50 = f"{url_base}&indicator=SMA&time_period=50"
        url_rsi = f"{url_base}&indicator=RSI"
        
        # Make requests
        resp_base = requests.get(url_base, timeout=10).json()
        resp_sma20 = requests.get(url_sma20, timeout=10).json()
        resp_sma50 = requests.get(url_sma50, timeout=10).json()
        resp_rsi = requests.get(url_rsi, timeout=10).json()

        if "values" not in resp_base or resp_base["status"] != "ok":
            return {"error": resp_base.get("message", "No data available")}

        df = pd.DataFrame(resp_base["values"]).astype(float)
        df["datetime"] = [datetime.datetime.fromtimestamp(int(t)) for t in df["datetime"]]
        df = df.set_index("datetime")
        
        # Combine indicators
        df_sma20 = pd.DataFrame(resp_sma20.get("values", []))
        df_sma50 = pd.DataFrame(resp_sma50.get("values", []))
        df_rsi = pd.DataFrame(resp_rsi.get("values", []))

        if not df_sma20.empty:
            df_sma20.set_index("datetime", inplace=True)
            df_sma20.index = [datetime.datetime.fromtimestamp(int(t)) for t in df_sma20.index]
            df["SMA_20"] = df_sma20["SMA"]

        if not df_sma50.empty:
            df_sma50.set_index("datetime", inplace=True)
            df_sma50.index = [datetime.datetime.fromtimestamp(int(t)) for t in df_sma50.index]
            df["SMA_50"] = df_sma50["SMA"]

        if not df_rsi.empty:
            df_rsi.set_index("datetime", inplace=True)
            df_rsi.index = [datetime.datetime.fromtimestamp(int(t)) for t in df_rsi.index]
            df["RSI"] = df_rsi["RSI"]

        # Calculate volatility and percentage change manually
        df["PctChange"] = df["close"].pct_change() * 100
        df["Volatility"] = df["PctChange"].rolling(window=20).std()

        # Get company info
        info_url = f"https://api.twelvedata.com/quote?symbol={ticker}&apikey={API_KEY}"
        info_resp = requests.get(info_url, timeout=5).json()
        
        result = {
            "ticker": ticker,
            "name": info_resp.get("name", ticker),
            "timestamps": df.index.strftime("%Y-%m-%d %H:%M:%S").tolist(),
            "prices": df["close"].tolist(),
            "volumes": df["volume"].tolist(),
            "sma_20": df["SMA_20"].tolist() if "SMA_20" in df else [],
            "sma_50": df["SMA_50"].tolist() if "SMA_50" in df else [],
            "rsi": df["RSI"].tolist() if "RSI" in df else [],
            "volatility": df["Volatility"].tolist(),
            "pct_change": df["PctChange"].tolist()
        }
        return result
    except Exception as e:
        print(f"Error in get_historical_data_twelve for {ticker}: {str(e)}")
        return {"error": f"Failed to fetch data: {str(e)}"}

@app.route('/simulate', methods=['POST'])
def simulate():
    try:
        data = request.json
        ticker = data.get("ticker", "AAPL")
        interval = data.get("interval", "1d")
        exchange = data.get("exchange", "US") # Twelve Data API handles exchange via symbol lookup
        result = get_historical_data_twelve(ticker, interval, exchange)
        if "error" in result:
            return jsonify(result), 400
        return jsonify(result)
    except Exception as e:
        print(f"Simulation Error: {str(e)}")
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
        print(f"Search Error: {str(e)}")
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
        print(f"Info Error: {str(e)}")
        return jsonify({"error": "Failed to get stock info"}), 500

@app.route('/market-summary', methods=['GET'])
def market_summary():
    try:
        indices = ["^GSPC", "^DJI", "^IXIC"]
        if request.args.get('exchange') == "IN":
            indices = ["BSE", "NSE"] # Twelve Data uses these symbols for Indian indices
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
                print(f"Error fetching {idx}: {str(e)}")
        return jsonify(results)
    except Exception as e:
        print(f"Market Summary Error: {str(e)}")
        return jsonify({"error": "Failed to get market summary"}), 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    socketio.run(app, debug=True, host="0.0.0.0", port=port)
