from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import numpy as np
import requests
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
import os, time
import yfinance as yf
from functools import lru_cache
import threading
import pandas as pd

load_dotenv()

app = Flask(__name__)
FRONTEND_URL = os.getenv("CORS_ORIGIN", "*")
API_KEY = os.getenv("TWELVE_DATA_KEY")

# Configure CORS for both Flask and Flask-SocketIO
CORS(app, origins=FRONTEND_URL)
socketio = SocketIO(app, cors_allowed_origins=FRONTEND_URL)

# --- Real-Time Stock Data (from socket.js) ---
active_subscriptions = {}
stock_cache = {}
lock = threading.Lock()

def get_realistic_price_movement(base_price):
    """Generate a realistic price movement."""
    volatility = 0.002
    change = base_price * volatility * (np.random.rand() * 2 - 1)
    return base_price + change

def fetch_and_cache_stock_data(ticker):
    """Fetch initial stock data from Yahoo Finance and cache it."""
    with lock:
        if ticker in stock_cache and time.time() - stock_cache[ticker]["timestamp"] < 60:
            return stock_cache[ticker]["data"]
        
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1m&range=1d"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if not data.get("chart") or not data["chart"].get("result") or not data["chart"]["result"][0]:
                raise ValueError("Invalid data format")
                
            result = data["chart"]["result"][0]
            current_price = result["meta"].get("regularMarketPrice", result["indicators"]["quote"][0]["close"][-1] if result["indicators"]["quote"][0]["close"] else 100)
            
            stock_data = {
                "basePrice": current_price,
                "latestPrice": current_price,
                "meta": result["meta"]
            }
            
            stock_cache[ticker] = {"data": stock_data, "timestamp": time.time()}
            return stock_data
            
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return {
                "basePrice": 100,
                "latestPrice": 100,
                "meta": {"currency": "USD", "exchangeName": "Unknown"}
            }

def generate_market_news(ticker=None):
    """Generate a random news item."""
    companies = {"AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Google", "AMZN": "Amazon", "META": "Meta", "TSLA": "Tesla", "NFLX": "Netflix"}
    events = ["announced new product line", "reported quarterly earnings", "CEO made a statement about future plans", "unveiled strategic partnership", "faces regulatory challenges", "stock upgraded by analysts", "stock downgraded by analysts", "plans expansion into new markets", "reported higher than expected revenue", "announced cost-cutting measures"]
    
    if ticker is None:
        ticker = np.random.choice(list(companies.keys()))
        
    company_name = companies.get(ticker, ticker)
    event = np.random.choice(events)
    
    return {
        "headline": f"{company_name} {event}",
        "source": np.random.choice(["Bloomberg", "CNBC", "Reuters", "Financial Times"]),
        "timestamp": datetime.datetime.now().isoformat(),
        "sentiment": np.random.choice(["positive", "neutral", "negative"])
    }

def start_stock_interval(ticker):
    """Start the real-time price update loop for a given ticker."""
    with lock:
        if ticker not in active_subscriptions or "thread" in active_subscriptions[ticker]:
            return
        
        def update_prices():
            current_price = fetch_and_cache_stock_data(ticker)["latestPrice"]
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
                if not active_subscriptions[ticker]["subscribers"]:
                    # The thread will clean itself up
                    pass

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
    emit("stockData", {
        "ticker": ticker,
        "price": round(stock_data["latestPrice"], 2),
        "currency": stock_data["meta"].get("currency", "USD"),
        "exchange": stock_data["meta"].get("exchangeName", "Unknown"),
        "timestamp": datetime.datetime.now().isoformat()
    })
    
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

# --- Stock Prediction (from app.py) ---
try:
    model_path = "stock_model_multihorizon_keras.keras"
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    print(f"Failed to load the model: {e}")
    model = None

CACHE = {}
CACHE_TTL = 60

def cached_fetch(symbol, interval="1day", outputsize=100):
    """Fetch stock data from Twelve Data with caching."""
    key = f"{symbol}_{interval}_{outputsize}"
    now = time.time()
    if key in CACHE and now - CACHE[key]["time"] < CACHE_TTL:
        return CACHE[key]["data"]

    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize={outputsize}&apikey={API_KEY}"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if "values" not in data:
            return None
        df = data["values"][::-1]
        CACHE[key] = {"data": df, "time": now}
        return df
    except Exception as e:
        print(f"⚠️ Twelve Data Fetch error: {e}")
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

# --- Historical and Search Functionality (from simulator.py) ---
@lru_cache(maxsize=100)
def get_stock_info(ticker):
    """Fetch basic stock info with caching."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "name": info.get("shortName", ticker),
            "sector": info.get("sector", "Unknown"),
            "marketCap": info.get("marketCap", 0),
            "currency": info.get("currency", "USD"),
            "exchange": info.get("exchange", "Unknown")
        }
    except Exception as e:
        print(f"Error fetching stock info for {ticker}: {str(e)}")
        return {"name": ticker, "sector": "Unknown", "marketCap": 0, "currency": "USD", "exchange": "Unknown"}

def get_historical_data(ticker, interval, exchange="US"):
    """Fetch historical data with technical indicators."""
    try:
        period_map = {"1d": "1d", "5d": "5d", "1mo": "1mo", "3mo": "3mo", "6mo": "6mo", "1y": "1y", "5y": "5y"}
        interval_map = {"1d": "5m", "5d": "15m", "1mo": "1h", "3mo": "1d", "6mo": "1d", "1y": "1d", "5y": "1wk"}
        if interval not in period_map:
            return {"error": "Invalid interval"}
        if exchange == "IN" and not ticker.endswith(".NS"):
            ticker = f"{ticker}.NS"
        stock = yf.Ticker(ticker)
        df = stock.history(period=period_map[interval], interval=interval_map[interval])
        if df.empty or "Close" not in df.columns:
            return {"error": "No data available for this ticker"}
        df["SMA_20"] = df["Close"].rolling(window=20).mean()
        df["SMA_50"] = df["Close"].rolling(window=50).mean()
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))
        df["PctChange"] = df["Close"].pct_change() * 100
        df["Volatility"] = df["PctChange"].rolling(window=20).std()
        result = {
            "ticker": ticker,
            "timestamps": df.index.strftime("%Y-%m-%d %H:%M:%S").tolist(),
            "prices": df["Close"].tolist(),
            "volumes": df["Volume"].tolist(),
            "sma_20": df["SMA_20"].tolist(),
            "sma_50": df["SMA_50"].tolist(),
            "rsi": df["RSI"].tolist(),
            "volatility": df["Volatility"].tolist(),
            "pct_change": df["PctChange"].tolist()
        }
        result.update(get_stock_info(ticker))
        return result
    except Exception as e:
        print(f"Error in get_historical_data for {ticker}: {str(e)}")
        return {"error": f"Failed to fetch data: {str(e)}"}

@app.route('/simulate', methods=['POST'])
def simulate():
    try:
        data = request.json
        ticker = data.get("ticker", "AAPL")
        interval = data.get("interval", "1d")
        exchange = data.get("exchange", "US")
        result = get_historical_data(ticker, interval, exchange)
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
        exchange = request.args.get('exchange', 'US')
        if not query or len(query) < 2:
            return jsonify({"results": []})
        results = []
        if exchange == "IN":
            tickers = yf.Tickers(f"{query}.*").tickers
            for symbol in tickers:
                if symbol.endswith(".NS"):
                    try:
                        info = tickers[symbol].info
                        results.append({"symbol": symbol, "name": info.get("shortName", symbol), "exchange": "NSE"})
                    except:
                        pass
        else:
            tickers = yf.Tickers(f"{query}.*").tickers
            for symbol in tickers:
                if not symbol.endswith(".NS"):
                    try:
                        info = tickers[symbol].info
                        results.append({"symbol": symbol, "name": info.get("shortName", symbol), "exchange": info.get("exchange", "Unknown")})
                    except:
                        pass
        return jsonify({"results": results[:10]})
    except Exception as e:
        print(f"Search Error: {str(e)}")
        return jsonify({"error": "Search failed"}), 500

@app.route('/stock-info', methods=['GET'])
def stock_info():
    try:
        ticker = request.args.get('ticker', 'AAPL')
        exchange = request.args.get('exchange', 'US')
        if exchange == "IN" and not ticker.endswith(".NS"):
            ticker = f"{ticker}.NS"
        info = get_stock_info(ticker)
        return jsonify(info)
    except Exception as e:
        print(f"Info Error: {str(e)}")
        return jsonify({"error": "Failed to get stock info"}), 500

@app.route('/market-summary', methods=['GET'])
def market_summary():
    try:
        indices = ["^GSPC", "^DJI", "^IXIC"]
        if request.args.get('exchange') == "IN":
            indices = ["^NSEI", "^BSESN"]
        results = {}
        for idx in indices:
            try:
                index_data = yf.Ticker(idx)
                hist = index_data.history(period="1d")
                if not hist.empty:
                    current = hist["Close"].iloc[-1]
                    prev_close = hist["Close"].iloc[-2] if len(hist) > 1 else current
                    change = current - prev_close
                    pct_change = (change / prev_close) * 100 if prev_close else 0
                    results[idx] = {
                        "name": index_data.info.get("shortName", idx),
                        "price": current,
                        "change": change,
                        "percentChange": pct_change
                    }
            except Exception as e:
                print(f"Error fetching {idx}: {str(e)}")
        return jsonify(results)
    except Exception as e:
        print(f"Market Summary Error: {str(e)}")
        return jsonify({"error": "Failed to get market summary"}), 500

# You can also keep the '/api/market-overview' from the Node.js file if you need it.
@app.route('/api/market-overview', methods=['GET'])
def market_overview():
    try:
        indices = ["^GSPC", "^DJI", "^IXIC"]
        results = {}
        for idx in indices:
            data = fetch_and_cache_stock_data(idx)
            results[idx] = {
                "name": {"^GSPC": "S&P 500", "^DJI": "Dow Jones", "^IXIC": "NASDAQ"}.get(idx, idx),
                "price": data["latestPrice"],
                "change": (np.random.rand() * 2 - 1) * (data["latestPrice"] * 0.01),
                "percentChange": (np.random.rand() * 2 - 1) * 1.5
            }
        return jsonify(results)
    except Exception as e:
        print(f"Market overview error: {e}")
        return jsonify({"error": "Failed to fetch market overview"}), 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    socketio.run(app, debug=True, host="0.0.0.0", port=port)






