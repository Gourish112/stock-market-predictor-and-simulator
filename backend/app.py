#!/usr/bin/env python3
"""
app.py - Flask + Flask-SocketIO backend with Twelve Data integration,
Redis caching (optional), and model-based prediction endpoint.
"""

import os
import json
import time
import logging
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
import requests
import threading
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# TensorFlow is optional (model may be large). If missing, endpoints will return helpful errors.
try:
    import tensorflow as tf
except Exception:
    tf = None

# Optional Redis
try:
    import redis
except Exception:
    redis = None

# ---------------------------
# Configuration
# ---------------------------
LOG = logging.getLogger("stock-backend")
logging.basicConfig(level=logging.INFO)

TWELVE_KEY = os.getenv("TWELVE_DATA_KEY")
if not TWELVE_KEY:
    LOG.warning("TWELVE_DATA_KEY not set — Twelve Data calls will fail.")

CORS_ORIGIN = os.getenv("CORS_ORIGIN", "*")
REDIS_URL = os.getenv("REDIS_URL", None)
MODEL_PATH = os.getenv("MODEL_PATH", "stock_model_multihorizon_keras.keras")
PORT = int(os.getenv("PORT", 5000))

# Flask + SocketIO
app = Flask(__name__)
# Allow either a single origin or comma separated list; fallback to all origins
if CORS_ORIGIN == "*" or not CORS_ORIGIN:
    CORS(app, resources={r"/*": {"origins": "*"}})
    socketio = SocketIO(app, cors_allowed_origins="*")
else:
    # support comma-separated origins
    origins = [o.strip() for o in CORS_ORIGIN.split(",")]
    CORS(app, resources={r"/*": {"origins": origins}})
    socketio = SocketIO(app, cors_allowed_origins=origins)

# ---------------------------
# Redis (optional) + in-memory cache fallback
# ---------------------------
redis_client = None
if REDIS_URL and redis:
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        LOG.info("Connected to Redis.")
    except Exception as e:
        LOG.warning(f"Could not connect to Redis at {REDIS_URL}: {e}")
        redis_client = None
else:
    if REDIS_URL and not redis:
        LOG.warning("REDIS_URL set but 'redis' library not installed; falling back to memory cache.")
    else:
        LOG.info("Redis not configured; using in-memory cache.")

MEM_CACHE = {}
CACHE_LOCK = threading.Lock()


def cache_get(key):
    """Return cached value (Python object) or None."""
    if redis_client:
        try:
            raw = redis_client.get(key)
            if raw:
                return json.loads(raw)
        except Exception as e:
            LOG.debug(f"Redis get error for {key}: {e}")
    with CACHE_LOCK:
        entry = MEM_CACHE.get(key)
        if entry and (time.time() - entry["ts"] < entry["ttl"]):
            return entry["value"]
    return None


def cache_set(key, value, ttl=60):
    """Cache a Python object as JSON (TTL seconds)."""
    if redis_client:
        try:
            redis_client.setex(key, ttl, json.dumps(value))
            return
        except Exception as e:
            LOG.debug(f"Redis set error for {key}: {e}")
    with CACHE_LOCK:
        MEM_CACHE[key] = {"value": value, "ts": time.time(), "ttl": ttl}


# ---------------------------
# Twelve Data helpers
# ---------------------------
TD_BASE = "https://api.twelvedata.com"


def _td_request(path, params, cache_key=None, cache_ttl=60):
    """Generic helper for Twelve Data requests with caching and basic error handling."""
    if cache_key:
        cached = cache_get(cache_key)
        if cached:
            return cached

    params = params.copy()
    params["apikey"] = TWELVE_KEY
    url = f"{TD_BASE}/{path}"
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        # Twelve Data returns {"status":"ok", "values":[...]} or {"status":"error","message":...}
        if isinstance(data, dict) and data.get("status") == "error":
            LOG.warning(f"Twelve Data returned error for {path} {params}: {data.get('message')}")
            return None
        if cache_key:
            cache_set(cache_key, data, ttl=cache_ttl)
        return data
    except Exception as e:
        LOG.exception(f"Twelve Data request failed for {url} - {e}")
        return None


def fetch_quote(symbol):
    """Fetch a single quote for a symbol (cached short-term)."""
    if not TWELVE_KEY:
        return None
    cache_key = f"td:quote:{symbol}"
    return _td_request("quote", {"symbol": symbol}, cache_key=cache_key, cache_ttl=10)


def fetch_time_series(symbol, interval="1day", outputsize=200):
    """Fetch time_series from Twelve Data. Returns the parsed JSON (or None)."""
    if not TWELVE_KEY:
        return None
    cache_key = f"td:ts:{symbol}:{interval}:{outputsize}"
    params = {"symbol": symbol, "interval": interval, "outputsize": outputsize}
    return _td_request("time_series", params, cache_key=cache_key, cache_ttl=60)


def symbol_search(query):
    """Search symbols via Twelve Data symbol_search (returns list)."""
    if not TWELVE_KEY:
        return []
    cache_key = f"td:search:{query}"
    data = _td_request("symbol_search", {"symbol": query}, cache_key=cache_key, cache_ttl=60)
    return data.get("data", []) if data else []


# ---------------------------
# Utility: indicators + data shaping
# ---------------------------
def time_series_to_df(values):
    """
    Twelve Data returns a list of dict rows (newest first).
    Convert to a pandas DataFrame chronological (oldest first) and cast numeric columns.
    """
    df = pd.DataFrame(values)
    if df.empty:
        return df
    # ensure chronological order for rolling windows (oldest -> newest)
    df = df.iloc[::-1].reset_index(drop=True)
    # parse numeric columns
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # parse datetime: Twelve Data field is "datetime" which may be ISO or epoch string depending on request
    if "datetime" in df.columns:
        try:
            # try parse ISO first
            df["datetime"] = pd.to_datetime(df["datetime"])
        except Exception:
            try:
                df["datetime"] = pd.to_datetime(df["datetime"].astype(int), unit="s")
            except Exception:
                pass
        df = df.set_index("datetime")
    return df


def compute_indicators(df):
    """Add SMA20, SMA50, RSI, PctChange and Volatility to df (in-place) and return it."""
    if df.empty or "close" not in df.columns:
        return df
    df["SMA_20"] = df["close"].rolling(window=20, min_periods=1).mean()
    df["SMA_50"] = df["close"].rolling(window=50, min_periods=1).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(window=14, min_periods=1).mean()
    loss = -delta.clip(upper=0).rolling(window=14, min_periods=1).mean()
    rs = gain / (loss.replace(0, np.nan))
    df["RSI"] = 100 - (100 / (1 + rs)).fillna(50)

    df["PctChange"] = df["close"].pct_change() * 100
    df["Volatility"] = df["PctChange"].rolling(window=20, min_periods=1).std().fillna(0)
    return df


# ---------------------------
# Model loading (optional)
# ---------------------------
MODEL = None
if tf:
    try:
        MODEL = tf.keras.models.load_model(MODEL_PATH)
        LOG.info(f"Loaded model from {MODEL_PATH}")
    except Exception as e:
        LOG.warning(f"Failed to load model at {MODEL_PATH}: {e}")
        MODEL = None
else:
    LOG.warning("TensorFlow not available; prediction endpoint will be disabled.")


# ---------------------------
# Socket / realtime infra
# ---------------------------
active_subs = {}  # ticker -> {"subs": set(sids), "task": background_task_handle, "base": float}
subs_lock = threading.Lock()


def realistic_movement(base_price):
    """Slight random walk movement."""
    volatility = 0.002
    change = base_price * volatility * (np.random.rand() * 2 - 1)
    return max(0.01, base_price + change)


def _start_price_background(ticker):
    """
    Background loop for a ticker: emits stockUpdate every second to subscribed clients.
    Uses socketio.start_background_task and socketio.sleep for cooperative scheduling.
    """
    def bg():
        LOG.info(f"Background price task starting for {ticker}")
        # get an initial base price from quote if possible
        q = fetch_quote(ticker)
        base_price = None
        if q and isinstance(q, dict) and q.get("status") != "error":
            try:
                base_price = float(q.get("close", q.get("price", 0)) or 0)
            except Exception:
                base_price = None
        if not base_price:
            base_price = 100.0

        # loop until no subscribers
        while True:
            with subs_lock:
                entry = active_subs.get(ticker)
                if not entry or not entry.get("subs"):
                    LOG.info(f"No subscribers left; stopping background task for {ticker}")
                    active_subs.pop(ticker, None)
                    return
                # snapshot of sids
                sids = list(entry["subs"])

            # update price
            base_price = realistic_movement(base_price)
            payload = {
                "ticker": ticker,
                "price": round(float(base_price), 2),
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "volume": int(np.random.rand() * 10000),
                # occasionally include simple generated news
                "news": ({"headline": f"{ticker} random event", "source": "Sim", "sentiment": "neutral"}
                         if (np.random.rand() < 0.05) else None)
            }

            # emit to each subscriber room (room = sid)
            for sid in sids:
                try:
                    socketio.emit("stockUpdate", payload, room=sid)
                except Exception as e:
                    LOG.debug(f"Emit to {sid} failed: {e}")

            # cooperative sleep (works with eventlet/gevent/async)
            socketio.sleep(1.0)

    # fire and forget
    task = socketio.start_background_task(bg)
    return task


@socketio.on("connect")
def handle_connect():
    LOG.info(f"Socket connected: {request.sid}")


@socketio.on("disconnect")
def handle_disconnect():
    sid = request.sid
    LOG.info(f"Socket disconnected: {sid}")
    with subs_lock:
        to_cleanup = []
        for ticker, entry in list(active_subs.items()):
            if sid in entry["subs"]:
                entry["subs"].remove(sid)
            if not entry["subs"]:
                # let background task detect and cleanup; we can also pop
                to_cleanup.append(ticker)
        for t in to_cleanup:
            active_subs.pop(t, None)


@socketio.on("subscribeStock")
def handle_subscribe(data):
    """Client may send either a string 'AAPL' or an object {'ticker':'AAPL'}"""
    sid = request.sid
    if isinstance(data, str):
        ticker = data.strip().upper()
    else:
        ticker = (data.get("ticker") or data.get("symbol") or "").strip().upper()

    if not ticker:
        socketio.emit("error", {"message": "Invalid ticker"}, room=sid)
        return

    LOG.info(f"Socket {sid} subscribe {ticker}")
    with subs_lock:
        if ticker not in active_subs:
            active_subs[ticker] = {"subs": set(), "task": None}
        active_subs[ticker]["subs"].add(sid)
        # start background task if not running
        if not active_subs[ticker].get("task"):
            active_subs[ticker]["task"] = _start_price_background(ticker)

    # send initial snapshot (quote if available)
    q = fetch_quote(ticker)
    if q and isinstance(q, dict) and q.get("status") != "error":
        try:
            initial_price = float(q.get("close", q.get("price", 0)) or 0)
        except Exception:
            initial_price = None
    else:
        initial_price = None

    socketio.emit("stockData", {
        "ticker": ticker,
        "price": round(initial_price, 2) if initial_price else None,
        "currency": q.get("currency") if q else None,
        "exchange": q.get("exchange") if q else None,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
    }, room=sid)


@socketio.on("unsubscribeStock")
def handle_unsubscribe(data):
    sid = request.sid
    ticker = data if isinstance(data, str) else (data.get("ticker") or data.get("symbol") or "").strip().upper()
    if not ticker:
        return
    LOG.info(f"Socket {sid} unsubscribe {ticker}")
    with subs_lock:
        if ticker in active_subs and sid in active_subs[ticker]["subs"]:
            active_subs[ticker]["subs"].remove(sid)
            if not active_subs[ticker]["subs"]:
                # background task will notice and cleanup
                pass


@socketio.on("subscribeMarketNews")
def handle_subscribe_news(_data=None):
    sid = request.sid
    LOG.info(f"{sid} subscribed to market news")

    def news_loop(room):
        while True:
            item = {
                "headline": "Simulated market news",
                "source": "Sim",
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "sentiment": np.random.choice(["positive", "neutral", "negative"]).item() if hasattr(np.random.choice(["positive", "neutral", "negative"]), 'item') else np.random.choice(["positive", "neutral", "negative"])
            }
            try:
                socketio.emit("marketNews", item, room=room)
            except Exception as e:
                LOG.debug(f"Failed to emit marketNews to {room}: {e}")
            socketio.sleep(10.0)

    socketio.start_background_task(news_loop, sid)


# ---------------------------
# HTTP Endpoints
# ---------------------------
@app.route("/health", methods=["GET"])
def health():
    usage = {}
    try:
        # minimal process info (avoid additional deps)
        import psutil
        p = psutil.Process()
        mem = p.memory_info().rss / 1024 / 1024
        usage["memory_mb"] = round(mem, 2)
    except Exception:
        usage["memory_mb"] = None
    return jsonify({
        "status": "ok",
        "time": datetime.datetime.utcnow().isoformat() + "Z",
        "redis": bool(redis_client),
        "model_loaded": bool(MODEL),
        "process": usage
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if MODEL is None:
        return jsonify({"error": "Model not loaded on server"}), 500

    payload = request.get_json(force=True, silent=True) or {}
    symbol = (payload.get("ticker") or payload.get("symbol") or "AAPL").strip().upper()

    # Prepare input for model
    try:
        ts = fetch_time_series(symbol, interval="1day", outputsize=250)
        if not ts or "values" not in ts:
            return jsonify({"error": f"No data from Twelve Data for {symbol}"}), 400
        df = time_series_to_df(ts["values"])
        if df.empty or len(df) < 100:
            return jsonify({"error": f"Insufficient data for {symbol} (need >=100 rows)"}), 400

        closes = df["close"].astype(float).values[-100:].reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(closes)
        X_input = np.array([scaled])  # shape (1, 100, 1) if model expects that
    except Exception as e:
        LOG.exception("Preparing model input failed")
        return jsonify({"error": "Failed to prepare input for prediction"}), 500

    try:
        y_pred = MODEL.predict(X_input)
        # inverse scale the predictions
        # NOTE: y_pred shape must match model output; adjust accordingly
        inv = scaler.inverse_transform(y_pred) if y_pred is not None else None
        # Build predictions object defensively
        preds = {}
        if inv is not None and len(inv.shape) >= 2:
            # map first 4 outputs if present
            for i, name in enumerate(["1_day", "5_days", "1_month", "1_year"]):
                if i < inv.shape[1]:
                    preds[name] = round(float(inv[0, i]), 2)
        else:
            preds["raw"] = y_pred.tolist() if hasattr(y_pred, "tolist") else str(y_pred)

        return jsonify({"symbol": symbol, "predictions": preds})
    except Exception as e:
        LOG.exception("Prediction failed")
        return jsonify({"error": "Prediction failed", "detail": str(e)}), 500


@app.route("/simulate", methods=["POST"])
def api_simulate():
    body = request.get_json(force=True, silent=True) or {}
    ticker = (body.get("ticker") or "AAPL").strip().upper()
    interval = body.get("interval", "1d")

    # map friendly intervals to Twelve Data intervals & outputsize
    interval_map = {
        "1d": ("1min", 390), "5d": ("5min", 390), "1mo": ("1h", 420),
        "3mo": ("1day", 90), "6mo": ("1day", 180), "1y": ("1day", 252), "5y": ("1week", 260)
    }
    if interval not in interval_map:
        return jsonify({"error": "Invalid interval"}), 400

    td_interval, outputsize = interval_map[interval]
    ts = fetch_time_series(ticker, interval=td_interval, outputsize=outputsize)
    if not ts or "values" not in ts:
        return jsonify({"error": f"No historical data for {ticker}"}), 400

    df = time_series_to_df(ts["values"])
    if df.empty or "close" not in df.columns:
        return jsonify({"error": f"No close price data for {ticker}"}), 400

    df = compute_indicators(df)
    result = {
        "ticker": ticker,
        "timestamps": df.index.strftime("%Y-%m-%d %H:%M:%S").tolist(),
        "prices": df["close"].tolist(),
        "volumes": df["volume"].tolist() if "volume" in df.columns else [],
        "sma_20": df["SMA_20"].tolist() if "SMA_20" in df.columns else [],
        "sma_50": df["SMA_50"].tolist() if "SMA_50" in df.columns else [],
        "rsi": df["RSI"].tolist() if "RSI" in df.columns else [],
        "volatility": df["Volatility"].tolist() if "Volatility" in df.columns else [],
        "pct_change": df["PctChange"].tolist() if "PctChange" in df.columns else []
    }
    return jsonify(result)


@app.route("/search-stocks", methods=["GET"])
def api_search_stocks():
    q = request.args.get("query", "").strip()
    if not q or len(q) < 2:
        return jsonify({"results": []})
    data = symbol_search(q)
    out = []
    for item in data:
        out.append({
            "symbol": item.get("symbol"),
            "name": item.get("instrument_name", item.get("symbol")),
            "exchange": item.get("exchange")
        })
    return jsonify({"results": out[:10]})


@app.route("/stock-info", methods=["GET"])
def api_stock_info():
    ticker = request.args.get("ticker", "AAPL").strip().upper()
    q = fetch_quote(ticker)
    if not q:
        return jsonify({"error": "Not found"}), 404
    return jsonify({
        "symbol": q.get("symbol"),
        "name": q.get("name"),
        "currency": q.get("currency"),
        "exchange": q.get("exchange"),
        "close": float(q.get("close", 0))
    })


@app.route("/market-summary", methods=["GET"])
def api_market_summary():
    exchange = request.args.get("exchange", "US")
    if exchange == "IN":
        indices = ["NSE", "BSE"]  # Twelve Data uses NSE/BSE codes in some cases
    else:
        indices = ["^GSPC", "^DJI", "^IXIC"]
    out = {}
    for idx in indices:
        q = fetch_quote(idx)
        if not q:
            continue
        try:
            price = float(q.get("close", 0))
            change = float(q.get("change", 0)) if q.get("change") is not None else 0
            pct = float(q.get("percent_change", 0)) if q.get("percent_change") is not None else 0
        except Exception:
            price, change, pct = 0, 0, 0
        out[idx] = {
            "name": q.get("name", idx),
            "price": price,
            "change": change,
            "percentChange": pct
        }
    return jsonify(out)


# ---------------------------
# App entrypoint for local runs
# ---------------------------
if __name__ == "__main__":
    LOG.info(f"Starting app on port {PORT}")
    # When running locally (python app.py), this will run the socketio server.
    # In Render (Docker) you can use gunicorn with eventlet worker:
    # gunicorn --worker-class eventlet -w 1 -b 0.0.0.0:$PORT app:app
    socketio.run(app, host="0.0.0.0", port=PORT, debug=False)

