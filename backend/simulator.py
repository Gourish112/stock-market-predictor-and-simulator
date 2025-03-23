from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import datetime

app = Flask(__name__)
CORS(app)

# Fetch past stock data based on interval
def get_past_data(ticker, interval):
    period_map = {"1d": "1d", "5d": "5d", "1mo": "1mo", "1y": "1y"}
    if interval not in period_map:
        return {"error": "Invalid interval"}

    stock = yf.Ticker(ticker)
    df = stock.history(period=period_map[interval], interval="1h")  # Hourly data for better granularity

    if df.empty or "Close" not in df.columns:
        return {"error": "No data available"}

    return {
        "timestamps": df.index.strftime("%Y-%m-%d %H:%M:%S").tolist(),
        "prices": df["Close"].tolist()
    }

# Fetch real-time stock data
def get_real_time_data(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period="1d", interval="1m")

    if df.empty or "Close" not in df.columns:
        return {"error": "No data available"}

    latest_price = df["Close"].iloc[-1]
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return {
        "timestamp": timestamp,
        "latest_price": latest_price
    }

@app.route('/simulate', methods=['POST'])
def simulate():
    try:
        data = request.json
        ticker = data.get("ticker", "AAPL")
        interval = data.get("interval", "1d")  # Default interval

        stock = yf.Ticker(ticker)
        df = stock.history(period=interval, interval="1h")

        if df.empty or "Close" not in df.columns:
            return jsonify({"error": "Invalid stock ticker or no data available"}), 400

        timestamps = df.index.strftime('%Y-%m-%d %H:%M:%S').tolist()
        prices = df["Close"].tolist()

        return jsonify({
            "ticker": ticker,
            "timestamps": timestamps,
            "prices": prices
        })

    except Exception as e:
        print("Simulation Error:", str(e))
        return jsonify({"error": "Simulation failed due to internal server error"}), 500

if __name__ == "__main__":
    app.run(debug=True)
