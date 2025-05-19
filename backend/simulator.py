from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import datetime
import numpy as np
import pandas as pd
from functools import lru_cache

app = Flask(__name__)
CORS(app)

# Cache stock info to reduce API calls
@lru_cache(maxsize=100)
def get_stock_info(ticker):
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
        return {
            "name": ticker,
            "sector": "Unknown",
            "marketCap": 0,
            "currency": "USD",
            "exchange": "Unknown"
        }

# Fetch historical stock data with proper error handling
def get_historical_data(ticker, interval, exchange="US"):
    try:
        period_map = {"1d": "1d", "5d": "5d", "1mo": "1mo", "3mo": "3mo", "6mo": "6mo", "1y": "1y", "5y": "5y"}
        interval_map = {"1d": "5m", "5d": "15m", "1mo": "1h", "3mo": "1d", "6mo": "1d", "1y": "1d", "5y": "1wk"}
        
        if interval not in period_map:
            return {"error": "Invalid interval"}
            
        # Add .NS suffix for Indian stocks if needed
        if exchange == "IN" and not ticker.endswith(".NS"):
            ticker = f"{ticker}.NS"

        stock = yf.Ticker(ticker)
        df = stock.history(period=period_map[interval], interval=interval_map[interval])
        
        if df.empty or "Close" not in df.columns:
            return {"error": "No data available for this ticker"}

        # Calculate additional technical indicators
        df["SMA_20"] = df["Close"].rolling(window=20).mean()
        df["SMA_50"] = df["Close"].rolling(window=50).mean()
        
        # Calculate relative strength index (RSI)
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))
        
        # Calculate price percent change
        df["PctChange"] = df["Close"].pct_change() * 100
        
        # Calculate volatility (standard deviation of percent changes)
        df["Volatility"] = df["PctChange"].rolling(window=20).std()

        # Format the results
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
        
        # Add stock info
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
        
        # Handle real-time simulation vs historical data
        if interval == "real-time":
            interval = "1d"  # Use 1d data for real-time simulation base
        
        result = get_historical_data(ticker, interval, exchange)
        
        if "error" in result:
            return jsonify(result), 400
            
        return jsonify(result)

    except Exception as e:
        print("Simulation Error:", str(e))
        return jsonify({"error": f"Simulation failed: {str(e)}"}), 500

@app.route('/search-stocks', methods=['GET'])
def search_stocks():
    try:
        query = request.args.get('query', '')
        exchange = request.args.get('exchange', 'US')
        
        if not query or len(query) < 2:
            return jsonify({"results": []})
            
        # Use yfinance search (limited capabilities)
        # For a production app, consider using a proper financial API
        if exchange == "IN":
            # Filter for Indian stocks
            tickers = yf.Tickers(f"{query}.*").tickers
            results = []
            for symbol in tickers:
                if symbol.endswith(".NS"):
                    try:
                        info = tickers[symbol].info
                        results.append({
                            "symbol": symbol,
                            "name": info.get("shortName", symbol),
                            "exchange": "NSE"
                        })
                    except:
                        pass
        else:
            # US stocks - simple implementation
            tickers = yf.Tickers(f"{query}.*").tickers
            results = []
            for symbol in tickers:
                if not symbol.endswith(".NS"):
                    try:
                        info = tickers[symbol].info
                        results.append({
                            "symbol": symbol,
                            "name": info.get("shortName", symbol),
                            "exchange": info.get("exchange", "Unknown")
                        })
                    except:
                        pass
                        
        return jsonify({"results": results[:10]})  # Limit to 10 results
        
    except Exception as e:
        print("Search Error:", str(e))
        return jsonify({"error": f"Search failed: {str(e)}"}), 500

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
        print("Info Error:", str(e))
        return jsonify({"error": f"Failed to get stock info: {str(e)}"}), 500

@app.route('/market-summary', methods=['GET'])
def market_summary():
    try:
        # Get major market indices
        indices = ["^GSPC", "^DJI", "^IXIC"]  # S&P 500, Dow Jones, NASDAQ
        if request.args.get('exchange') == "IN":
            indices = ["^NSEI", "^BSESN"]  # NIFTY 50, BSE SENSEX
            
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
        print("Market Summary Error:", str(e))
        return jsonify({"error": f"Failed to get market summary: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)