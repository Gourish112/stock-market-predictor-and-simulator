from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
CORS(app)  # Allow all origins for CORS

# Load the trained model
model_path = r"C:\Users\Gourish\Desktop\stockpredictor\models\stock_model_multihorizon_keras.keras"
model = tf.keras.models.load_model(model_path)

# Portfolio Simulation Variables
portfolio = {}
balance = 10000  # Starting balance
transaction_history = []

# Function to fetch stock data and prepare input
def get_stock_input(ticker, period="200d", interval="1d"):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)


        if df.empty or len(df) < 100:
            return None, None  # Ensure enough data is available

        data = df["Close"].values[-100:].reshape(-1, 1)  # Last 100 days
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data)

        return np.array([data_scaled]), scaler  # Reshape for LSTM

    except Exception as e:
        print(f"Error fetching stock data for {ticker}: {e}")
        return None, None

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        ticker = data.get("ticker", "^GSPC")  

        X_input, scaler = get_stock_input(ticker)
        if X_input is None or scaler is None:
            return jsonify({"error": f"Invalid ticker '{ticker}' or insufficient data"}), 400

        # ✅ Predict for all timeframes using the same model
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
if __name__ == '__main__':
    app.run(debug=True)