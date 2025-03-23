import sys
import json
import numpy as np
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load trained LSTM model
model = load_model(r"C:\Users\Gourish\Desktop\stockpredictor\models\stock_model_multihorizon_keras.keras", compile=False)

# Function to fetch stock data
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period="5y")  # Fetch 5 years of data
    if df.empty:
        raise ValueError(f"Failed to fetch data for {ticker}")
    return df["Close"].values.reshape(-1, 1)  # Extract closing prices

# Function to predict stock price for multiple timeframes
def predict_stock_price(ticker):
    try:
        # Fetch stock data
        close_prices = get_stock_data(ticker)
        
        # Normalize data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)

        # Prepare input sequence (last 100 days)
        x_test = np.array([scaled_data[-100:]])

        # Predict for multiple timeframes using iterative forecasting
        predictions = {}
        time_intervals = {
            "1_day": 1,
            "5_days": 5,
            "1_month": 30,
            "1_year": 365
        }

        for key, steps in time_intervals.items():
            future_prices = []
            input_seq = x_test.copy()

            for _ in range(steps):  
                y_pred = model.predict(input_seq)  # Predict next day
                future_prices.append(y_pred[0][0])  # Store prediction
                input_seq = np.roll(input_seq, -1, axis=1)  # Shift window
                input_seq[0, -1, 0] = y_pred  # Append new prediction

            # Inverse transform to get actual prices
            predicted_value = scaler.inverse_transform(np.array(future_prices).reshape(-1, 1))[-1][0]
            predictions[key] = float(predicted_value)

        return predictions

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    ticker = sys.argv[1]
    prediction_result = predict_stock_price(ticker)
    print(json.dumps(prediction_result))  # Output JSON for frontend
