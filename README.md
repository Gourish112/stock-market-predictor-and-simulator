# 📈 Stock Market Predictor & Simulator

A powerful web application built using the MERN stack that allows users to predict stock prices using AI/ML models and simulate trading in real time. The platform combines LSTM-based deep learning predictions with interactive graphs and paper trading functionality for an educational and practical experience.

## 🚀 Features

### 🔮 Stock Predictor
- Predict stock prices for:
  - 1 Day
  - 5 Days
  - 1 Month
  - 1 Year
- Uses advanced LSTM + Conv1D + Bidirectional neural networks
- Trained on historical data from Yahoo Finance
- Includes sentiment analysis for 1-year prediction

### 📊 Stock Simulator
- Real-time stock data visualization
- Trading simulation with buy/sell options
- Real-time profit/loss calculations
- Transaction history tracking
- Graphs similar to TradingView
- Dropdown to switch between time intervals (1D, 5D, 1M, 1Y)

## 🧠 Tech Stack

### Frontend
- React.js
- Chart.js / Recharts
- Axios

### Backend
- Node.js
- Express.js
- MongoDB (Mongoose)

### ML Model
- Python (TensorFlow / Keras)
- LSTM, Conv1D, Bidirectional Layers
- Trained using Yahoo Finance data (`yfinance`)
- Flask API 

## 📦 Installation

### Prerequisites
- Node.js & npm
- Python 3.x
- MongoDB
- TensorFlow (for training)
- Render (for deployment)

### 1. Clone the repo
```bash
git clone https://github.com/your-username/stock-market-predictor-simulator.git
cd stock-market-predictor-simulator
```
### 2. Run backend
```bash
cd backend
python app.py
```
### 3. Run frontend
```bash
cd frontend
npm start
```

---

## 📷 Screenshots

### 📈 Stock Predictor

![Stock Predictor Screenshot](./assets/screenshots/predictor.png)

### 💹 Trading Simulator

![Trading Simulator Screenshot](./assets/screenshots/simulator.png)

## Contact
Gourish Bhatia
gourishbhatia2004@gmail.com
