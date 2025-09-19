// Simulator.js (Rectified)
import React, { useState, useEffect, useCallback, useMemo } from "react";
import axios from "axios";
import { Line } from "react-chartjs-2";
import io from "socket.io-client";

// Constants
// Use the URL of your deployed Render backend
const API_URL = process.env.REACT_APP_API_URL;
const DEFAULT_TICKER = "AAPL";
const DEFAULT_INTERVAL = "1mo";
const DEFAULT_EXCHANGE = "US";
const US_STARTING_BALANCE = 10000;
const IN_STARTING_BALANCE = 800000;

const Simulator = () => {
  // Core state
  const [stockData, setStockData] = useState({
    ticker: DEFAULT_TICKER,
    prices: [],
    timestamps: [],
    volumes: [],
    sma20: [],
    sma50: [],
    rsi: [],
    name: "",
    sector: "",
    exchange: ""
  });
  const [selectedInterval, setSelectedInterval] = useState(DEFAULT_INTERVAL);
  const [ticker, setTicker] = useState(DEFAULT_TICKER);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [exchange, setExchange] = useState(DEFAULT_EXCHANGE);
  
  // Portfolio state
  const [balance, setBalance] = useState(US_STARTING_BALANCE);
  const [holdings, setHoldings] = useState({});
  const [transactions, setTransactions] = useState([]);

  // Socket connection with the correct URL
  const socket = useMemo(() => io(API_URL), []);

  // Update balance when exchange changes
  useEffect(() => {
    setBalance(exchange === "US" ? US_STARTING_BALANCE : IN_STARTING_BALANCE);
  }, [exchange]);

  // Fetch historical data from API
  const fetchStockData = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.post(`${API_URL}/simulate`, {
        ticker: ticker,
        interval: selectedInterval,
        exchange: exchange,
      });

      if (response.data.error) {
        setError(response.data.error);
        return;
      }

      setStockData({
        ticker: response.data.ticker,
        prices: response.data.prices || [],
        timestamps: response.data.timestamps || [],
        volumes: response.data.volumes || [],
        sma20: response.data.sma_20 || [],
        sma50: response.data.sma_50 || [],
        rsi: response.data.rsi || [],
        name: response.data.name || ticker,
        sector: response.data.sector || "Unknown",
        exchange: response.data.exchange || "Unknown"
      });
    } catch (error) {
      console.error("Error fetching stock data:", error);
      setError("Failed to fetch stock data. Please try again.");
    } finally {
      setLoading(false);
    }
  }, [ticker, selectedInterval, exchange]);

  // Fetch data on component mount and when dependencies change
  useEffect(() => {
    fetchStockData();
  }, [fetchStockData]);

  // Set up socket connection for real-time updates
  useEffect(() => {
    if (selectedInterval === "real-time") {
      // Unsubscribe from any previous stock
      socket.emit("unsubscribeStock", stockData.ticker);
      
      // Subscribe to the selected stock
      socket.emit("subscribeStock", ticker);
      
      // Listen for stock updates
      socket.on("stockUpdate", (data) => {
        if (data.ticker === ticker) {
          setStockData(prevData => ({
            ...prevData,
            prices: [...prevData.prices, data.price],
            timestamps: [...prevData.timestamps, data.timestamp]
          }));
        }
      });
      
      // Listen for stock data (initial data)
      socket.on("stockData", (data) => {
        if (data.ticker === ticker) {
          setStockData(prevData => ({
            ...prevData,
            name: data.name || ticker,
            exchange: data.exchange || "Unknown",
            prices: data.prices, // Use initial data
            timestamps: data.timestamps
          }));
        }
      });
      
      // Listen for errors
      socket.on("error", (err) => {
        setError(err.message);
      });
    }
    
    // Cleanup function
    return () => {
      if (selectedInterval === "real-time") {
        socket.emit("unsubscribeStock", ticker);
        socket.off("stockUpdate");
        socket.off("stockData");
        socket.off("error");
      }
    };
  }, [socket, ticker, selectedInterval, stockData.ticker]);

  // Handle trade execution
  const handleTrade = (type) => {
    const latestPrice = stockData.prices[stockData.prices.length - 1] || 0;
    const currency = exchange === "US" ? "USD" : "INR";
    
    if (!latestPrice) {
      setError("Cannot execute trade: No price data available");
      return;
    }
    
    if (type === "buy" && balance >= latestPrice) {
      setBalance(prevBalance => prevBalance - latestPrice);
      setHoldings(prevHoldings => ({
        ...prevHoldings,
        [ticker]: (prevHoldings[ticker] || 0) + 1
      }));
      
      setTransactions(prevTransactions => [
        ...prevTransactions, 
        { 
          type: "Buy", 
          ticker, 
          price: latestPrice, 
          quantity: 1, 
          timestamp: new Date().toISOString(),
          profit: 0, 
          currency
        }
      ]);
    } else if (type === "sell" && holdings[ticker] > 0) {
      const buyTransactions = transactions.filter(tx => 
        tx.ticker === ticker && tx.type === "Buy"
      );
      
      const purchasePrice = buyTransactions.length > 0 
        ? buyTransactions[0].price 
        : latestPrice;
        
      const profit = latestPrice - purchasePrice;
      
      setBalance(prevBalance => prevBalance + latestPrice);
      setHoldings(prevHoldings => ({
        ...prevHoldings,
        [ticker]: prevHoldings[ticker] - 1
      }));
      
      setTransactions(prevTransactions => [
        ...prevTransactions, 
        { 
          type: "Sell", 
          ticker, 
          price: latestPrice, 
          quantity: 1, 
          timestamp: new Date().toISOString(),
          profit, 
          currency
        }
      ]);
    } else {
      setError(type === "buy" 
        ? "Insufficient funds to buy this stock" 
        : "No shares available to sell");
    }
  };

  // Format chart data
  const chartData = {
    labels: stockData.timestamps,
    datasets: [
      {
        label: `${stockData.name || ticker} Stock Price`,
        data: stockData.prices,
        fill: false,
        borderColor: stockData.prices.map((price, index) =>
          index > 0 && price > stockData.prices[index - 1] ? "#26a69a" : "#ef5350"
        ),
        backgroundColor: "rgba(38, 166, 154, 0.1)",
        borderWidth: 2,
        pointRadius: 1,
        pointHoverRadius: 5,
        tension: 0.2,
      },
      {
        label: "20-Day SMA",
        data: stockData.sma20,
        fill: false,
        borderColor: "rgba(255, 206, 86, 1)",
        borderWidth: 1.5,
        pointRadius: 0,
        borderDash: [5, 5],
        tension: 0.4,
        hidden: selectedInterval === "real-time",
      },
      {
        label: "50-Day SMA",
        data: stockData.sma50,
        fill: false,
        borderColor: "rgba(153, 102, 255, 1)",
        borderWidth: 1.5,
        pointRadius: 0,
        borderDash: [3, 3],
        tension: 0.4,
        hidden: selectedInterval === "real-time",
      },
    ],
  };

  // Chart options
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: {
        ticks: {
          color: "#aaa",
          maxTicksLimit: 10,
        },
        grid: {
          color: "rgba(255, 255, 255, 0.05)",
        },
      },
      y: {
        ticks: {
          color: "#aaa",
        },
        grid: {
          color: "rgba(255, 255, 255, 0.05)",
        },
      },
    },
    plugins: {
      legend: {
        labels: {
          color: "#fff",
        },
      },
      tooltip: {
        mode: "index",
        intersect: false,
      },
    },
    interaction: {
      mode: "nearest",
      axis: "x",
      intersect: false,
    },
  };

  return (
    <div className="simulator-container" style={{ 
      margin: "auto", 
      padding: "20px", 
      textAlign: "center", 
      backgroundColor: "rgb(18,18,18)", 
      color: "#fff", 
      minHeight: "100vh" 
    }}>
      <h2 style={{ marginBottom: "20px" }}>Stock Market Simulator</h2>
      
      {/* Portfolio Summary */}
      <div className="portfolio-summary" style={{ 
        display: "flex", 
        justifyContent: "space-around", 
        backgroundColor: "#222", 
        padding: "15px", 
        borderRadius: "10px", 
        marginBottom: "20px" 
      }}>
        <div>
          <h3 style={{ color: "#aaa", fontSize: "14px", margin: "0" }}>Balance</h3>
          <p style={{ fontSize: "18px", margin: "5px 0" }}>
            {exchange === "US" ? "$" : "₹"}{balance.toFixed(2)}
          </p>
        </div>
        <div>
          <h3 style={{ color: "#aaa", fontSize: "14px", margin: "0" }}>Current Stock</h3>
          <p style={{ fontSize: "18px", margin: "5px 0" }}>
            {stockData.name || ticker} ({stockData.sector || "Unknown"})
          </p>
        </div>
        <div>
          <h3 style={{ color: "#aaa", fontSize: "14px", margin: "0" }}>Holdings</h3>
          <p style={{ fontSize: "18px", margin: "5px 0" }}>
            {holdings[ticker] || 0} shares
          </p>
        </div>
      </div>
      
      {/* Controls */}
      <div className="controls" style={{ 
        display: "flex", 
        gap: "10px", 
        justifyContent: "center", 
        marginBottom: "20px", 
        flexWrap: "wrap" 
      }}>
        <select 
          value={exchange} 
          onChange={(e) => setExchange(e.target.value)} 
          style={{ 
            padding: "10px", 
            fontSize: "16px", 
            borderRadius: "5px", 
            backgroundColor: "#222", 
            color: "#fff", 
            border: "1px solid #333" 
          }}
        >
          <option value="US">US Stock Exchange (USD)</option>
          <option value="IN">Indian Stock Exchange (INR)</option>
        </select>
        
        <input
          type="text"
          value={ticker}
          onChange={(e) => setTicker(e.target.value.toUpperCase())}
          placeholder="Enter Stock Symbol (e.g., AAPL)"
          style={{ 
            padding: "10px", 
            fontSize: "16px", 
            borderRadius: "5px", 
            backgroundColor: "#222", 
            color: "#fff", 
            border: "1px solid #333" 
          }}
        />
        
        <select 
          value={selectedInterval} 
          onChange={(e) => setSelectedInterval(e.target.value)} 
          style={{ 
            padding: "10px", 
            fontSize: "16px", 
            borderRadius: "5px", 
            backgroundColor: "#222", 
            color: "#fff", 
            border: "1px solid #333" 
          }}
        >
          <option value="real-time">Real-Time</option>
          <option value="1d">1 Day</option>
          <option value="5d">5 Days</option>
          <option value="1mo">1 Month</option>
        </select>
        
        <button 
          onClick={fetchStockData} 
          style={{ 
            padding: "10px 20px", 
            backgroundColor: "#4527a0", 
            color: "white", 
            borderRadius: "5px", 
            border: "none", 
            cursor: "pointer" 
          }}
        >
          {loading ? "Loading..." : "Refresh"}
        </button>
        
        <button 
          onClick={() => handleTrade("buy")} 
          disabled={loading || !stockData.prices.length || balance < stockData.prices[stockData.prices.length - 1]}
          style={{ 
            padding: "10px 20px", 
            backgroundColor: "#26a69a", 
            color: "white", 
            borderRadius: "5px", 
            border: "none", 
            cursor: "pointer", 
            opacity: loading || !stockData.prices.length || balance < stockData.prices[stockData.prices.length - 1] ? 0.6 : 1 
          }}
        >
          Buy
        </button>
        
        <button 
          onClick={() => handleTrade("sell")} 
          disabled={loading || !stockData.prices.length || !holdings[ticker]}
          style={{ 
            padding: "10px 20px", 
            backgroundColor: "#ef5350", 
            color: "white", 
            borderRadius: "5px", 
            border: "none", 
            cursor: "pointer", 
            opacity: loading || !stockData.prices.length || !holdings[ticker] ? 0.6 : 1 
          }}
        >
          Sell
        </button>
      </div>
      
      {/* Error Message */}
      {error && (
        <div style={{ 
          backgroundColor: "rgba(239, 83, 80, 0.2)", 
          color: "#ef5350", 
          padding: "10px", 
          borderRadius: "5px", 
          marginBottom: "20px" 
        }}>
          {error}
        </div>
      )}
      
      {/* Chart */}
      <div style={{ 
        display: "flex", 
        justifyContent: "center", 
        alignItems: "center", 
        height: "500px", 
        backgroundColor: "rgb(30,30,30)", 
        padding: "10px", 
        borderRadius: "10px", 
        boxShadow: "0 4px 6px rgba(0, 0, 0, 0.1)" 
      }}>
        {loading ? (
          <div>Loading stock data...</div>
        ) : stockData.prices.length > 0 ? (
          <div style={{ width: "100%", height: "100%" }}>
            <Line data={chartData} options={chartOptions} />
          </div>
        ) : (
          <div>No data available for {ticker}</div>
        )}
      </div>
      
      {/* Transaction History */}
      <h3 style={{ marginTop: "30px" }}>Transaction History</h3>
      <div style={{ overflowX: "auto" }}>
        <table style={{ 
          width: "100%", 
          margin: "20px auto", 
          borderCollapse: "collapse", 
          color: "#fff" 
        }}>
          <thead>
            <tr style={{ borderBottom: "1px solid #333" }}>
              <th style={{ padding: "10px" }}>Type</th>
              <th style={{ padding: "10px" }}>Ticker</th>
              <th style={{ padding: "10px" }}>Price</th>
              <th style={{ padding: "10px" }}>Quantity</th>
              <th style={{ padding: "10px" }}>Profit/Loss</th>
              <th style={{ padding: "10px" }}>Currency</th>
              <th style={{ padding: "10px" }}>Time</th>
            </tr>
          </thead>
          <tbody>
            {transactions.length > 0 ? (
              transactions.map((txn, index) => (
                <tr key={index} style={{ borderBottom: "1px solid #222" }}>
                  <td style={{ padding: "10px", color: txn.type === "Buy" ? "#26a69a" : "#ef5350" }}>{txn.type}</td>
                  <td style={{ padding: "10px" }}>{txn.ticker}</td>
                  <td style={{ padding: "10px" }}>{txn.price.toFixed(2)}</td>
                  <td style={{ padding: "10px" }}>{txn.quantity}</td>
                  <td style={{ 
                    padding: "10px", 
                    color: txn.profit > 0 ? "#26a69a" : txn.profit < 0 ? "#ef5350" : "#fff" 
                  }}>
                    {txn.profit.toFixed(2)}
                  </td>
                  <td style={{ padding: "10px" }}>{txn.currency}</td>
                  <td style={{ padding: "10px" }}>{new Date(txn.timestamp).toLocaleTimeString()}</td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan="7" style={{ padding: "20px", textAlign: "center" }}>No transactions yet</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default Simulator;
