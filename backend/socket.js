// Rectified socket.js (combining all backend logic)
const express = require("express");
const http = require("http");
const { Server } = require("socket.io");
const axios = require("axios");
const cors = require("cors");
const yahooFinance = require('yahoo-finance2').default; // Use a dedicated library for better data fetching

const app = express();
app.use(cors());
app.use(express.json());

const server = http.createServer(app);
const io = new Server(server, {
  cors: { origin: "*" },
});

// Store active subscriptions
const activeSubscriptions = new Map();

// Helper to get realistic price movements
function getRealisticPriceMovement(basePrice) {
  const volatility = 0.002;
  const change = basePrice * volatility * (Math.random() * 2 - 1);
  return basePrice + change;
}

// Cache for stock data to reduce API calls
const stockCache = new Map();

// --- Integrated backend logic from simulator.py ---

// Helper function to calculate Simple Moving Averages
function calculateSMA(data, window) {
  const sma = [];
  for (let i = 0; i < data.length; i++) {
    if (i < window - 1) {
      sma.push(null);
    } else {
      const sum = data.slice(i - window + 1, i + 1).reduce((acc, val) => acc + val, 0);
      sma.push(sum / window);
    }
  }
  return sma;
}

// Fetch historical stock data and calculate technical indicators
async function getHistoricalData(ticker, interval, exchange = "US") {
  try {
    const periodMap = { "1d": "1d", "5d": "5d", "1mo": "1mo", "3mo": "3mo", "6mo": "6mo", "1y": "1y", "5y": "5y" };
    const intervalMap = { "1d": "5m", "5d": "15m", "1mo": "1h", "3mo": "1d", "6mo": "1d", "1y": "1d", "5y": "1wk" };

    if (!periodMap[interval]) {
      throw new Error("Invalid interval");
    }

    // Add .NS suffix for Indian stocks
    const fullTicker = (exchange === "IN" && !ticker.endsWith(".NS")) ? `${ticker}.NS` : ticker;

    const data = await yahooFinance.chart(fullTicker, { period1: periodMap[interval] });

    if (!data.indicators || !data.indicators.quote[0] || !data.indicators.quote[0].close) {
      throw new Error("No data available for this ticker");
    }

    const prices = data.indicators.quote[0].close;
    const timestamps = data.timestamp.map(ts => new Date(ts * 1000).toISOString());
    const volumes = data.indicators.quote[0].volume;

    const sma20 = calculateSMA(prices, 20);
    const sma50 = calculateSMA(prices, 50);

    // Get stock info using a dedicated lookup
    const info = await yahooFinance.quoteSummary(fullTicker, { modules: ["assetProfile", "price"] });
    const name = info.price?.shortName || fullTicker;
    const sector = info.assetProfile?.sector || "Unknown";
    const meta = info.price || {};

    return {
      ticker: fullTicker,
      prices,
      timestamps,
      volumes,
      sma_20: sma20,
      sma_50: sma50,
      name,
      sector,
      exchange: meta.exchangeName || "Unknown",
      currency: meta.currency || "USD"
    };

  } catch (error) {
    console.error(`Error in getHistoricalData for ${ticker}:`, error.message);
    throw new Error(`Failed to fetch data: ${error.message}`);
  }
}

// --- End of integrated backend logic ---

// Get initial stock data for real-time and cache it
async function fetchAndCacheStockData(ticker) {
  if (stockCache.has(ticker) && Date.now() - stockCache.get(ticker).timestamp < 60000) {
    return stockCache.get(ticker).data;
  }
  
  try {
    const result = await yahooFinance.chart(ticker, { interval: '1m', range: '1d' });
    
    if (!result.indicators?.quote?.[0]) {
      throw new Error("Invalid data format");
    }
    
    const quote = result.indicators.quote[0];
    const timestamps = result.timestamp || [];
    const prices = quote.close || [];
    
    const currentPrice = result.meta.regularMarketPrice || prices[prices.length - 1] || 100;
    
    const data = {
      basePrice: currentPrice,
      latestPrice: currentPrice,
      timestamps,
      prices,
      meta: result.meta
    };
    
    stockCache.set(ticker, {
      data,
      timestamp: Date.now()
    });
    
    return data;
  } catch (error) {
    console.error(`Error fetching data for ${ticker}:`, error.message);
    return {
      basePrice: 100,
      latestPrice: 100,
      timestamps: [Math.floor(Date.now() / 1000)],
      prices: [100],
      meta: { currency: "USD", exchangeName: "Unknown" }
    };
  }
}

io.on("connection", (socket) => {
  console.log("Client connected:", socket.id);
  let subscriptions = new Set();
  
  // Handle subscription to stock updates
  socket.on("subscribeStock", async (ticker) => {
    if (!ticker || typeof ticker !== 'string') {
      socket.emit("error", { message: "Invalid ticker provided" });
      return;
    }
    
    console.log(`Client ${socket.id} subscribed to ${ticker}`);
    
    subscriptions.add(ticker);
    
    if (!activeSubscriptions.has(ticker)) {
      activeSubscriptions.set(ticker, new Set());
    }
    activeSubscriptions.get(ticker).add(socket.id);
    
    const stockData = await fetchAndCacheStockData(ticker);
    let currentPrice = stockData.latestPrice;
    
    socket.emit("stockData", {
      ticker,
      price: currentPrice,
      currency: stockData.meta.currency || "USD",
      exchange: stockData.meta.exchangeName || "Unknown",
      timestamp: new Date().toISOString(),
      name: stockData.meta.shortName || ticker,
      prices: stockData.prices,
      timestamps: stockData.timestamps,
    });
    
    if (activeSubscriptions.get(ticker).size === 1) {
      console.log(`Starting interval for ${ticker}`);
      
      const intervalId = setInterval(() => {
        currentPrice = getRealisticPriceMovement(currentPrice);
        
        for (const clientId of activeSubscriptions.get(ticker)) {
          const clientSocket = io.sockets.sockets.get(clientId);
          if (clientSocket) {
            clientSocket.emit("stockUpdate", {
              ticker,
              price: currentPrice,
              timestamp: new Date().toISOString(),
              volume: Math.floor(Math.random() * 10000),
            });
          }
        }
      }, 1000);
      
      activeSubscriptions.get(ticker).intervalId = intervalId;
    }
  });
  
  socket.on("unsubscribeStock", (ticker) => {
    console.log(`Client ${socket.id} unsubscribed from ${ticker}`);
    
    subscriptions.delete(ticker);
    
    if (activeSubscriptions.has(ticker)) {
      activeSubscriptions.get(ticker).delete(socket.id);
      
      if (activeSubscriptions.get(ticker).size === 0) {
        console.log(`Stopping interval for ${ticker}`);
        clearInterval(activeSubscriptions.get(ticker).intervalId);
        activeSubscriptions.delete(ticker);
      }
    }
  });
  
  socket.on("disconnect", () => {
    console.log(`Client ${socket.id} disconnected`);
    
    for (const ticker of subscriptions) {
      if (activeSubscriptions.has(ticker)) {
        activeSubscriptions.get(ticker).delete(socket.id);
        
        if (activeSubscriptions.get(ticker).size === 0) {
          console.log(`Stopping interval for ${ticker}`);
          clearInterval(activeSubscriptions.get(ticker).intervalId);
          activeSubscriptions.delete(ticker);
        }
      }
    }
  });
});

// REST API for historical data simulation
app.post('/simulate', async (req, res) => {
  try {
    const data = req.body;
    const { ticker, interval, exchange } = data;
    
    const result = await getHistoricalData(ticker, interval, exchange);
    
    return res.json(result);
  } catch (error) {
    console.error("Simulation Error:", error);
    res.status(500).json({ error: error.message });
  }
});

// REST API for static data
app.get('/api/market-overview', async (req, res) => {
  try {
    const indices = ["^GSPC", "^DJI", "^IXIC"];
    const results = {};
    
    for (const idx of indices) {
      try {
        const data = await fetchAndCacheStockData(idx);
        results[idx] = {
          name: {
            "^GSPC": "S&P 500",
            "^DJI": "Dow Jones",
            "^IXIC": "NASDAQ"
          }[idx] || idx,
          price: data.latestPrice,
          change: (Math.random() * 2 - 1) * (data.latestPrice * 0.01),
          percentChange: (Math.random() * 2 - 1) * 1.5
        };
      } catch (error) {
        console.error(`Error with index ${idx}:`, error);
      }
    }
    
    res.json(results);
  } catch (error) {
    console.error("Market overview error:", error);
    res.status(500).json({ error: "Failed to fetch market overview" });
  }
});

server.listen(5000, () => console.log("Server running on port 5000"));
