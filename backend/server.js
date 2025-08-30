// combined-server.js

const express = require("express");
const http = require("http");
const cors = require("cors");
const axios = require("axios");
const { Server } = require("socket.io");
const { PythonShell } = require("python-shell");
const yahooFinance = require("yahoo-finance2").default; // Using a modern, well-maintained library

// --- INITIALIZE SERVERS AND MIDDLEWARE ---
const app = express();
const server = http.createServer(app); // Create a single HTTP server to be used by both Express and Socket.IO

// Configure CORS for both API and Socket.IO
app.use(cors());
app.use(express.json());

// Set up Socket.IO with the server instance
const io = new Server(server, {
  cors: { origin: "*" }, // Allow all origins for the socket connection
});

const PORT = 5000;

// --- SOCKET.IO LOGIC (MOVED FROM socket.js) ---

// Store active subscriptions and their intervals
const activeSubscriptions = new Map();
const stockCache = new Map();

// Helper to get realistic price movements
function getRealisticPriceMovement(basePrice) {
  const volatility = 0.002;
  const change = basePrice * volatility * (Math.random() * 2 - 1);
  return basePrice + change;
}

// Get initial stock data and cache it
async function fetchAndCacheStockData(ticker) {
  // Use cached data if it's recent enough
  if (stockCache.has(ticker) && Date.now() - stockCache.get(ticker).timestamp < 60000) {
    return stockCache.get(ticker).data;
  }

  try {
    const result = await yahooFinance.chart(ticker, { interval: '1m', range: '1d' });
    const quote = result.indicators.quote[0];
    const timestamps = result.timestamp || [];
    const prices = quote.close || [];
    const volumes = quote.volume || [];
    const currentPrice = result.meta.regularMarketPrice || prices[prices.length - 1] || 100;

    const data = {
      basePrice: currentPrice,
      latestPrice: currentPrice,
      timestamps,
      prices,
      volumes,
      meta: result.meta,
    };

    stockCache.set(ticker, {
      data,
      timestamp: Date.now(),
    });

    return data;
  } catch (error) {
    console.error(`Error fetching data for ${ticker}:`, error.message);
    return {
      basePrice: 100,
      latestPrice: 100,
      timestamps: [Math.floor(Date.now() / 1000)],
      prices: [100],
      volumes: [1000],
      meta: { currency: "USD", exchangeName: "Unknown" },
    };
  }
}

// Helper to generate news items
function generateMarketNews(ticker) {
  const companies = {
    AAPL: "Apple",
    MSFT: "Microsoft",
    GOOGL: "Google",
    AMZN: "Amazon",
    META: "Meta",
    TSLA: "Tesla",
    NFLX: "Netflix"
  };

  const events = [
    "announced new product line",
    "reported quarterly earnings",
    "CEO made a statement about future plans",
    "unveiled strategic partnership",
    "faces regulatory challenges",
    "stock upgraded by analysts",
    "stock downgraded by analysts",
    "plans expansion into new markets",
    "reported higher than expected revenue",
    "announced cost-cutting measures"
  ];

  const companyName = companies[ticker] || ticker;
  const event = events[Math.floor(Math.random() * events.length)];

  return {
    headline: `${companyName} ${event}`,
    source: ["Bloomberg", "CNBC", "Reuters", "Financial Times"][Math.floor(Math.random() * 4)],
    timestamp: new Date().toISOString(),
    sentiment: ["positive", "neutral", "negative"][Math.floor(Math.random() * 3)],
  };
}

io.on("connection", (socket) => {
  console.log("Client connected:", socket.id);
  let subscriptions = new Set();

  socket.on("subscribeStock", async (ticker) => {
    // ... (rest of the socket subscription logic)
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
      timestamp: new Date().toISOString()
    });

    if (activeSubscriptions.get(ticker).size === 1) {
      console.log(`Starting interval for ${ticker}`);
      const intervalId = setInterval(async () => {
        currentPrice = getRealisticPriceMovement(currentPrice);
        const includeNews = Math.random() < 0.05;
        const newsItem = includeNews ? generateMarketNews(ticker) : null;

        for (const clientId of activeSubscriptions.get(ticker)) {
          const clientSocket = io.sockets.sockets.get(clientId);
          if (clientSocket) {
            clientSocket.emit("stockUpdate", {
              ticker,
              price: currentPrice,
              timestamp: new Date().toISOString(),
              volume: Math.floor(Math.random() * 10000),
              news: newsItem,
            });
          }
        }
      }, 1000);

      activeSubscriptions.get(ticker).intervalId = intervalId;
    }
  });

  socket.on("unsubscribeStock", (ticker) => {
    // ... (rest of the unsubscription logic)
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
    // ... (rest of the disconnect cleanup logic)
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
    if (socket.newsIntervalId) {
      clearInterval(socket.newsIntervalId);
    }
  });
});

// --- REST API ROUTES (MOVED FROM server.js) ---

// Route to fetch real-time stock data
app.get("/api/stock/:ticker", async (req, res) => {
    try {
        const ticker = req.params.ticker.toUpperCase();
        const url = `https://query1.finance.yahoo.com/v8/finance/chart/${ticker}?interval=5m&range=5d`;
        
        const response = await axios.get(url);
        const result = response.data.chart.result[0];
        
        const timestamps = result.timestamp;
        const prices = result.indicators.quote[0];

        const stockData = timestamps.map((time, i) => ({
            time: new Date(time * 1000),
            open: prices.open[i],
            high: prices.high[i],
            low: prices.low[i],
            close: prices.close[i],
        }));

        res.json(stockData);
    } catch (error) {
        res.status(500).json({ error: "Error fetching stock data" });
    }
});

// Route to fetch historical stock data
app.get("/api/stock/:ticker/:range", async (req, res) => {
    try {
        const { ticker, range } = req.params;
        const validRanges = { "1d": "1m", "5d": "5m", "1mo": "1h", "1y": "1d" };
        if (!validRanges[range]) return res.status(400).json({ error: "Invalid range" });

        const url = `https://query1.finance.yahoo.com/v8/finance/chart/${ticker}?interval=${validRanges[range]}&range=${range}`;
        
        const response = await axios.get(url);
        const result = response.data.chart.result[0];
        
        const timestamps = result.timestamp;
        const prices = result.indicators.quote[0];

        const stockData = timestamps.map((time, i) => ({
            time: new Date(time * 1000),
            open: prices.open[i],
            high: prices.high[i],
            low: prices.low[i],
            close: prices.close[i],
        }));

        res.json(stockData);
    } catch (error) {
        res.status(500).json({ error: "Error fetching stock data" });
    }
});

// Route to predict stock price using PythonShell (assuming predict_stock.py is a simple script)
app.post("/api/predict", (req, res) => {
    const { ticker, closePrices } = req.body;
    if (!closePrices || closePrices.length < 100) {
        return res.status(400).json({ error: "Insufficient data" });
    }

    let options = {
        mode: "text",
        pythonOptions: ["-u"],
        scriptPath: "./",
        args: [ticker, JSON.stringify(closePrices)],
    };

    PythonShell.run("predict_stock.py", options, (err, results) => {
        if (err) return res.status(500).json({ error: err.message });
        res.json({ predictedPrice: parseFloat(results[0]) });
    });
});

// Route for market overview
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

// Start the server
server.listen(PORT, () => console.log(`Server running on port ${PORT}`));
