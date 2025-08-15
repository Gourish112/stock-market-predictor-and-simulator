const express = require("express");
const http = require("http");
const { Server } = require("socket.io");
const axios = require("axios");
const cors = require("cors");
const allowedOrigin = process.env.CORS_ORIGIN;
const app = express();
app.use(cors({ origin: allowedOrigin }));
app.use(express.json());

const server = http.createServer(app);
const io = new Server(server, {
  cors: { origin: allowedOrigin }
});

// Store active subscriptions
const activeSubscriptions = new Map();

// Helper to get realistic price movements
function getRealisticPriceMovement(basePrice) {
  // Generate random movement with slight volatility
  const volatility = 0.002; // 0.2% volatility per update
  const change = basePrice * volatility * (Math.random() * 2 - 1);
  return basePrice + change;
}

// Cache for stock data to reduce API calls
const stockCache = new Map();

// Get initial stock data and cache it
async function fetchAndCacheStockData(ticker) {
  if (stockCache.has(ticker) && Date.now() - stockCache.get(ticker).timestamp < 60000) {
    return stockCache.get(ticker).data;
  }
  
  try {
    const response = await axios.get(
      `https://query1.finance.yahoo.com/v8/finance/chart/${ticker}?interval=1m&range=1d`
    );
    
    if (!response.data?.chart?.result?.[0]) {
      throw new Error("Invalid data format");
    }
    
    const result = response.data.chart.result[0];
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
      meta: result.meta
    };
    
    stockCache.set(ticker, {
      data,
      timestamp: Date.now()
    });
    
    return data;
  } catch (error) {
    console.error(`Error fetching data for ${ticker}:`, error.message);
    // Return fallback data
    return {
      basePrice: 100,
      latestPrice: 100,
      timestamps: [Math.floor(Date.now() / 1000)],
      prices: [100],
      volumes: [1000],
      meta: { currency: "USD", exchangeName: "Unknown" }
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
    sentiment: ["positive", "neutral", "negative"][Math.floor(Math.random() * 3)]
  };
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
    
    // Add to client's subscriptions
    subscriptions.add(ticker);
    
    // Initialize or update global subscription counter
    if (!activeSubscriptions.has(ticker)) {
      activeSubscriptions.set(ticker, new Set());
    }
    activeSubscriptions.get(ticker).add(socket.id);
    
    // Get initial data
    const stockData = await fetchAndCacheStockData(ticker);
    let currentPrice = stockData.latestPrice;
    
    // Send initial data
    socket.emit("stockData", {
      ticker,
      price: currentPrice,
      currency: stockData.meta.currency || "USD",
      exchange: stockData.meta.exchangeName || "Unknown",
      timestamp: new Date().toISOString()
    });
    
    // Only start interval if this is the first subscription for this ticker
    if (activeSubscriptions.get(ticker).size === 1) {
      console.log(`Starting interval for ${ticker}`);
      
      // Store interval reference for cleanup
      const intervalId = setInterval(async () => {
        // Generate a realistic price movement
        currentPrice = getRealisticPriceMovement(currentPrice);
        
        // Occasionally generate market news (5% chance per update)
        const includeNews = Math.random() < 0.05;
        const newsItem = includeNews ? generateMarketNews(ticker) : null;
        
        // Send to all subscribed clients
        for (const clientId of activeSubscriptions.get(ticker)) {
          const clientSocket = io.sockets.sockets.get(clientId);
          if (clientSocket) {
            clientSocket.emit("stockUpdate", {
              ticker,
              price: currentPrice,
              timestamp: new Date().toISOString(),
              volume: Math.floor(Math.random() * 10000),
              news: newsItem
            });
          }
        }
      }, 1000);
      
      // Store the interval ID with the ticker
      activeSubscriptions.get(ticker).intervalId = intervalId;
    }
  });
  
  // Handle unsubscription
  socket.on("unsubscribeStock", (ticker) => {
    console.log(`Client ${socket.id} unsubscribed from ${ticker}`);
    
    // Remove from client's subscriptions
    subscriptions.delete(ticker);
    
    // Update global subscription counter
    if (activeSubscriptions.has(ticker)) {
      activeSubscriptions.get(ticker).delete(socket.id);
      
      // If no more subscribers, clear the interval
      if (activeSubscriptions.get(ticker).size === 0) {
        console.log(`Stopping interval for ${ticker}`);
        clearInterval(activeSubscriptions.get(ticker).intervalId);
        activeSubscriptions.delete(ticker);
      }
    }
  });
  
  // Handle market news subscription
  socket.on("subscribeMarketNews", () => {
    console.log(`Client ${socket.id} subscribed to market news`);
    
    // Send random news every 10 seconds
    const newsIntervalId = setInterval(() => {
      // Generate a random market news
      const tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA"];
      const randomTicker = tickers[Math.floor(Math.random() * tickers.length)];
      
      socket.emit("marketNews", generateMarketNews(randomTicker));
    }, 10000);
    
    // Store interval ID for cleanup
    socket.newsIntervalId = newsIntervalId;
  });
  
  // Handle disconnect to clean up all resources
  socket.on("disconnect", () => {
    console.log(`Client ${socket.id} disconnected`);
    
    // Clean up all subscriptions for this client
    for (const ticker of subscriptions) {
      if (activeSubscriptions.has(ticker)) {
        activeSubscriptions.get(ticker).delete(socket.id);
        
        // If no more subscribers, clear the interval
        if (activeSubscriptions.get(ticker).size === 0) {
          console.log(`Stopping interval for ${ticker}`);
          clearInterval(activeSubscriptions.get(ticker).intervalId);
          activeSubscriptions.delete(ticker);
        }
      }
    }
    
    // Clear news interval if exists
    if (socket.newsIntervalId) {
      clearInterval(socket.newsIntervalId);
    }
  });
});

// REST API for static data
app.get('/api/market-overview', async (req, res) => {
  try {
    const indices = ["^GSPC", "^DJI", "^IXIC"]; // S&P 500, Dow Jones, NASDAQ
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
          change: (Math.random() * 2 - 1) * (data.latestPrice * 0.01), // Random change for demo
          percentChange: (Math.random() * 2 - 1) * 1.5 // Random % change
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
