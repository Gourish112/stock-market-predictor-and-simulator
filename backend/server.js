const express = require("express");
const http = require("http");
const { Server } = require("socket.io");
const axios = require("axios");
const cors = require("cors");
const { PythonShell } = require("python-shell");

const app = express();
app.use(express.json());

// ✅ Fix CORS issue for all origins
const corsOptions = {
    origin: "*", 
    methods: ["GET", "POST"],
    allowedHeaders: ["Content-Type"],
};
app.use(cors(corsOptions));

// Create a single HTTP server for both Express and Socket.IO
const server = http.createServer(app);
const io = new Server(server, {
    cors: { origin: "*" },
});

const PORT = process.env.PORT || 5000;

// ✅ Caching and Subscription Logic for Real-time Data
// Maps to store cached stock data and active subscriptions
const stockCache = new Map();
const activeSubscriptions = new Map();

// ✅ Fetches and caches stock data to prevent rate-limiting
async function fetchAndCacheStockData(ticker) {
    // Check if data is in cache and is less than 60 seconds old
    if (stockCache.has(ticker) && (Date.now() - stockCache.get(ticker).timestamp < 60000)) {
        console.log(`✅ Returning cached data for ${ticker}`);
        return stockCache.get(ticker).data;
    }

    try {
        const response = await axios.get(
            `https://query1.finance.yahoo.com/v8/finance/chart/${ticker}?interval=1m&range=1d`
        );

        if (!response.data.chart.result || response.data.chart.result.length === 0) {
            throw new Error("No data found");
        }

        const latestPrice = response.data.chart.result[0].meta.regularMarketPrice;
        const previousClose = response.data.chart.result[0].meta.chartPreviousClose;
        const percentChange = ((latestPrice - previousClose) / previousClose) * 100;
        const name = response.data.chart.result[0].meta.symbol;

        const data = {
            latestPrice: latestPrice,
            change: latestPrice - previousClose,
            percentChange: percentChange,
            name: name
        };

        // Update the cache with new data and a timestamp
        stockCache.set(ticker, {
            data: data,
            timestamp: Date.now(),
        });
        console.log(`✅ Fetched and cached new data for ${ticker}`);
        return data;
    } catch (error) {
        console.error(`❌ Error fetching stock data for ${ticker}:`, error.message);
        throw new Error(`Error fetching stock data for ${ticker}: ${error.message}`);
    }
}

// REST API endpoint to fetch static stock data
app.get("/api/stock/:ticker", async (req, res) => {
    try {
        const ticker = req.params.ticker.toUpperCase();
        const data = await fetchAndCacheStockData(ticker);
        res.json(data);
    } catch (error) {
        res.status(502).json({ error: error.message });
    }
});

// Route to predict stock price using Python shell
app.post("/api/predict", (req, res) => {
    const { ticker, closePrices } = req.body;

    if (!closePrices || closePrices.length < 100) {
        return res.status(400).json({ error: "Insufficient data for prediction" });
    }

    let options = {
        mode: "text",
        pythonOptions: ["-u"],
        scriptPath: "./",
        args: [ticker, JSON.stringify(closePrices)],
    };

    PythonShell.run("app.py", options)
        .then((messages) => {
            const predictions = JSON.parse(messages[messages.length - 1]);
            res.json(predictions);
        })
        .catch((err) => {
            console.error("❌ Prediction Error:", err);
            res.status(500).json({ error: "Prediction failed due to internal server error" });
        });
});

// WebSocket connection logic for real-time updates
io.on("connection", (socket) => {
    console.log("✅ A user connected via WebSocket");

    // Listen for 'subscribe' event from client
    socket.on("subscribe", (ticker) => {
        console.log(`User subscribed to ${ticker}`);
        
        // Add the socket to the set of subscribers for this ticker
        if (!activeSubscriptions.has(ticker)) {
            // No one is subscribed yet, start a new interval
            const intervalId = setInterval(async () => {
                try {
                    const data = await fetchAndCacheStockData(ticker);
                    // Emit the update to all sockets in the ticker's room
                    io.to(ticker).emit("stockUpdate", data);
                } catch (error) {
                    io.to(ticker).emit("error", { ticker, message: error.message });
                }
            }, 5000); // 🚨 Increased interval to 5 seconds to reduce API calls and avoid rate limits
            activeSubscriptions.set(ticker, {
                intervalId,
                subscribers: new Set(),
            });
        }
        activeSubscriptions.get(ticker).subscribers.add(socket.id);
        socket.join(ticker); // Join a room for the ticker
    });

    // Listen for 'unsubscribe' event
    socket.on("unsubscribe", (ticker) => {
        if (activeSubscriptions.has(ticker)) {
            const subscription = activeSubscriptions.get(ticker);
            subscription.subscribers.delete(socket.id);
            socket.leave(ticker);
            
            // If no more subscribers, clear the interval
            if (subscription.subscribers.size === 0) {
                clearInterval(subscription.intervalId);
                activeSubscriptions.delete(ticker);
                console.log(`Stopping interval for ${ticker}`);
            }
        }
    });

    // Handle client disconnection
    socket.on("disconnect", () => {
        console.log("❌ User disconnected");
        // Remove the disconnected socket from all subscriptions
        activeSubscriptions.forEach((subscription, ticker) => {
            if (subscription.subscribers.has(socket.id)) {
                subscription.subscribers.delete(socket.id);
                if (subscription.subscribers.size === 0) {
                    clearInterval(subscription.intervalId);
                    activeSubscriptions.delete(ticker);
                    console.log(`Stopping interval for ${ticker}`);
                }
            }
        });
    });
});

// Start the server
server.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});
