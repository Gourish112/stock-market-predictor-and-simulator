const express = require("express");
const cors = require("cors");
const axios = require("axios");
const { PythonShell } = require("python-shell");

const app = express();
app.use(express.json());

// ✅ Fix CORS issue
const corsOptions = {
    origin: "http://localhost:3000", // Allow frontend requests
    methods: "GET,POST",
    allowedHeaders: "Content-Type"
};
app.use(cors(corsOptions));

const PORT = 5000;

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

// Route to fetch historical stock data for different time ranges
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

// Route to predict stock price
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

app.listen(PORT, () => console.log(`Server running on port ${PORT}`));