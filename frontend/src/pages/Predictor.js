import React, { useState } from "react";
import StockChart from "../components/StockChart"; // Import StockChart component

const Predictor = () => {
    const [exchange, setExchange] = useState("US"); // Default stock exchange
    const [ticker, setTicker] = useState("AAPL"); // Default stock ticker
    const [timeframe, setTimeframe] = useState("1_day"); // Default prediction timeframe
    const [prediction, setPrediction] = useState(null);
    const [chartData, setChartData] = useState(null); // ✅ Store chart data
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const formatTicker = (ticker) => {
        if (exchange === "India") {
            return `${ticker}`; // NSE format (e.g., RELIANCE.NS, TCS.NS)
        }
        return ticker; // US format (e.g., AAPL, TSLA)
    };

    const handlePredict = async () => {
        setLoading(true);
        setError(null);
        setPrediction(null);
        setChartData(null);

        try {
            const formattedTicker = formatTicker(ticker);
            const response = await fetch("http://localhost:5000/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ ticker: formattedTicker }),
            });

            const data = await response.json();
            console.log("Prediction Response:", data);

            if (response.ok) {
                setPrediction(`Predicted Price for ${timeframe.replace("_", " ")}: ${exchange==='India'?(ticker.slice(-3)===".NS"?`₹`:`$`):ticker.slice(-3)!==".NS"?`$`:`₹`} ${data[timeframe].toFixed(2)}`);
                setChartData({
                    "1_day": data["1_day"],
                    "5_days": data["5_days"],
                    "1_month": data["1_month"],
                    "1_year": data["1_year"],
                });
            } else {
                setError(`Error: ${data.error || "Prediction failed"}`);
            }
        } catch (error) {
            console.error("API Error:", error);
            setError("Prediction failed.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div style={styles.container}>
            <h1 style={styles.heading}>Stock Price Predictor</h1>

            <div style={styles.inputContainer}>
                <select 
                    value={exchange} 
                    onChange={(e) => setExchange(e.target.value)} 
                    style={styles.dropdown}
                >
                    <option value="US">US Stock Exchange</option>
                    <option value="India">Indian Stock Exchange (NSE)</option>
                </select>

                <input 
                    type="text" 
                    value={ticker} 
                    onChange={(e) => setTicker(e.target.value.toUpperCase())}
                    placeholder="Enter stock ticker (e.g., AAPL or RELIANCE)"
                    style={styles.input}
                />

                <select 
                    value={timeframe} 
                    onChange={(e) => setTimeframe(e.target.value)} 
                    style={styles.dropdown}
                >
                    <option value="1_day">1 Day</option>
                    <option value="5_days">5 Days</option>
                    <option value="1_month">1 Month</option>
                    <option value="1_year">1 Year</option>
                </select>

                <button onClick={handlePredict} style={styles.button} disabled={loading}>
                    {loading ? "Predicting..." : "Predict"}
                </button>
            </div>

            {error && <h2 style={{ color: "red" }}>{error}</h2>}
            {prediction && <h2 style={styles.prediction}>{prediction}</h2>}

            {/* ✅ Show chart only when data is available */}
            {chartData && <StockChart data={chartData} ticker={ticker} />}
        </div>
    );
};

const styles = {
    container: { marginTop: "60px", padding: "20px", display: "flex", flexDirection: "column", alignItems: "center", backgroundColor: "#121212", minHeight: "100vh" },
    heading: { color: "#fff", fontSize: "28px", marginBottom: "20px" },
    inputContainer: { display: "flex", justifyContent: "center", alignItems: "center", gap: "10px", marginBottom: "20px" },
    input: { padding: "12px", width: "200px", borderRadius: "6px", border: "1px solid #555", backgroundColor: "#1E1E1E", color: "#fff", fontSize: "16px", textAlign: "center" },
    dropdown: { padding: "12px", borderRadius: "6px", border: "1px solid #555", backgroundColor: "#1E1E1E", color: "#fff", fontSize: "16px", cursor: "pointer" },
    button: { padding: "12px 20px", backgroundColor: "#007BFF", border: "none", borderRadius: "6px", color: "#fff", cursor: "pointer", fontSize: "16px", fontWeight: "bold" },
    prediction: { color: "#4CAF50", fontSize: "20px", marginBottom: "15px" },
};

export default Predictor;
