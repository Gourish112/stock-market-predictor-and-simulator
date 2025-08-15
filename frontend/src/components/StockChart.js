import React, { useState, useEffect } from "react";
import {
    Chart as ChartJS,
    LineElement,
    PointElement,
    LinearScale,
    CategoryScale,
    Title,
    Tooltip,
    Legend
} from "chart.js";
import { Line } from "react-chartjs-2";

ChartJS.register(LineElement, PointElement, LinearScale, CategoryScale, Title, Tooltip, Legend);
const API_URL = process.env.REACT_APP_API_URL;
const StockChart = ({ ticker }) => {
    const [chartData, setChartData] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchStockData = async () => {
            setLoading(true);
            try {
                const response = await fetch(`${API_URL}/api/predict`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ ticker }),
                });
    
                const data = await response.json();
                console.log("API Response:", data); // Debugging log
    
                if (!data || typeof data !== "object") {
                    console.error("API returned invalid data:", data);
                    setChartData(null);
                    return;
                }
    
                // Convert API response into chart-friendly format
                const labels = ["1 Day", "5 Days", "1 Month", "1 Year"];
                const prices = [data["1_day"], data["5_days"], data["1_month"], data["1_year"]];
    
                setChartData({
                    labels,
                    datasets: [
                        {
                            label: `${ticker} Predicted Price`,
                            data: prices,
                            borderColor: "#4CAF50",
                            backgroundColor: "rgba(76, 175, 80, 0.1)",
                            pointRadius: 5,
                            pointHoverRadius: 8,
                            borderWidth: 2,
                            fill: true,
                            tension: 0.4, // Smooth curve
                        },
                    ],
                });
            } catch (error) {
                console.error("Error fetching stock data:", error);
                setChartData(null);
            }
            setLoading(false);
        };
    
        fetchStockData();
    }, [ticker]);

    const chartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: true,
                labels: {
                    color: "#ffffff",
                    font: {
                        size: 14,
                    },
                },
            },
        },
        scales: {
            x: {
                ticks: { color: "#ffffff" },
                grid: { color: "rgba(255, 255, 255, 0.1)" },
            },
            y: {
                ticks: { color: "#ffffff" },
                grid: { color: "rgba(255, 255, 255, 0.1)" },
            },
        },
    };

    return (
        <div style={{
            backgroundColor: "#121212",
            padding: "20px",
            borderRadius: "10px",
            boxShadow: "0px 4px 10px rgba(0,0,0,0.2)",
            width: "80%",
            margin: "20px auto",
            minHeight: "400px",
        }}>
            {loading ? (
                <h2 style={{ color: "#ffffff", textAlign: "center" }}>Loading...</h2>
            ) : chartData ? (
                <Line data={chartData} options={chartOptions} />
            ) : (
                <h2 style={{ color: "#ff4444", textAlign: "center" }}>No Data Available</h2>
            )}
        </div>
    );
};

export default StockChart;
