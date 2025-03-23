const express = require("express");
const http = require("http");
const { Server } = require("socket.io");
const axios = require("axios");

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  cors: { origin: "*" },
});

io.on("connection", (socket) => {
  console.log("Client connected");

  socket.on("subscribeStock", async (ticker) => {
    console.log(`Subscribed to ${ticker}`);

    // Send updates every second
    setInterval(async () => {
      try {
        const response = await axios.get(`https://query1.finance.yahoo.com/v8/finance/chart/${ticker}`);
        const price = response.data.chart.result[0].meta.regularMarketPrice;
        socket.emit("stockUpdate", { ticker, price, timestamp: new Date().toISOString() });
      } catch (error) {
        console.error("Error fetching price:", error);
      }
    }, 1000);
  });

  socket.on("disconnect", () => console.log("Client disconnected"));
});

server.listen(5000, () => console.log("Server running on port 5000"));
