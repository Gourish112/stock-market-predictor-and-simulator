const mongoose = require("mongoose");

const TransactionSchema = new mongoose.Schema({
  userId: { type: String, required: true },
  stock: { type: String, required: true },
  type: { type: String, enum: ["buy", "sell"], required: true },
  quantity: { type: Number, required: true },
  price: { type: Number, required: true },
  timestamp: { type: Date, default: Date.now },
});

const PortfolioSchema = new mongoose.Schema({
  userId: { type: String, required: true },
  balance: { type: Number, default: 100000 },
  holdings: [
    {
      stock: String,
      quantity: Number,
      averagePrice: Number,
    },
  ],
});

const Transaction = mongoose.model("Transaction", TransactionSchema);
const Portfolio = mongoose.model("Portfolio", PortfolioSchema);

module.exports = { Transaction, Portfolio };
