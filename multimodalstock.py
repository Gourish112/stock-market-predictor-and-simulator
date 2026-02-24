"""
Advanced Multi-Modal Stock Price Prediction System
====================================================
Production-grade implementation combining market data, technical indicators,
sentiment analysis, and multi-model ensemble for multi-horizon forecasting.

Author: Quantitative Finance AI Research Team
Version: 1.0.0
GPU-Optimized for RTX 3050 (6GB VRAM)
"""

# ============================================================================
# IMPORTS AND DEPENDENCIES
# ============================================================================

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import pickle
import json
from datetime import datetime, timedelta
import logging

# Data fetching
import yfinance as yf
import requests

# Technical analysis
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice

# Machine Learning - Classical
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge

# Deep Learning - PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# NLP - Transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import scipy.stats as stats

# Visualization (optional, for analysis)
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# GPU Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for the entire prediction system"""
    
    # Data Parameters
    lookback_period: int = 252  # Trading days (~1 year)
    prediction_horizons: List[int] = None  # [1, 5, 21, 252] days
    train_test_split: float = 0.7
    
    # Technical Indicator Windows
    sma_windows: List[int] = None  # [5, 10, 20, 50, 200]
    ema_windows: List[int] = None  # [5, 10, 20, 50, 200]
    rsi_window: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_window: int = 20
    atr_window: int = 14
    
    # Statistical Rolling Windows
    rolling_windows: List[int] = None  # [5, 10, 21, 63]
    
    # Model Architecture
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 3
    lstm_dropout: float = 0.2
    
    transformer_d_model: int = 128
    transformer_nhead: int = 8
    transformer_num_layers: int = 4
    transformer_dropout: float = 0.1
    
    xgb_n_estimators: int = 500
    xgb_max_depth: int = 7
    xgb_learning_rate: float = 0.01
    
    sentiment_embedding_dim: int = 768
    sentiment_hidden_dim: int = 256
    
    # Training Parameters
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    early_stopping_patience: int = 15
    
    # Ensemble Weights
    lstm_weight: float = 0.3
    transformer_weight: float = 0.3
    xgb_weight: float = 0.25
    sentiment_weight: float = 0.15
    
    def __post_init__(self):
        if self.prediction_horizons is None:
            self.prediction_horizons = [1, 5, 21, 252]
        if self.sma_windows is None:
            self.sma_windows = [5, 10, 20, 50, 200]
        if self.ema_windows is None:
            self.ema_windows = [5, 10, 20, 50, 200]
        if self.rolling_windows is None:
            self.rolling_windows = [5, 10, 21, 63]

config = ModelConfig()

# ============================================================================
# DATA FETCHING MODULE
# ============================================================================

class MarketDataFetcher:
    """Fetch and preprocess market data from Yahoo Finance"""
    
    def __init__(self, symbol: str, period: str = "5y", interval: str = "1d"):
        """
        Initialize market data fetcher
        
        Args:
            symbol: Stock ticker symbol
            period: Historical period to fetch (e.g., '5y', '10y')
            interval: Data interval (e.g., '1d', '1h')
        """
        self.symbol = symbol
        self.period = period
        self.interval = interval
        self.data: Optional[pd.DataFrame] = None
        
    def fetch_data(self) -> pd.DataFrame:
        """Fetch OHLCV data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(period=self.period, interval=self.interval)
            
            if self.data.empty:
                raise ValueError(f"No data found for symbol: {self.symbol}")
            
            # Ensure proper datetime index
            self.data.index = pd.to_datetime(self.data.index)
            self.data.sort_index(inplace=True)
            
            # Remove timezone info if present
            if self.data.index.tz is not None:
                self.data.index = self.data.index.tz_localize(None)
            
            logger.info(f"Fetched {len(self.data)} rows for {self.symbol}")
            return self.data
            
        except Exception as e:
            logger.error(f"Error fetching data for {self.symbol}: {str(e)}")
            raise
    
    def compute_returns(self) -> pd.DataFrame:
        """Compute simple and log returns"""
        if self.data is None:
            raise ValueError("Must fetch data first")
        
        self.data['simple_return'] = self.data['Close'].pct_change()
        self.data['log_return'] = np.log(self.data['Close'] / self.data['Close'].shift(1))
        
        return self.data

# ============================================================================
# TECHNICAL INDICATORS MODULE
# ============================================================================

class TechnicalIndicators:
    """Compute comprehensive technical indicators"""
    
    @staticmethod
    def compute_all_indicators(df: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
        """
        Compute all technical indicators
        
        Args:
            df: DataFrame with OHLCV data
            config: Model configuration
            
        Returns:
            DataFrame with added indicator columns
        """
        data = df.copy()
        
        # Moving Averages
        for window in config.sma_windows:
            data[f'SMA_{window}'] = SMAIndicator(close=data['Close'], window=window).sma_indicator()
            
        for window in config.ema_windows:
            data[f'EMA_{window}'] = EMAIndicator(close=data['Close'], window=window).ema_indicator()
        
        # RSI
        data['RSI'] = RSIIndicator(close=data['Close'], window=config.rsi_window).rsi()
        
        # MACD
        macd = MACD(
            close=data['Close'],
            window_fast=config.macd_fast,
            window_slow=config.macd_slow,
            window_sign=config.macd_signal
        )
        data['MACD'] = macd.macd()
        data['MACD_signal'] = macd.macd_signal()
        data['MACD_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = BollingerBands(close=data['Close'], window=config.bb_window, window_dev=2)
        data['BB_upper'] = bb.bollinger_hband()
        data['BB_middle'] = bb.bollinger_mavg()
        data['BB_lower'] = bb.bollinger_lband()
        data['BB_width'] = bb.bollinger_wband()
        data['BB_pct'] = bb.bollinger_pband()
        
        # ATR
        data['ATR'] = AverageTrueRange(
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            window=config.atr_window
        ).average_true_range()
        
        # Stochastic Oscillator
        stoch = StochasticOscillator(
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            window=14,
            smooth_window=3
        )
        data['STOCH_k'] = stoch.stoch()
        data['STOCH_d'] = stoch.stoch_signal()
        
        # OBV
        data['OBV'] = OnBalanceVolumeIndicator(close=data['Close'], volume=data['Volume']).on_balance_volume()
        
        # VWAP
        data['VWAP'] = VolumeWeightedAveragePrice(
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            volume=data['Volume']
        ).volume_weighted_average_price()
        
        # Price-based features
        data['High_Low_Pct'] = (data['High'] - data['Low']) / data['Close']
        data['Close_Open_Pct'] = (data['Close'] - data['Open']) / data['Open']
        
        logger.info(f"Computed technical indicators. Total features: {len(data.columns)}")
        return data
    
    @staticmethod
    def compute_rolling_statistics(df: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
        """
        Compute rolling statistical features
        
        Args:
            df: DataFrame with price data
            config: Model configuration
            
        Returns:
            DataFrame with added statistical features
        """
        data = df.copy()
        
        for window in config.rolling_windows:
            # Returns-based statistics
            data[f'rolling_mean_{window}'] = data['simple_return'].rolling(window=window).mean()
            data[f'rolling_std_{window}'] = data['simple_return'].rolling(window=window).std()
            data[f'rolling_skew_{window}'] = data['simple_return'].rolling(window=window).skew()
            data[f'rolling_kurt_{window}'] = data['simple_return'].rolling(window=window).kurt()
            
            # Price-based statistics
            data[f'rolling_max_{window}'] = data['Close'].rolling(window=window).max()
            data[f'rolling_min_{window}'] = data['Close'].rolling(window=window).min()
            
            # Drawdown
            rolling_max = data['Close'].rolling(window=window, min_periods=1).max()
            data[f'drawdown_{window}'] = (data['Close'] - rolling_max) / rolling_max
            
            # Momentum
            data[f'momentum_{window}'] = data['Close'].pct_change(periods=window)
        
        logger.info(f"Computed rolling statistics. Total features: {len(data.columns)}")
        return data

# ============================================================================
# SENTIMENT ANALYSIS MODULE
# ============================================================================

class SentimentAnalyzer:
    """Financial sentiment analysis using FinBERT"""
    
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """
        Initialize sentiment analyzer
        
        Args:
            model_name: Hugging Face model identifier
        """
        self.model_name = model_name
        self.device = device
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Loaded sentiment model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading sentiment model: {str(e)}")
            raise
    
    def analyze_sentiment(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Analyze sentiment of text list
        
        Args:
            texts: List of text strings to analyze
            
        Returns:
            List of sentiment dictionaries with scores
        """
        sentiments = []
        
        for text in texts:
            try:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = F.softmax(outputs.logits, dim=-1)
                
                # FinBERT outputs: [positive, negative, neutral]
                sentiment_score = probs[0][0].item() - probs[0][1].item()  # positive - negative
                
                sentiments.append({
                    'positive': probs[0][0].item(),
                    'negative': probs[0][1].item(),
                    'neutral': probs[0][2].item(),
                    'compound': sentiment_score
                })
                
            except Exception as e:
                logger.warning(f"Error analyzing sentiment: {str(e)}")
                sentiments.append({'positive': 0.33, 'negative': 0.33, 'neutral': 0.34, 'compound': 0.0})
        
        return sentiments
    
    def fetch_news_headlines(self, symbol: str, days: int = 30) -> List[Dict[str, Any]]:
        """
        Fetch news headlines for a stock symbol
        
        Args:
            symbol: Stock ticker symbol
            days: Number of days to look back
            
        Returns:
            List of news articles with metadata
        """
        articles = []
        
        try:
            # Using NewsAPI (free tier - requires API key)
            # Alternative: Use yfinance news or other free sources
            
            # For demo purposes, we'll use a placeholder
            # In production, integrate with NewsAPI, Alpha Vantage, or similar
            
            # Example using yfinance news
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            for item in news[:20]:  # Limit to recent 20 articles
                articles.append({
                    'title': item.get('title', ''),
                    'publisher': item.get('publisher', ''),
                    'link': item.get('link', ''),
                    'timestamp': datetime.fromtimestamp(item.get('providerPublishTime', 0))
                })
            
            logger.info(f"Fetched {len(articles)} news articles for {symbol}")
            
        except Exception as e:
            logger.warning(f"Error fetching news: {str(e)}")
        
        return articles
    
    def compute_sentiment_features(
        self,
        symbol: str,
        end_date: datetime,
        windows: List[int] = [1, 5, 21]
    ) -> Dict[str, float]:
        """
        Compute aggregated sentiment features over multiple time windows
        
        Args:
            symbol: Stock ticker symbol
            end_date: End date for sentiment analysis
            windows: List of day windows to aggregate over
            
        Returns:
            Dictionary of sentiment features
        """
        features = {}
        
        # Fetch recent news
        articles = self.fetch_news_headlines(symbol, days=max(windows))
        
        if not articles:
            # Return neutral sentiment if no articles
            for window in windows:
                features[f'sentiment_mean_{window}d'] = 0.0
                features[f'sentiment_std_{window}d'] = 0.0
                features[f'sentiment_positive_{window}d'] = 0.33
                features[f'sentiment_negative_{window}d'] = 0.33
            return features
        
        # Analyze sentiment for all articles
        texts = [article['title'] for article in articles]
        sentiments = self.analyze_sentiment(texts)
        
        # Add sentiment to articles
        for article, sentiment in zip(articles, sentiments):
            article['sentiment'] = sentiment
        
        # Compute features for each window
        for window in windows:
            cutoff_date = end_date - timedelta(days=window)
            
            # Filter articles within window
            window_articles = [
                a for a in articles
                if a['timestamp'] >= cutoff_date and a['timestamp'] <= end_date
            ]
            
            if window_articles:
                compounds = [a['sentiment']['compound'] for a in window_articles]
                positives = [a['sentiment']['positive'] for a in window_articles]
                negatives = [a['sentiment']['negative'] for a in window_articles]
                
                features[f'sentiment_mean_{window}d'] = np.mean(compounds)
                features[f'sentiment_std_{window}d'] = np.std(compounds)
                features[f'sentiment_positive_{window}d'] = np.mean(positives)
                features[f'sentiment_negative_{window}d'] = np.mean(negatives)
            else:
                features[f'sentiment_mean_{window}d'] = 0.0
                features[f'sentiment_std_{window}d'] = 0.0
                features[f'sentiment_positive_{window}d'] = 0.33
                features[f'sentiment_negative_{window}d'] = 0.33
        
        return features

# ============================================================================
# FEATURE ENGINEERING MODULE
# ============================================================================

class FeatureEngineer:
    """Comprehensive feature engineering pipeline"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.feature_scaler = RobustScaler()
        self.target_scaler = StandardScaler()
        self.feature_names: List[str] = []
        
    def create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variables for multi-horizon prediction
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with target columns added
        """
        data = df.copy()
        
        for horizon in self.config.prediction_horizons:
            # Future price (shifted backward for alignment)
            data[f'target_{horizon}d'] = data['Close'].shift(-horizon)
            
            # Future return
            data[f'target_return_{horizon}d'] = (
                data[f'target_{horizon}d'] / data['Close'] - 1
            )
        
        return data
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        sentiment_features: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare final feature matrix
        
        Args:
            df: DataFrame with all computed features
            sentiment_features: Optional DataFrame with sentiment features
            
        Returns:
            Tuple of (feature DataFrame, feature names list)
        """
        data = df.copy()
        
        # Exclude non-feature columns
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
        exclude_cols += [col for col in data.columns if col.startswith('target_')]
        
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        # Add sentiment features if provided
        if sentiment_features is not None:
            data = pd.concat([data, sentiment_features], axis=1)
            feature_cols += list(sentiment_features.columns)
        
        # Remove any remaining NaN rows (from indicator calculations)
        data = data.replace([np.inf, -np.inf], np.nan)
        
        self.feature_names = feature_cols
        
        return data, feature_cols
    
    def create_sequences(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        sequence_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time-series models
        
        Args:
            features: Feature matrix
            targets: Target matrix
            sequence_length: Length of input sequences
            
        Returns:
            Tuple of (X_sequences, y_targets)
        """
        X, y = [], []
        
        for i in range(sequence_length, len(features)):
            X.append(features[i-sequence_length:i])
            y.append(targets[i])
        
        return np.array(X), np.array(y)

# ============================================================================
# DEEP LEARNING MODELS
# ============================================================================

class LSTMPredictor(nn.Module):
    """Multi-layer LSTM for time-series prediction"""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float = 0.2
    ):
        super(LSTMPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, sequence, features)
            
        Returns:
            Output predictions of shape (batch, output_size)
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last hidden state
        last_hidden = h_n[-1]
        
        # Fully connected layers
        output = self.fc(last_hidden)
        
        return output


class TransformerPredictor(nn.Module):
    """Temporal Transformer Encoder for time-series prediction"""
    
    def __init__(
        self,
        input_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        output_size: int,
        dropout: float = 0.1
    ):
        super(TransformerPredictor, self).__init__()
        
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, sequence, features)
            
        Returns:
            Output predictions of shape (batch, output_size)
        """
        # Project input to d_model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        transformer_out = self.transformer_encoder(x)
        
        # Use mean pooling over sequence
        pooled = transformer_out.mean(dim=1)
        
        # Output layer
        output = self.fc(pooled)
        
        return output


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SentimentEncoder(nn.Module):
    """MLP for processing sentiment embeddings"""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.2
    ):
        super(SentimentEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# ============================================================================
# PYTORCH DATASET
# ============================================================================

class StockDataset(Dataset):
    """PyTorch Dataset for time-series stock data"""
    
    def __init__(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
        sentiment_features: Optional[np.ndarray] = None
    ):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
        self.sentiment_features = (
            torch.FloatTensor(sentiment_features)
            if sentiment_features is not None
            else None
        )
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        if self.sentiment_features is not None:
            return (
                self.sequences[idx],
                self.sentiment_features[idx],
                self.targets[idx]
            )
        else:
            return self.sequences[idx], self.targets[idx]


# ============================================================================
# ENSEMBLE MODEL
# ============================================================================

class EnsembleStockPredictor:
    """Multi-model ensemble combining LSTM, Transformer, XGBoost, and Sentiment"""
    
    def __init__(self, config: ModelConfig, num_features: int, num_sentiment_features: int):
        self.config = config
        self.num_features = num_features
        self.num_sentiment_features = num_sentiment_features
        self.device = device
        
        # Initialize models
        self.lstm_model = LSTMPredictor(
            input_size=num_features,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            output_size=len(config.prediction_horizons),
            dropout=config.lstm_dropout
        ).to(self.device)
        
        self.transformer_model = TransformerPredictor(
            input_size=num_features,
            d_model=config.transformer_d_model,
            nhead=config.transformer_nhead,
            num_layers=config.transformer_num_layers,
            output_size=len(config.prediction_horizons),
            dropout=config.transformer_dropout
        ).to(self.device)
        
        self.sentiment_encoder = SentimentEncoder(
            input_size=num_sentiment_features,
            hidden_size=config.sentiment_hidden_dim,
            output_size=len(config.prediction_horizons),
            dropout=0.2
        ).to(self.device)
        
        # XGBoost models (one per horizon)
        self.xgb_models = []
        for _ in config.prediction_horizons:
            self.xgb_models.append(
                xgb.XGBRegressor(
                    n_estimators=config.xgb_n_estimators,
                    max_depth=config.xgb_max_depth,
                    learning_rate=config.xgb_learning_rate,
                    tree_method='gpu_hist' if torch.cuda.is_available() else 'hist',
                    random_state=42
                )
            )
        
        # Meta-learner (stacking)
        self.meta_learner = Ridge(alpha=1.0)
        
        # Optimizers
        self.lstm_optimizer = optim.Adam(
            self.lstm_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.transformer_optimizer = optim.Adam(
            self.transformer_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.sentiment_optimizer = optim.Adam(
            self.sentiment_encoder.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.criterion = nn.MSELoss()
    
    def train_deep_models(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int
    ) -> Dict[str, List[float]]:
        """Train LSTM, Transformer, and Sentiment models"""
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'lstm_loss': [],
            'transformer_loss': [],
            'sentiment_loss': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training phase
            self.lstm_model.train()
            self.transformer_model.train()
            self.sentiment_encoder.train()
            
            train_loss = 0.0
            lstm_loss_epoch = 0.0
            transformer_loss_epoch = 0.0
            sentiment_loss_epoch = 0.0
            
            for batch in train_loader:
                sequences, sentiment_feats, targets = batch
                sequences = sequences.to(self.device)
                sentiment_feats = sentiment_feats.to(self.device)
                targets = targets.to(self.device)
                
                # LSTM forward + backward
                self.lstm_optimizer.zero_grad()
                lstm_pred = self.lstm_model(sequences)
                lstm_loss = self.criterion(lstm_pred, targets)
                lstm_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.lstm_model.parameters(), 1.0)
                self.lstm_optimizer.step()
                
                # Transformer forward + backward
                self.transformer_optimizer.zero_grad()
                transformer_pred = self.transformer_model(sequences)
                transformer_loss = self.criterion(transformer_pred, targets)
                transformer_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.transformer_model.parameters(), 1.0)
                self.transformer_optimizer.step()
                
                # Sentiment forward + backward
                self.sentiment_optimizer.zero_grad()
                sentiment_pred = self.sentiment_encoder(sentiment_feats)
                sentiment_loss = self.criterion(sentiment_pred, targets)
                sentiment_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.sentiment_encoder.parameters(), 1.0)
                self.sentiment_optimizer.step()
                
                # Accumulate losses
                batch_loss = lstm_loss.item() + transformer_loss.item() + sentiment_loss.item()
                train_loss += batch_loss
                lstm_loss_epoch += lstm_loss.item()
                transformer_loss_epoch += transformer_loss.item()
                sentiment_loss_epoch += sentiment_loss.item()
            
            train_loss /= len(train_loader)
            lstm_loss_epoch /= len(train_loader)
            transformer_loss_epoch /= len(train_loader)
            sentiment_loss_epoch /= len(train_loader)
            
            # Validation phase
            val_loss = self.evaluate(val_loader)
            
            # Record history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['lstm_loss'].append(lstm_loss_epoch)
            history['transformer_loss'].append(transformer_loss_epoch)
            history['sentiment_loss'].append(sentiment_loss_epoch)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best models
                self.save_models('best_models')
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
            
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                )
        
        return history
    
    def evaluate(self, data_loader: DataLoader) -> float:
        """Evaluate models on validation data"""
        self.lstm_model.eval()
        self.transformer_model.eval()
        self.sentiment_encoder.eval()
        if len(data_loader) == 0:
            logger.warning("Validation data loader is empty. Skipping evaluation.")
            return float('inf')
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in data_loader:
                sequences, sentiment_feats, targets = batch
                sequences = sequences.to(self.device)
                sentiment_feats = sentiment_feats.to(self.device)
                targets = targets.to(self.device)
                
                # Get predictions from all models
                lstm_pred = self.lstm_model(sequences)
                transformer_pred = self.transformer_model(sequences)
                sentiment_pred = self.sentiment_encoder(sentiment_feats)
                
                # Ensemble prediction
                ensemble_pred = (
                    self.config.lstm_weight * lstm_pred +
                    self.config.transformer_weight * transformer_pred +
                    self.config.sentiment_weight * sentiment_pred
                )
                
                loss = self.criterion(ensemble_pred, targets)
                total_loss += loss.item()
        
        return total_loss / len(data_loader)
    
    # Inside the EnsembleStockPredictor class

    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,  # This is X_val in the function context
        y_test: np.ndarray   # This is y_val in the function context
    ):
        """Train XGBoost models for each horizon"""
        
        # --- ADDED SAFETY CHECK for empty sequences ---
        # X_train is X_train_seq (must be 3D)
        # X_test is X_test_seq (must be 3D)
        
        if len(X_train) == 0:
            logger.warning("XGBoost training skipped: Training sequence data is empty.")
            return
        
        # Handle the empty test set case by using only the training data for fitting
        X_val_flat = None
        y_val_flat = None
        eval_set = []
        
        if len(X_test) == 0:
            logger.warning("XGBoost validation set is empty (Test samples: 0). Training without early stopping/evaluation.")
            X_train_flat = X_train[:, -1, :] # Safe to index X_train
        else:
            # Flatten sequences for XGBoost (use last timestep features)
            X_train_flat = X_train[:, -1, :]  # Shape: (samples, features)
            # The fix: This only runs if X_test is NOT empty
            X_val_flat = X_test[:, -1, :]
            y_val_flat = y_test
        # ---------------------------------------------
        
        logger.info(f"XGBoost training starts with {len(X_train_flat)} samples.")
        
        for i, horizon in enumerate(self.config.prediction_horizons):
            logger.info(f"Training XGBoost for {horizon}-day horizon...")
            
            # Prepare evaluation set if validation data exists
            if X_val_flat is not None:
                eval_set = [(X_val_flat, y_val_flat[:, i])]
            else:
                eval_set = [] # No evaluation set
                
            self.xgb_models[i].fit(
                X_train_flat,
                y_train[:, i],
                eval_set=eval_set,
                verbose=False
            )
    
    def train_meta_learner(
        self,
        X_sequences: np.ndarray,
        X_sentiment: np.ndarray,
        y_true: np.ndarray
    ):
        """Train meta-learner on predictions from base models"""
        
        # Get predictions from all base models
        base_predictions = self.get_base_predictions(X_sequences, X_sentiment)
        
        # Train meta-learner
        self.meta_learner.fit(base_predictions, y_true)
        logger.info("Meta-learner trained")
    
    def get_base_predictions(
        self,
        X_sequences: np.ndarray,
        X_sentiment: np.ndarray
    ) -> np.ndarray:
        """Get predictions from all base models"""
        
        self.lstm_model.eval()
        self.transformer_model.eval()
        self.sentiment_encoder.eval()
        
        with torch.no_grad():
            sequences_tensor = torch.FloatTensor(X_sequences).to(self.device)
            sentiment_tensor = torch.FloatTensor(X_sentiment).to(self.device)
            
            lstm_pred = self.lstm_model(sequences_tensor).cpu().numpy()
            transformer_pred = self.transformer_model(sequences_tensor).cpu().numpy()
            sentiment_pred = self.sentiment_encoder(sentiment_tensor).cpu().numpy()
        
        # XGBoost predictions
        X_flat = X_sequences[:, -1, :]
        xgb_preds = []
        for model in self.xgb_models:
            xgb_preds.append(model.predict(X_flat))
        xgb_pred = np.column_stack(xgb_preds)
        
        # Stack all predictions
        base_preds = np.hstack([lstm_pred, transformer_pred, sentiment_pred, xgb_pred])
        
        return base_preds
    
    def predict(
        self,
        X_sequences: np.ndarray,
        X_sentiment: np.ndarray,
        use_meta_learner: bool = True
    ) -> np.ndarray:
        """Make ensemble predictions"""
        
        if use_meta_learner:
            # Use meta-learner
            base_preds = self.get_base_predictions(X_sequences, X_sentiment)
            predictions = self.meta_learner.predict(base_preds)
        else:
            # Weighted average ensemble
            self.lstm_model.eval()
            self.transformer_model.eval()
            self.sentiment_encoder.eval()
            
            with torch.no_grad():
                sequences_tensor = torch.FloatTensor(X_sequences).to(self.device)
                sentiment_tensor = torch.FloatTensor(X_sentiment).to(self.device)
                
                lstm_pred = self.lstm_model(sequences_tensor).cpu().numpy()
                transformer_pred = self.transformer_model(sequences_tensor).cpu().numpy()
                sentiment_pred = self.sentiment_encoder(sentiment_tensor).cpu().numpy()
            
            # XGBoost predictions
            X_flat = X_sequences[:, -1, :]
            xgb_preds = []
            for model in self.xgb_models:
                xgb_preds.append(model.predict(X_flat))
            xgb_pred = np.column_stack(xgb_preds)
            
            # Weighted ensemble
            predictions = (
                self.config.lstm_weight * lstm_pred +
                self.config.transformer_weight * transformer_pred +
                self.config.xgb_weight * xgb_pred +
                self.config.sentiment_weight * sentiment_pred
            )
        
        return predictions
    
    def save_models(self, path: str):
        """Save all models and components"""
        import os
        os.makedirs(path, exist_ok=True)
        
        # Save PyTorch models
        torch.save(self.lstm_model.state_dict(), f"{path}/lstm_model.pth")
        torch.save(self.transformer_model.state_dict(), f"{path}/transformer_model.pth")
        torch.save(self.sentiment_encoder.state_dict(), f"{path}/sentiment_encoder.pth")
        
        # Save XGBoost models
        for i, model in enumerate(self.xgb_models):
            model.save_model(f"{path}/xgb_model_{i}.json")
        
        # Save meta-learner
        with open(f"{path}/meta_learner.pkl", 'wb') as f:
            pickle.dump(self.meta_learner, f)
        
        logger.info(f"Models saved to {path}")
    
    def load_models(self, path: str):
        """Load all models and components"""
        
        # Load PyTorch models
        self.lstm_model.load_state_dict(torch.load(f"{path}/lstm_model.pth"))
        self.transformer_model.load_state_dict(torch.load(f"{path}/transformer_model.pth"))
        self.sentiment_encoder.load_state_dict(torch.load(f"{path}/sentiment_encoder.pth"))
        
        # Load XGBoost models
        for i, model in enumerate(self.xgb_models):
            model.load_model(f"{path}/xgb_model_{i}.json")
        
        # Load meta-learner
        with open(f"{path}/meta_learner.pkl", 'rb') as f:
            self.meta_learner = pickle.load(f)
        
        logger.info(f"Models loaded from {path}")


# ============================================================================
# TRAINING PIPELINE
# ============================================================================

class TrainingPipeline:
    """Complete training pipeline for the stock prediction system"""
    
    def __init__(self, symbol: str, config: ModelConfig):
        self.symbol = symbol
        self.config = config
        self.feature_engineer = FeatureEngineer(config)
        self.sentiment_analyzer = SentimentAnalyzer()
        
    def prepare_data(self) -> Tuple[np.ndarray, ...]:
        """Complete data preparation pipeline (Optimized and Hardened)"""
        
        logger.info(f"Starting data preparation for {self.symbol}")
        
        # 1. Fetch market data
        market_fetcher = MarketDataFetcher(self.symbol, period="5y")
        df = market_fetcher.fetch_data()
        df = market_fetcher.compute_returns()
        
        # 2. Compute technical indicators
        df = TechnicalIndicators.compute_all_indicators(df, self.config)
        df = TechnicalIndicators.compute_rolling_statistics(df, self.config)
        
        # 3. Create targets
        df = self.feature_engineer.create_targets(df)
        
        # 4. Compute sentiment features (OPTIMIZED: Analyze once, use rolling windows)
        logger.info("Starting (Placeholder) Sentiment Feature Generation...")
        
        # A. Fetch and analyze news once (proxy for historical data)
        articles = self.sentiment_analyzer.fetch_news_headlines(self.symbol, days=365*5)
        texts = [article['title'] for article in articles]
        
        # The expensive NLP operation runs once.
        sentiments_list = self.sentiment_analyzer.analyze_sentiment(texts)
        
        # B. Aggregate the single batch of sentiment scores
        if sentiments_list:
            mean_pos = np.mean([s['positive'] for s in sentiments_list])
            mean_neg = np.mean([s['negative'] for s in sentiments_list])
            mean_comp = np.mean([s['compound'] for s in sentiments_list])
        else:
            # Default to neutral if no articles were found
            mean_pos, mean_neg, mean_comp = 0.33, 0.33, 0.0

        # C. Create a placeholder DataFrame where every day has the same base sentiment.
        base_sent_df = pd.DataFrame(index=df.index, data={
            'daily_sent_compound': mean_comp,
            'daily_sent_positive': mean_pos,
            'daily_sent_negative': mean_neg,
        })
        
        # D. Compute rolling statistics (The historical features)
        sentiment_features_list = []
        for window in [1, 5, 21]:
            base_sent_df[f'sentiment_mean_{window}d'] = base_sent_df['daily_sent_compound'].rolling(window=window).mean()
            base_sent_df[f'sentiment_std_{window}d'] = base_sent_df['daily_sent_compound'].rolling(window=window).std().fillna(0)
            base_sent_df[f'sentiment_positive_{window}d'] = base_sent_df['daily_sent_positive'].rolling(window=window).mean()
            base_sent_df[f'sentiment_negative_{window}d'] = base_sent_df['daily_sent_negative'].rolling(window=window).mean()
            
            sentiment_features_list.extend([
                f'sentiment_mean_{window}d', f'sentiment_std_{window}d',
                f'sentiment_positive_{window}d', f'sentiment_negative_{window}d'
            ])
            
        sentiment_df = base_sent_df[sentiment_features_list]
        logger.info(f"Sentiment features generated for {len(sentiment_df)} dates.")

        # 5. Prepare features
        # Get the list of technical/price features
        df, feature_cols = self.feature_engineer.prepare_features(df, None)
        
        # --- HARDENED ALIGNMENT AND NaN REMOVAL ---
        
        target_cols = [f'target_{h}d' for h in self.config.prediction_horizons]
        
        # 5a. Merge all features and targets into one DataFrame for reliable NaN removal
        df_features = df[feature_cols + target_cols] # Technical/Price features + Targets
        df_full = pd.concat([df_features, sentiment_df], axis=1) # Add Sentiment features
        
        # 5b. Determine all columns required for the final dataset
        all_required_cols = feature_cols + target_cols + list(sentiment_df.columns)
        
        # 5c. Remove NaN rows based on all required columns
        df_clean = df_full.dropna(subset=all_required_cols)
        
        # 7. Extract features and targets from the CLEANED DataFrame
        # Extract ALL features
        X_all_features = df_clean[feature_cols + list(sentiment_df.columns)].values
        y = df_clean[target_cols].values

        # Separate X into X_tech and X_sentiment (for the separate model inputs)
        num_sentiment_features = len(sentiment_df.columns)
        
        X = X_all_features[:, :-num_sentiment_features]        # Technical/Price Features
        X_sentiment = X_all_features[:, -num_sentiment_features:] # Sentiment Features
        
        # 8. Split data (time-series split)
        split_idx = int(len(X) * self.config.train_test_split)
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        X_sent_train, X_sent_test = X_sentiment[:split_idx], X_sentiment[split_idx:]
        
        # 9. Scale features (with safety check for empty test set)
        X_train_scaled = self.feature_engineer.feature_scaler.fit_transform(X_train)
        if X_test.size > 0:
            X_test_scaled = self.feature_engineer.feature_scaler.transform(X_test)
        else:
            # Return correctly shaped empty array if the test set is empty
            X_test_scaled = np.empty((0, X_train_scaled.shape[1]))
        
        # 10. Scale targets (with safety check for empty test set)
        y_train_scaled = self.feature_engineer.target_scaler.fit_transform(y_train)
        if y_test.size > 0:
            y_test_scaled = self.feature_engineer.target_scaler.transform(y_test)
        else:
            y_test_scaled = np.empty((0, y_train_scaled.shape[1]))
        
        # 11. Create sequences
        X_train_seq, y_train_seq = self.feature_engineer.create_sequences(
            X_train_scaled, y_train_scaled, self.config.lookback_period
        )
        X_test_seq, y_test_seq = self.feature_engineer.create_sequences(
            X_test_scaled, y_test_scaled, self.config.lookback_period
        )
        
        # Adjust sentiment features to match sequence length (with safety check)
        if len(X_sent_train) >= self.config.lookback_period:
            X_sent_train_seq = X_sent_train[self.config.lookback_period:]
        else:
            X_sent_train_seq = np.empty((0, X_sent_train.shape[1] if X_sent_train.ndim > 1 else 0))

        if len(X_sent_test) >= self.config.lookback_period:
            X_sent_test_seq = X_sent_test[self.config.lookback_period:]
        else:
            X_sent_test_seq = np.empty((0, X_sent_test.shape[1] if X_sent_test.ndim > 1 else 0))

        logger.info(f"Data preparation complete. Train samples: {len(X_train_seq)}, Test samples: {len(X_test_seq)}")
        
        return (
            X_train_seq, y_train_seq, X_sent_train_seq,
            X_test_seq, y_test_seq, X_sent_test_seq
        )
        
    def train(self) -> EnsembleStockPredictor:
        """Execute complete training pipeline"""
        
        # Prepare data
        (X_train, y_train, X_sent_train,
         X_test, y_test, X_sent_test) = self.prepare_data()
        
        # Create data loaders
        train_dataset = StockDataset(X_train, y_train, X_sent_train)
        test_dataset = StockDataset(X_test, y_test, X_sent_test)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=False  # Don't shuffle time-series data
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        # Initialize ensemble
        num_features = X_train.shape[2]
        num_sentiment_features = X_sent_train.shape[1]
        
        ensemble = EnsembleStockPredictor(
            self.config,
            num_features,
            num_sentiment_features
        )
        
        # Train deep learning models
        logger.info("Training deep learning models...")
        history = ensemble.train_deep_models(
            train_loader,
            test_loader,
            self.config.num_epochs
        )
        
        # Train XGBoost
        logger.info("Training XGBoost models...")
        ensemble.train_xgboost(X_train, y_train, X_test, y_test)
        
        # Train meta-learner
        logger.info("Training meta-learner...")
        ensemble.train_meta_learner(X_test, X_sent_test, y_test)
        
        logger.info("Training complete!")
        
        return ensemble


# ============================================================================
# EVALUATION MODULE
# ============================================================================

class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    @staticmethod
    def compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        horizon_names: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Compute all evaluation metrics"""
        
        metrics = {}
        
        for i, horizon in enumerate(horizon_names):
            y_t = y_true[:, i]
            y_p = y_pred[:, i]
            
            # Regression metrics
            mae = mean_absolute_error(y_t, y_p)
            rmse = np.sqrt(mean_squared_error(y_t, y_p))
            mape = np.mean(np.abs((y_t - y_p) / y_t)) * 100
            
            # Directional accuracy
            direction_true = np.sign(y_t - np.roll(y_t, 1)[1:])
            direction_pred = np.sign(y_p - np.roll(y_p, 1)[1:])
            directional_accuracy = np.mean(direction_true == direction_pred)
            
            # R-squared
            ss_res = np.sum((y_t - y_p) ** 2)
            ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            
            metrics[horizon] = {
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,
                'R2': r2,
                'Directional_Accuracy': directional_accuracy
            }
        
        return metrics
    
    @staticmethod
    def evaluate_economic_performance(
        prices_true: np.ndarray,
        prices_pred: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate economic performance metrics"""
        
        # Simple trading strategy: buy if prediction > current, sell otherwise
        returns_true = np.diff(prices_true) / prices_true[:-1]
        returns_pred = np.diff(prices_pred) / prices_pred[:-1]
        
        # Trading signals
        signals = np.sign(returns_pred)
        strategy_returns = signals * returns_true
        
        # Cumulative return
        cumulative_return = np.prod(1 + strategy_returns) - 1
        
        # Sharpe ratio (annualized)
        sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
        
        # Max drawdown
        cumulative = np.cumprod(1 + strategy_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        return {
            'Cumulative_Return': cumulative_return,
            'Sharpe_Ratio': sharpe_ratio,
            'Max_Drawdown': max_drawdown
        }


# ============================================================================
# INFERENCE MODULE
# ============================================================================

class InferenceEngine:
    """Production inference engine"""
    
    def __init__(
        self,
        ensemble: EnsembleStockPredictor,
        feature_engineer: FeatureEngineer,
        sentiment_analyzer: SentimentAnalyzer,
        config: ModelConfig
    ):
        self.ensemble = ensemble
        self.feature_engineer = feature_engineer
        self.sentiment_analyzer = sentiment_analyzer
        self.config = config
    
    def get_stock_input(self, symbol: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fetch and process recent data for a stock
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Tuple of (feature_sequence, sentiment_features)
        """
        # Fetch recent market data
        market_fetcher = MarketDataFetcher(symbol, period="1y")
        df = market_fetcher.fetch_data()
        df = market_fetcher.compute_returns()
        
        # Compute indicators
        df = TechnicalIndicators.compute_all_indicators(df, self.config)
        df = TechnicalIndicators.compute_rolling_statistics(df, self.config)
        
        # Get sentiment features for latest date
        latest_date = df.index[-1]
        sentiment_features = self.sentiment_analyzer.compute_sentiment_features(
            symbol,
            latest_date,
            windows=[1, 5, 21]
        )
        
        # Prepare features
        sentiment_df = pd.DataFrame([sentiment_features])
        df, feature_cols = self.feature_engineer.prepare_features(df, None)
        
        # Get last sequence
        X = df[feature_cols].iloc[-self.config.lookback_period:].values
        
        # Scale features
        X_scaled = self.feature_engineer.feature_scaler.transform(X)
        
        # Reshape to sequence
        X_seq = X_scaled.reshape(1, self.config.lookback_period, -1)
        
        # Sentiment features
        X_sent = np.array(list(sentiment_features.values())).reshape(1, -1)
        
        return X_seq, X_sent
    
    def predict_price(
        self,
        symbol: str,
        horizon: str = '1d'
    ) -> Dict[str, float]:
        """
        Predict stock price for specified horizon
        
        Args:
            symbol: Stock ticker symbol
            horizon: Prediction horizon ('1d', '5d', '21d', '252d')
            
        Returns:
            Dictionary with prediction details
        """
        # Get input data
        X_seq, X_sent = self.get_stock_input(symbol)
        
        # Make prediction
        prediction_scaled = self.ensemble.predict(X_seq, X_sent, use_meta_learner=True)
        
        # Inverse transform
        prediction = self.feature_engineer.target_scaler.inverse_transform(prediction_scaled)
        
        # Get current price
        current_price = yf.Ticker(symbol).history(period='1d')['Close'].iloc[-1]
        
        # Map horizon to index
        horizon_map = {
            '1d': 0, '5d': 1, '21d': 2, '252d': 3
        }
        
        if horizon not in horizon_map:
            raise ValueError(f"Invalid horizon: {horizon}. Must be one of {list(horizon_map.keys())}")
        
        idx = horizon_map[horizon]
        predicted_price = prediction[0, idx]
        
        # Calculate change
        price_change = predicted_price - current_price
        price_change_pct = (price_change / current_price) * 100
        
        return {
            'symbol': symbol,
            'current_price': float(current_price),
            'predicted_price': float(predicted_price),
            'price_change': float(price_change),
            'price_change_pct': float(price_change_pct),
            'horizon': horizon,
            'timestamp': datetime.now().isoformat()
        }
    
    def save_artifacts(self, path: str = 'model_artifacts'):
        """Save all model artifacts for production deployment"""
        import os
        os.makedirs(path, exist_ok=True)
        
        # Save ensemble models
        self.ensemble.save_models(f"{path}/ensemble")
        
        # Save scalers
        with open(f"{path}/feature_scaler.pkl", 'wb') as f:
            pickle.dump(self.feature_engineer.feature_scaler, f)
        
        with open(f"{path}/target_scaler.pkl", 'wb') as f:
            pickle.dump(self.feature_engineer.target_scaler, f)
        
        # Save feature names
        with open(f"{path}/feature_names.json", 'w') as f:
            json.dump(self.feature_engineer.feature_names, f)
        
        # Save config
        with open(f"{path}/config.json", 'w') as f:
            json.dump({
                'lookback_period': self.config.lookback_period,
                'prediction_horizons': self.config.prediction_horizons,
                'feature_count': len(self.feature_engineer.feature_names)
            }, f, indent=2)
        
        logger.info(f"All artifacts saved to {path}")
    
    @classmethod
    def load_from_artifacts(cls, path: str = 'model_artifacts') -> 'InferenceEngine':
        """Load inference engine from saved artifacts"""
        
        # Load config
        with open(f"{path}/config.json", 'r') as f:
            config_dict = json.load(f)
        
        config = ModelConfig()
        
        # Load scalers
        with open(f"{path}/feature_scaler.pkl", 'rb') as f:
            feature_scaler = pickle.load(f)
        
        with open(f"{path}/target_scaler.pkl", 'rb') as f:
            target_scaler = pickle.load(f)
        
        # Load feature names
        with open(f"{path}/feature_names.json", 'r') as f:
            feature_names = json.load(f)
        
        # Create feature engineer
        feature_engineer = FeatureEngineer(config)
        feature_engineer.feature_scaler = feature_scaler
        feature_engineer.target_scaler = target_scaler
        feature_engineer.feature_names = feature_names
        
        # Load ensemble
        num_features = config_dict['feature_count']
        num_sentiment_features = 12  # 4 features * 3 windows
        
        ensemble = EnsembleStockPredictor(config, num_features, num_sentiment_features)
        ensemble.load_models(f"{path}/ensemble")
        
        # Create sentiment analyzer
        sentiment_analyzer = SentimentAnalyzer()
        
        # Create inference engine
        engine = cls(ensemble, feature_engineer, sentiment_analyzer, config)
        
        logger.info(f"Inference engine loaded from {path}")
        
        return engine


# ============================================================================
# UNIT TESTS
# ============================================================================

def test_data_fetching():
    """Test data fetching functionality"""
    logger.info("Testing data fetching...")
    
    fetcher = MarketDataFetcher("AAPL", period="1y")
    df = fetcher.fetch_data()
    
    assert df is not None, "Data fetch failed"
    assert len(df) > 0, "Empty dataframe"
    assert 'Close' in df.columns, "Missing Close column"
    
    logger.info("✓ Data fetching test passed")


def test_technical_indicators():
    """Test technical indicator calculation"""
    logger.info("Testing technical indicators...")
    
    fetcher = MarketDataFetcher("AAPL", period="1y")
    df = fetcher.fetch_data()
    df = TechnicalIndicators.compute_all_indicators(df, config)
    
    assert 'SMA_20' in df.columns, "Missing SMA indicator"
    assert 'RSI' in df.columns, "Missing RSI indicator"
    assert 'MACD' in df.columns, "Missing MACD indicator"
    
    logger.info("✓ Technical indicators test passed")


def test_sentiment_analysis():
    """Test sentiment analysis"""
    logger.info("Testing sentiment analysis...")
    
    analyzer = SentimentAnalyzer()
    texts = ["Stock prices are rising sharply", "Market crash imminent"]
    sentiments = analyzer.analyze_sentiment(texts)
    
    assert len(sentiments) == 2, "Wrong number of sentiments"
    assert 'compound' in sentiments[0], "Missing compound score"
    
    logger.info("✓ Sentiment analysis test passed")


def test_model_prediction():
    """Test model prediction pipeline"""
    logger.info("Testing model prediction...")

    # --- Setup Dummy Data ---
    # The feature count (50) and sentiment feature count (12) must match the ensemble initialization.
    num_samples = 10
    num_features = 50
    num_sentiment_features = 12
    num_horizons = len(config.prediction_horizons)

    # Create dummy data for sequences (X_seq) and sentiment (X_sent)
    X_seq = np.random.randn(num_samples, config.lookback_period, num_features)
    X_sent = np.random.randn(num_samples, num_sentiment_features)

    # Create dummy targets for training (y)
    y_dummy = np.random.randn(num_samples, num_horizons)

    # --- Initialize Ensemble Model ---
    ensemble = EnsembleStockPredictor(config, num_features, num_sentiment_features)
    
    # NOTE: PyTorch models (LSTM, Transformer, SentimentEncoder) will work without training
    #       because their parameters are randomly initialized, and .eval() is the default for a fresh model.
    #       However, XGBoost models MUST be trained/fitted.

    # --- Fit XGBoost Models on Dummy Data ---
    # XGBoost expects a flat feature matrix, using the last time step's features.
    X_flat = X_seq[:, -1, :] # Shape: (samples, features)

    for i, model in enumerate(ensemble.xgb_models):
        # Fit each XGBoost model to its corresponding horizon's dummy target
        model.fit(X_flat, y_dummy[:, i])

    # --- Test Prediction ---
    # We test with use_meta_learner=False because we haven't trained the meta-learner.
    predictions = ensemble.predict(X_seq, X_sent, use_meta_learner=False)

    assert predictions.shape == (num_samples, num_horizons), f"Wrong prediction shape: {predictions.shape} instead of ({num_samples}, {num_horizons})"

    logger.info("✓ Model prediction test passed")


def run_all_tests():
    """Run all unit tests"""
    logger.info("="*60)
    logger.info("RUNNING UNIT TESTS")
    logger.info("="*60)
    
    try:
        test_data_fetching()
        test_technical_indicators()
        test_sentiment_analysis()
        test_model_prediction()
        
        logger.info("="*60)
        logger.info("ALL TESTS PASSED ✓")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"TEST FAILED: {str(e)}")
        raise


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # Run unit tests
    run_all_tests()
    
    # Example: Train model for a stock
    logger.info("\n" + "="*60)
    logger.info("TRAINING STOCK PREDICTION MODEL")
    logger.info("="*60)
    
    SYMBOL = "AAPL"  # Change this to any stock symbol
    
    # Initialize training pipeline
    pipeline = TrainingPipeline(SYMBOL, config)
    
    # Train the ensemble
    ensemble = pipeline.train()
    
    # Create inference engine
    inference_engine = InferenceEngine(
        ensemble,
        pipeline.feature_engineer,
        pipeline.sentiment_analyzer,
        config
    )
    
    # Save all artifacts
    inference_engine.save_artifacts('model_artifacts')
    
    # Make predictions for all horizons
    logger.info("\n" + "="*60)
    logger.info("MAKING PREDICTIONS")
    logger.info("="*60)
    
    for horizon in ['1d', '5d', '21d', '252d']:
        prediction = inference_engine.predict_price(SYMBOL, horizon)
        
        logger.info(f"\n{horizon} Prediction:")
        logger.info(f"  Current Price: ${prediction['current_price']:.2f}")
        logger.info(f"  Predicted Price: ${prediction['predicted_price']:.2f}")
        logger.info(f"  Change: ${prediction['price_change']:.2f} ({prediction['price_change_pct']:.2f}%)")
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING AND INFERENCE COMPLETE!")
    logger.info("="*60)
    
    # Example: Load from saved artifacts and make prediction
    logger.info("\n" + "="*60)
    logger.info("TESTING ARTIFACT LOADING")
    logger.info("="*60)
    
    # Load inference engine from artifacts
    loaded_engine = InferenceEngine.load_from_artifacts('model_artifacts')
    
    # Make a prediction
    test_prediction = loaded_engine.predict_price(SYMBOL, '5d')
    logger.info(f"\nTest prediction loaded from artifacts:")
    logger.info(f"  Symbol: {test_prediction['symbol']}")
    logger.info(f"  Predicted 5-day price: ${test_prediction['predicted_price']:.2f}")
    
    logger.info("\n" + "="*60)
    logger.info("ALL OPERATIONS COMPLETED SUCCESSFULLY!")
    logger.info("="*60)


# ============================================================================
# ADVANCED USAGE EXAMPLES
# ============================================================================

"""
ADVANCED USAGE EXAMPLES
========================

1. Training a model for a specific stock:
-----------------------------------------
from datetime import datetime

symbol = "TSLA"
config = ModelConfig(
    lookback_period=252,
    num_epochs=100,
    batch_size=32,
    lstm_hidden_size=128,
    transformer_d_model=128
)

pipeline = TrainingPipeline(symbol, config)
ensemble = pipeline.train()

inference_engine = InferenceEngine(
    ensemble,
    pipeline.feature_engineer,
    pipeline.sentiment_analyzer,
    config
)

# Save for production
inference_engine.save_artifacts(f'models/{symbol}')


2. Making predictions in production:
-------------------------------------
# Load trained model
engine = InferenceEngine.load_from_artifacts('models/TSLA')

# Get predictions for all horizons
predictions = {}
for horizon in ['1d', '5d', '21d', '252d']:
    predictions[horizon] = engine.predict_price('TSLA', horizon)

# Use predictions
print(f"1-day prediction: ${predictions['1d']['predicted_price']:.2f}")
print(f"1-week prediction: ${predictions['5d']['predicted_price']:.2f}")
print(f"1-month prediction: ${predictions['21d']['predicted_price']:.2f}")
print(f"1-year prediction: ${predictions['252d']['predicted_price']:.2f}")


3. Batch prediction for multiple stocks:
-----------------------------------------
symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
engine = InferenceEngine.load_from_artifacts('model_artifacts')

all_predictions = {}
for symbol in symbols:
    try:
        all_predictions[symbol] = engine.predict_price(symbol, '5d')
    except Exception as e:
        print(f"Error predicting {symbol}: {e}")

# Sort by predicted return
sorted_stocks = sorted(
    all_predictions.items(),
    key=lambda x: x[1]['price_change_pct'],
    reverse=True
)

print("Top 3 stocks by predicted 5-day return:")
for symbol, pred in sorted_stocks[:3]:
    print(f"{symbol}: {pred['price_change_pct']:.2f}%")


4. Custom model configuration:
-------------------------------
custom_config = ModelConfig(
    lookback_period=126,  # 6 months
    prediction_horizons=[1, 5, 10, 21, 63, 126, 252],  # Custom horizons
    
    # Model architecture
    lstm_hidden_size=256,
    lstm_num_layers=4,
    transformer_d_model=256,
    transformer_nhead=16,
    transformer_num_layers=6,
    
    # Training
    num_epochs=200,
    batch_size=64,
    learning_rate=0.0005,
    early_stopping_patience=20,
    
    # Ensemble weights
    lstm_weight=0.25,
    transformer_weight=0.35,
    xgb_weight=0.25,
    sentiment_weight=0.15
)


5. Walk-forward validation:
----------------------------
def walk_forward_validation(symbol, config, n_splits=5):
    '''
    Perform walk-forward validation for time-series
    '''
    fetcher = MarketDataFetcher(symbol, period="5y")
    df = fetcher.fetch_data()
    
    total_samples = len(df)
    test_size = total_samples // n_splits
    
    results = []
    
    for i in range(n_splits):
        train_end = total_samples - (n_splits - i) * test_size
        test_end = train_end + test_size
        
        # Split data
        train_df = df.iloc[:train_end]
        test_df = df.iloc[train_end:test_end]
        
        # Train model on this fold
        pipeline = TrainingPipeline(symbol, config)
        ensemble = pipeline.train()
        
        # Evaluate on test fold
        # ... evaluation code ...
        
        results.append({
            'fold': i,
            'train_size': len(train_df),
            'test_size': len(test_df),
            # ... metrics ...
        })
    
    return results


6. Feature importance analysis:
--------------------------------
def analyze_feature_importance(ensemble, feature_names):
    '''
    Analyze feature importance from XGBoost models
    '''
    import pandas as pd
    
    importance_data = []
    
    for i, model in enumerate(ensemble.xgb_models):
        importance = model.feature_importances_
        horizon = config.prediction_horizons[i]
        
        for feat, imp in zip(feature_names, importance):
            importance_data.append({
                'feature': feat,
                'importance': imp,
                'horizon': f'{horizon}d'
            })
    
    df = pd.DataFrame(importance_data)
    
    # Top features per horizon
    for horizon in df['horizon'].unique():
        print(f"\nTop 10 features for {horizon}:")
        top_features = df[df['horizon'] == horizon].nlargest(10, 'importance')
        print(top_features[['feature', 'importance']])


7. Real-time monitoring and alerts:
------------------------------------
def monitor_predictions(symbols, engine, threshold=5.0):
    '''
    Monitor predictions and generate alerts for significant moves
    '''
    alerts = []
    
    for symbol in symbols:
        try:
            pred = engine.predict_price(symbol, '1d')
            
            if abs(pred['price_change_pct']) > threshold:
                alerts.append({
                    'symbol': symbol,
                    'current': pred['current_price'],
                    'predicted': pred['predicted_price'],
                    'change_pct': pred['price_change_pct'],
                    'timestamp': pred['timestamp']
                })
        except Exception as e:
            print(f"Error monitoring {symbol}: {e}")
    
    return alerts


8. Model performance tracking:
-------------------------------
class PerformanceTracker:
    '''
    Track model predictions vs actual outcomes
    '''
    def __init__(self):
        self.predictions = []
        self.actuals = []
    
    def log_prediction(self, symbol, horizon, predicted_price, timestamp):
        self.predictions.append({
            'symbol': symbol,
            'horizon': horizon,
            'predicted_price': predicted_price,
            'timestamp': timestamp
        })
    
    def log_actual(self, symbol, horizon, actual_price, timestamp):
        self.actuals.append({
            'symbol': symbol,
            'horizon': horizon,
            'actual_price': actual_price,
            'timestamp': timestamp
        })
    
    def compute_accuracy(self):
        # Match predictions with actuals
        # Compute MAE, RMSE, directional accuracy
        pass


9. Hyperparameter optimization:
--------------------------------
def optimize_hyperparameters(symbol, param_grid):
    '''
    Grid search for optimal hyperparameters
    '''
    from itertools import product
    
    best_score = float('inf')
    best_params = None
    
    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    
    for combination in product(*values):
        params = dict(zip(keys, combination))
        
        # Create config with these params
        config = ModelConfig(**params)
        
        # Train and evaluate
        pipeline = TrainingPipeline(symbol, config)
        ensemble = pipeline.train()
        
        # Compute validation score
        score = evaluate_model(ensemble)
        
        if score < best_score:
            best_score = score
            best_params = params
    
    return best_params, best_score


10. Ensemble of ensembles:
---------------------------
class MetaEnsemble:
    '''
    Train multiple ensemble models and combine their predictions
    '''
    def __init__(self, symbols, config):
        self.models = {}
        
        for symbol in symbols:
            pipeline = TrainingPipeline(symbol, config)
            self.models[symbol] = pipeline.train()
    
    def predict_portfolio(self, symbols, horizon='5d'):
        predictions = {}
        
        for symbol in symbols:
            if symbol in self.models:
                # Use symbol-specific model
                model = self.models[symbol]
            else:
                # Use nearest similar model (by sector, market cap, etc.)
                model = self.find_similar_model(symbol)
            
            # Make prediction
            engine = InferenceEngine(model, ...)
            predictions[symbol] = engine.predict_price(symbol, horizon)
        
        return predictions


INTEGRATION NOTES FOR FLASK BACKEND:
=====================================

1. Model Loading (app initialization):
---------------------------------------
from flask import Flask, jsonify, request

app = Flask(__name__)

# Load model once at startup
inference_engine = InferenceEngine.load_from_artifacts('model_artifacts')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    symbol = data.get('symbol')
    horizon = data.get('horizon', '5d')
    
    try:
        prediction = inference_engine.predict_price(symbol, horizon)
        return jsonify(prediction)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


2. Async prediction with Celery:
---------------------------------
from celery import Celery

celery = Celery('tasks', broker='redis://localhost:6379')

@celery.task
def predict_async(symbol, horizon):
    engine = InferenceEngine.load_from_artifacts('model_artifacts')
    return engine.predict_price(symbol, horizon)


3. Caching predictions:
------------------------
from flask_caching import Cache

cache = Cache(app, config={'CACHE_TYPE': 'redis'})

@app.route('/predict/<symbol>/<horizon>')
@cache.cached(timeout=300)  # Cache for 5 minutes
def cached_predict(symbol, horizon):
    prediction = inference_engine.predict_price(symbol, horizon)
    return jsonify(prediction)


4. Batch prediction endpoint:
------------------------------
@app.route('/predict/batch', methods=['POST'])
def batch_predict():
    data = request.json
    symbols = data.get('symbols', [])
    horizon = data.get('horizon', '5d')
    
    results = {}
    for symbol in symbols:
        try:
            results[symbol] = inference_engine.predict_price(symbol, horizon)
        except Exception as e:
            results[symbol] = {'error': str(e)}
    
    return jsonify(results)


5. Model versioning:
--------------------
MODELS = {
    'v1': InferenceEngine.load_from_artifacts('models/v1'),
    'v2': InferenceEngine.load_from_artifacts('models/v2')
}

@app.route('/predict/<version>/<symbol>/<horizon>')
def versioned_predict(version, symbol, horizon):
    if version not in MODELS:
        return jsonify({'error': 'Invalid model version'}), 400
    
    engine = MODELS[version]
    prediction = engine.predict_price(symbol, horizon)
    return jsonify(prediction)


DEPLOYMENT CHECKLIST:
======================
☐ Train models on comprehensive historical data (5-10 years)
☐ Save all artifacts (models, scalers, feature names)
☐ Implement model versioning
☐ Set up monitoring and logging
☐ Configure caching (Redis)
☐ Add rate limiting
☐ Implement error handling and fallbacks
☐ Set up CI/CD pipeline
☐ Configure GPU servers (if using)
☐ Add authentication/API keys
☐ Set up model retraining schedule (weekly/monthly)
☐ Implement A/B testing for model versions
☐ Add performance monitoring dashboard
☐ Configure alerts for model degradation
☐ Document API endpoints
☐ Load testing
☐ Security audit
☐ Backup strategy for models and data


PERFORMANCE OPTIMIZATION TIPS:
================================
1. Use mixed precision training (FP16) to reduce memory usage
2. Implement gradient accumulation for larger effective batch sizes
3. Use DataLoader with num_workers > 0 for parallel data loading
4. Cache technical indicators to avoid recomputation
5. Use model quantization for faster inference
6. Implement prediction batching in production
7. Use Redis for caching frequently requested predictions
8. Profile code to identify bottlenecks
9. Use async processing for multiple stock predictions
10. Consider model distillation for faster inference


MONITORING METRICS:
====================
1. Prediction accuracy (MAE, RMSE, MAPE)
2. Directional accuracy
3. Inference latency (p50, p95, p99)
4. Model confidence/uncertainty
5. Feature drift detection
6. Prediction vs actual tracking
7. API response times
8. Cache hit rates
9. GPU utilization
10. Error rates by stock symbol


DISCLAIMER:
===========
This model is for educational and research purposes only.
Stock market predictions are inherently uncertain and should not
be used as the sole basis for investment decisions. Always consult
with qualified financial advisors before making investment choices.
Past performance does not guarantee future results.

The model combines state-of-the-art deep learning and machine learning
techniques but cannot account for unforeseen market events, regulatory
changes, or black swan events. Use at your own risk.
"""