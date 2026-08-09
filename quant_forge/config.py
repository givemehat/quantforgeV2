"""
Quant Forge Configuration
========================
Central config for NSE analytics platform.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Market Config ──────────────────────────────────────────────────────────────
RISK_FREE_RATE = 0.07  # INR rf: RBI repo rate ~7%
MARKET_TICKER = "^NSEI"  # Nifty50 benchmark
TRADING_DAYS = 252
MARKET_OPEN = "09:15"  # IST
MARKET_CLOSE = "15:30"  # IST

# ── NSE Universe (Top Nifty50 subset) ────────────────────────────────────────
NSE_TICKERS = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    "ICICIBANK.NS",
    "HINDUNILVR.NS",
    "SBIN.NS",
    "BHARTIARTL.NS",
    "ITC.NS",
    "KOTAKBANK.NS",
    "LT.NS",
    "HCLTECH.NS",
    "AXISBANK.NS",
    "BAJFINANCE.NS",
    "WIPRO.NS",
    "MARUTI.NS",
    "TITAN.NS",
    "ASIANPAINT.NS",
    "NTPC.NS",
    "POWERGRID.NS",
]

# US tickers for comparison
US_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

# Crypto
CRYPTO_TICKERS = ["BTC-USD", "ETH-USD"]

# ── API Config ─────────────────────────────────────────────────────────────────
NSE_API_BASE = "https://nse-api-khaki.vercel.app"
CRYPTO_API_BASE = "https://api.freecryptoapi.com/v1"
IBMQ_TOKEN = os.getenv("IBMQ_TOKEN", "")  # optional

# ── Backend ────────────────────────────────────────────────────────────────────
DB_PATH = "quant_forge.db"
LOG_LEVEL = "INFO"
POLL_INTERVAL = 5  # seconds
CACHE_TTL = 300  # 5 min SQLite cache

# ── Simulation ─────────────────────────────────────────────────────────────────
MC_PATHS = 10_000
MC_HORIZON = 252  # 1 trading year
VAR_CONFIDENCE = [0.95, 0.99]
GBM_DT = 1 / TRADING_DAYS

# ── Quantum ────────────────────────────────────────────────────────────────────
QAOA_REPS = 2
VQE_MAX_ITER = 200
N_ASSETS_QUANTUM = 10  # feasible on simulator
