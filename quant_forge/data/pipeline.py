"""
Quant Forge — Data Pipeline
============================
Live NSE data via NSE-API + yfinance fallback.
Async streaming, SQLite caching, market-hours guard.

Architecture:
  DataFetcher  →  AsyncDataStream  →  FeatureEngineer  →  DataStore
"""

import asyncio
import sqlite3
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import aiohttp

from config import (
    NSE_API_BASE,
    NSE_TICKERS,
    US_TICKERS,
    CRYPTO_TICKERS,
    DB_PATH,
    CACHE_TTL,
    MARKET_OPEN,
    MARKET_CLOSE,
    POLL_INTERVAL,
    TRADING_DAYS,
)

log = logging.getLogger("quant_forge.data")
IST = ZoneInfo("Asia/Kolkata")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Market Hours Guard
# ─────────────────────────────────────────────────────────────────────────────


def is_market_open() -> bool:
    """Returns True during NSE trading hours (Mon-Fri, 09:15-15:30 IST)."""
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    open_t = now.replace(hour=9, minute=15, second=0, microsecond=0)
    close_t = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return open_t <= now <= close_t


# ─────────────────────────────────────────────────────────────────────────────
# 2. SQLite Cache / Data Store
# ─────────────────────────────────────────────────────────────────────────────


class DataStore:
    """Lightweight SQLite cache for prices + metadata."""

    def __init__(self, db_path: str = DB_PATH):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self):
        cur = self.conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS price_cache (
                ticker   TEXT,
                ts       REAL,
                price    REAL,
                volume   REAL,
                cached_at REAL,
                PRIMARY KEY (ticker, ts)
            );
            CREATE TABLE IF NOT EXISTS ohlcv_history (
                ticker TEXT,
                date   TEXT,
                open   REAL,
                high   REAL,
                low    REAL,
                close  REAL,
                volume REAL,
                PRIMARY KEY (ticker, date)
            );
        """)
        self.conn.commit()

    def cache_price(self, ticker: str, price: float, volume: float = 0.0):
        ts = time.time()
        self.conn.execute(
            "INSERT OR REPLACE INTO price_cache VALUES (?,?,?,?,?)",
            (ticker, round(ts, 2), price, volume, ts),
        )
        self.conn.commit()

    def get_cached_price(self, ticker: str) -> Optional[float]:
        cutoff = time.time() - CACHE_TTL
        row = self.conn.execute(
            "SELECT price FROM price_cache WHERE ticker=? AND cached_at>=? ORDER BY ts DESC LIMIT 1",
            (ticker, cutoff),
        ).fetchone()
        return row[0] if row else None

    def upsert_ohlcv(self, ticker: str, df: pd.DataFrame):
        df = df.copy()
        df.index = df.index.astype(str)
        for date, row in df.iterrows():
            self.conn.execute(
                "INSERT OR REPLACE INTO ohlcv_history VALUES (?,?,?,?,?,?,?)",
                (
                    ticker,
                    date,
                    row.get("Open", np.nan),
                    row.get("High", np.nan),
                    row.get("Low", np.nan),
                    row.get("Close", np.nan),
                    row.get("Volume", 0),
                ),
            )
        self.conn.commit()

    def load_ohlcv(self, ticker: str, days: int = 365) -> Optional[pd.DataFrame]:
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        rows = self.conn.execute(
            "SELECT date,open,high,low,close,volume FROM ohlcv_history "
            "WHERE ticker=? AND date>=? ORDER BY date",
            (ticker, cutoff),
        ).fetchall()
        if not rows:
            return None
        df = pd.DataFrame(
            rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"]
        )
        df["Date"] = pd.to_datetime(df["Date"])
        return df.set_index("Date")


store = DataStore()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Live Price Fetcher
# ─────────────────────────────────────────────────────────────────────────────


class DataFetcher:
    """
    Fetches live prices:
      1. NSE-API-Khaki (REST, free)
      2. yfinance (fallback / historical OHLCV)
    """

    @staticmethod
    def fetch_nse_quote(ticker: str) -> Optional[Dict]:
        """Hit NSE API for live quote. ticker e.g. 'RELIANCE'."""
        symbol = ticker.replace(".NS", "")
        try:
            url = f"{NSE_API_BASE}/getQuote?symbol={symbol}"
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "ticker": ticker,
                    "price": float(
                        data.get(
                            "lastPrice", data.get("data", [{}])[0].get("lastPrice", 0)
                        )
                    ),
                    "change": float(data.get("change", 0)),
                    "pct": float(data.get("pChange", 0)),
                    "volume": float(data.get("totalTradedVolume", 0)),
                    "high": float(data.get("dayHigh", 0)),
                    "low": float(data.get("dayLow", 0)),
                }
        except Exception as e:
            log.warning(f"NSE API fail for {ticker}: {e}")
        return None

    @staticmethod
    def fetch_yf_quote(ticker: str) -> Optional[Dict]:
        """yfinance live quote fallback."""
        try:
            t = yf.Ticker(ticker)
            info = t.fast_info
            return {
                "ticker": ticker,
                "price": float(info.last_price or 0),
                "change": float(
                    info.last_price - info.previous_close if info.previous_close else 0
                ),
                "pct": float(
                    (info.last_price / info.previous_close - 1) * 100
                    if info.previous_close
                    else 0
                ),
                "volume": float(info.three_month_average_volume or 0),
                "high": float(info.day_high or 0),
                "low": float(info.day_low or 0),
            }
        except Exception as e:
            log.warning(f"yfinance quote fail for {ticker}: {e}")
        return None

    @classmethod
    def get_quote(cls, ticker: str) -> Optional[Dict]:
        """Try NSE API → yfinance. Cache result."""
        quote = cls.fetch_nse_quote(ticker) or cls.fetch_yf_quote(ticker)
        if quote and quote["price"] > 0:
            store.cache_price(ticker, quote["price"], quote.get("volume", 0))
        return quote

    @staticmethod
    def get_historical(
        tickers: List[str], period: str = "2y", interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Returns adjusted close DataFrame (dates × tickers).
        Downloads via yfinance; caches to SQLite.
        """
        log.info(f"Fetching historical data: {tickers}, period={period}")
        raw = yf.download(
            tickers,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        if isinstance(raw.columns, pd.MultiIndex):
            closes = raw["Close"].copy()
        else:
            closes = raw[["Close"]].rename(columns={"Close": tickers[0]})

        closes.dropna(how="all", inplace=True)
        # Cache to SQLite
        for ticker in tickers:
            if ticker in closes.columns:
                df_t = (
                    raw.xs(ticker, axis=1, level=1)
                    if isinstance(raw.columns, pd.MultiIndex)
                    else raw
                )
                store.upsert_ohlcv(ticker, df_t)
        return closes

    @staticmethod
    def get_nifty50() -> pd.Series:
        """Nifty50 index historical closes."""
        df = yf.download(
            "^NSEI", period="2y", interval="1d", auto_adjust=True, progress=False
        )
        return df["Close"].dropna()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Async Stream (Polling Loop)
# ─────────────────────────────────────────────────────────────────────────────


class AsyncDataStream:
    """
    Continuously polls NSE prices and publishes to asyncio.Queue.
    Consumers: Dashboard refresh, risk engine triggers.
    """

    def __init__(self, tickers: List[str], interval: int = POLL_INTERVAL):
        self.tickers = tickers
        self.interval = interval
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=500)
        self._running = False

    async def _poll_once(self):
        quotes = {}
        for ticker in self.tickers:
            q = DataFetcher.get_quote(ticker)
            if q:
                quotes[ticker] = q
        if quotes:
            await self.queue.put({"ts": time.time(), "quotes": quotes})

    async def run(self):
        self._running = True
        log.info(f"Stream started: {self.tickers}")
        while self._running:
            try:
                await self._poll_once()
            except Exception as e:
                log.error(f"Stream error: {e}")
            await asyncio.sleep(self.interval)

    def stop(self):
        self._running = False


# ─────────────────────────────────────────────────────────────────────────────
# 5. Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────


class FeatureEngineer:
    """
    Computes derived features from price DataFrames.
    All operations vectorized with NumPy/Pandas.
    """

    @staticmethod
    def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """r_t = ln(S_t / S_{t-1})"""
        return np.log(prices / prices.shift(1)).dropna()

    @staticmethod
    def rolling_volatility(returns: pd.DataFrame, window: int = 21) -> pd.DataFrame:
        """σ_t = std(r) * √252, rolling window."""
        return returns.rolling(window).std() * np.sqrt(TRADING_DAYS)

    @staticmethod
    def rolling_correlation(returns: pd.DataFrame, window: int = 63) -> pd.DataFrame:
        """63-day (1Q) rolling correlation matrix (most recent window)."""
        return returns.iloc[-window:].corr()

    @staticmethod
    def momentum(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Price momentum: (S_t - S_{t-w}) / S_{t-w}"""
        return (prices - prices.shift(window)) / prices.shift(window)

    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index."""
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - 100 / (1 + rs)

    @staticmethod
    def bollinger_bands(
        prices: pd.Series, window: int = 20, n_std: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Returns (upper, mid, lower) Bollinger Bands."""
        mid = prices.rolling(window).mean()
        sigma = prices.rolling(window).std()
        return mid + n_std * sigma, mid, mid - n_std * sigma

    @classmethod
    def full_feature_matrix(cls, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Build concatenated feature matrix for ML/PCA:
          - Log returns, 5/21/63d volatility, momentum(5,20), RSI, BB-z
        """
        rets = cls.log_returns(prices)
        vol5 = rets.rolling(5).std() * np.sqrt(TRADING_DAYS)
        vol21 = rets.rolling(21).std() * np.sqrt(TRADING_DAYS)
        vol63 = rets.rolling(63).std() * np.sqrt(TRADING_DAYS)
        mom5 = cls.momentum(prices, 5)
        mom20 = cls.momentum(prices, 20)

        blocks = []
        for col in prices.columns:
            r5_c = vol5[col].rename(f"{col}_vol5")
            r21_c = vol21[col].rename(f"{col}_vol21")
            r63_c = vol63[col].rename(f"{col}_vol63")
            m5_c = mom5[col].rename(f"{col}_mom5")
            m20_c = mom20[col].rename(f"{col}_mom20")
            rsi_c = cls.rsi(prices[col]).rename(f"{col}_rsi")
            blocks += [r5_c, r21_c, r63_c, m5_c, m20_c, rsi_c]

        feat = pd.concat(blocks, axis=1).dropna()
        return feat


# ─────────────────────────────────────────────────────────────────────────────
# 6. Convenience entry-point
# ─────────────────────────────────────────────────────────────────────────────


def load_universe(
    tickers: Optional[List[str]] = None, period: str = "2y"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns: (prices, returns, features)
    """
    tickers = tickers or NSE_TICKERS
    prices = DataFetcher.get_historical(tickers, period=period)
    prices = prices.dropna(axis=1, thresh=int(0.8 * len(prices)))  # drop sparse
    prices = prices.ffill().bfill()
    rets = FeatureEngineer.log_returns(prices)
    feats = FeatureEngineer.full_feature_matrix(prices)
    log.info(f"Universe loaded: {prices.shape[1]} assets, {len(prices)} days")
    return prices, rets, feats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    prices, returns, features = load_universe(NSE_TICKERS[:5], period="1y")
    print(prices.tail(3))
    print(f"Features: {features.shape}")
