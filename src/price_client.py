"""Daily closing prices and split history from Yahoo Finance (via yfinance).
EDGAR has no market data, so prices come from here.

Adjusted closes matter: an unadjusted series turns a 4-for-1 split into a
fake -75% "return" in the training labels. Returns use the split- and
dividend-adjusted close; valuation (market cap) uses the split-adjusted
close scaled back to the share count the filing reported.
"""

from __future__ import annotations

import logging
import threading
import time

import pandas as pd
import yfinance as yf

from .disk_cache import DiskCache
from .sec_client import normalize_symbol

logger = logging.getLogger(__name__)


class PriceAPIError(Exception):
    """Raised when prices can't be downloaded."""


LIVE_TTL_SECONDS = 30


class PriceClient:
    # Shared so several open testing pages don't each hit Yahoo every refresh.
    _live_cache: dict[str, tuple[float, dict]] = {}
    _live_lock = threading.Lock()

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self.cache = DiskCache("prices_v2")

    def get_adjusted_closes(self, symbol: str, date_from: str, date_to: str) -> pd.Series:
        """Split- and dividend-adjusted close indexed by (tz-naive) trading date, ascending."""
        return self.get_prices(symbol, date_from, date_to)["adj_close"]

    def get_prices(self, symbol: str, date_from: str, date_to: str) -> pd.DataFrame:
        """Daily `adj_close` (split + dividend adjusted, for returns) and
        `close` (split adjusted only, for valuation), ascending by date."""
        symbol = normalize_symbol(symbol)
        key = f"{symbol}|{date_from}|{date_to}"
        cached = self.cache.get(key)
        if cached is None:
            cached = self._download(symbol, date_from, date_to)
            self.cache.set(key, cached)
        if not cached:
            return pd.DataFrame({"adj_close": [], "close": []}, dtype="float64")
        dates, adj, close = zip(*cached)
        return pd.DataFrame(
            {"adj_close": adj, "close": close}, index=pd.DatetimeIndex(dates), dtype="float64"
        ).sort_index()

    def get_splits(self, symbol: str) -> pd.Series:
        """Every stock split in the symbol's history: date -> ratio (4.0 for 4-for-1)."""
        symbol = normalize_symbol(symbol)
        key = f"splits|{symbol}"
        cached = self.cache.get(key)
        if cached is None:
            cached = self._with_retries(
                symbol, lambda: [[d.strftime("%Y-%m-%d"), float(r)] for d, r in yf.Ticker(symbol).splits.items() if r]
            )
            self.cache.set(key, cached)
        if not cached:
            return pd.Series(dtype="float64")
        dates, ratios = zip(*cached)
        return pd.Series(ratios, index=pd.DatetimeIndex(dates), dtype="float64").sort_index()

    def get_live_quotes(self, symbols: list[str]) -> dict[str, dict]:
        """Latest traded price per symbol from 1-minute bars of the current
        (or most recent) session, in one request. Not disk-cached; repeated
        calls within LIVE_TTL_SECONDS reuse the last answer."""
        symbols = sorted({normalize_symbol(s) for s in symbols if s})
        key = ",".join(symbols)
        now = time.monotonic()
        with PriceClient._live_lock:
            hit = PriceClient._live_cache.get(key)
            if hit and now - hit[0] < LIVE_TTL_SECONDS:
                return hit[1]

        def fetch():
            bars = yf.download(
                symbols, period="1d", interval="1m", progress=False,
                group_by="ticker", auto_adjust=False, threads=True,
            )
            out = {}
            for sym in symbols:
                try:
                    closes = bars[sym]["Close"].dropna()  # group_by="ticker" keys columns by symbol
                except KeyError:
                    continue
                if not closes.empty:
                    out[sym] = {"price": float(closes.iloc[-1]), "time": closes.index[-1].isoformat()}
            return out

        quotes = self._with_retries(key, fetch)
        with PriceClient._live_lock:
            PriceClient._live_cache[key] = (now, quotes)
        return quotes

    def _download(self, symbol: str, date_from: str, date_to: str) -> list[list]:
        def fetch():
            hist = yf.Ticker(symbol).history(
                start=date_from, end=date_to, interval="1d",
                auto_adjust=False, actions=False, raise_errors=True,
            )
            hist = hist[["Adj Close", "Close"]].dropna()
            return [[d.strftime("%Y-%m-%d"), float(a), float(c)] for d, a, c in hist.itertuples()]
        return self._with_retries(symbol, fetch)

    def _with_retries(self, symbol: str, fn):
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return fn()
            except Exception as exc:  # yfinance raises a zoo of exception types
                last_exc = exc
                if attempt < self.max_retries:
                    wait = 2 ** attempt
                    logger.warning("Price download for %s failed, retry in %ss: %s", symbol, wait, exc)
                    time.sleep(wait)
        raise PriceAPIError(f"Could not download prices for {symbol}: {last_exc}")
