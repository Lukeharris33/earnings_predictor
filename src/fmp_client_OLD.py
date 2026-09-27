"""
Thin wrapper around the Financial Modeling Prep "stable" API.

Endpoints used:
  /earnings                    -> report date, EPS/revenue actual vs estimate
  /income-statement            -> quarterly income statement (all numeric fields)
  /balance-sheet-statement     -> quarterly balance sheet (all numeric fields)
  /cash-flow-statement         -> quarterly cash flow statement (all numeric fields)
  /historical-price-eod/full   -> daily OHLCV history

All numeric fields returned by the statement endpoints are treated as model
features -- that's what "all the numerical values of an earnings report" maps to.
"""
import time
import logging
import requests

from .config import config

logger = logging.getLogger(__name__)


class FMPAPIError(Exception):
    """Raised when the FMP API can't be reached or returns an error payload."""


class FMPClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or config.FMP_API_KEY
        if not self.api_key:
            raise FMPAPIError("FMP_API_KEY is not set. Add it to your .env file.")
        self.session = requests.Session()

    def _get(self, path: str, params: dict) -> list | dict:
        params = {**params, "apikey": self.api_key}
        url = f"{config.FMP_BASE_URL}/{path}"
        last_exc = None
        for attempt in range(1, config.FMP_MAX_RETRIES + 1):
            try:
                resp = self.session.get(url, params=params, timeout=config.FMP_TIMEOUT_SECONDS)
                if resp.status_code == 429:
                    wait = 2**attempt
                    logger.warning("FMP rate limited on %s, backing off %ss", path, wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, dict) and data.get("Error Message"):
                    raise FMPAPIError(data["Error Message"])
                time.sleep(config.FMP_REQUEST_DELAY_SECONDS)
                return data
            except requests.RequestException as exc:
                last_exc = exc
                wait = 2**attempt
                logger.warning("FMP request failed (%s), retry in %ss: %s", path, wait, exc)
                time.sleep(wait)
        raise FMPAPIError(f"Failed to fetch {path} after {config.FMP_MAX_RETRIES} attempts: {last_exc}")

    # ------------------------------------------------------------------ earnings
    def get_earnings(self, symbol: str, limit: int = 60) -> list[dict]:
        """Historical earnings report dates + EPS/revenue actual vs estimate."""
        data = self._get("earnings", {"symbol": symbol, "limit": limit})
        return data if isinstance(data, list) else []

    # --------------------------------------------------------------- statements
    def get_income_statement(self, symbol: str, period: str = "quarter", limit: int = 60) -> list[dict]:
        data = self._get("income-statement", {"symbol": symbol, "period": period, "limit": limit})
        return data if isinstance(data, list) else []

    def get_balance_sheet(self, symbol: str, period: str = "quarter", limit: int = 60) -> list[dict]:
        data = self._get("balance-sheet-statement", {"symbol": symbol, "period": period, "limit": limit})
        return data if isinstance(data, list) else []

    def get_cash_flow(self, symbol: str, period: str = "quarter", limit: int = 60) -> list[dict]:
        data = self._get("cash-flow-statement", {"symbol": symbol, "period": period, "limit": limit})
        return data if isinstance(data, list) else []

    # ------------------------------------------------------------------- prices
    def get_historical_prices(self, symbol: str, date_from: str, date_to: str) -> list[dict]:
        """Daily OHLCV between two ISO dates (inclusive), oldest data may vary by plan."""
        data = self._get(
            "historical-price-eod/full",
            {"symbol": symbol, "from": date_from, "to": date_to},
        )
        if isinstance(data, dict):
            data = data.get("historical", [])
        return data if isinstance(data, list) else []
