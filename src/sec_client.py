"""Thin, cached wrapper around SEC EDGAR's free JSON APIs.

Endpoints used (no API key needed):
- https://www.sec.gov/files/company_tickers.json       ticker -> CIK map
- https://data.sec.gov/submissions/CIK##########.json   filing history
- https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json
                                                        every XBRL-tagged number
                                                        the company has filed

SEC fair-access rules: every request must send a User-Agent that identifies
you with a contact email (SEC_USER_AGENT), and stay under 10 requests/second.
Requests are paced and retried on 429/5xx; responses are cached on disk.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import requests

from .config import config
from .disk_cache import DiskCache

logger = logging.getLogger(__name__)

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
EXCHANGE_TICKERS_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/{name}"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"


class SECAPIError(Exception):
    """Raised when EDGAR cannot be reached or returns an error."""


class SECAccessError(SECAPIError):
    """Raised when EDGAR refuses the request (usually a missing/invalid User-Agent)."""


def normalize_symbol(symbol: str) -> str:
    # EDGAR and Yahoo both write class shares with a dash: BRK-B, not BRK.B.
    return symbol.strip().upper().replace(".", "-")


class SECClient:
    # Shared across instances so concurrent training runs still respect the
    # SEC's global per-client rate limit.
    _rate_lock = threading.Lock()
    _last_request_at = 0.0

    def __init__(self, user_agent: str | None = None):
        self.user_agent = user_agent or config.SEC_USER_AGENT
        if not self.user_agent or "@" not in self.user_agent:
            raise SECAPIError(
                "SEC_USER_AGENT must be set to something like "
                "'YourName your.email@example.com' (SEC requires a contact email)."
            )
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate",
        })
        self.cache = DiskCache("sec")
        self._ticker_map: dict[str, int] | None = None

    def _throttle(self):
        with SECClient._rate_lock:
            wait = SECClient._last_request_at + config.SEC_REQUEST_INTERVAL_SECONDS - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            SECClient._last_request_at = time.monotonic()

    def _get(self, url: str) -> Any:
        cached = self.cache.get(url)
        if cached is not None:
            return cached

        max_retries = max(1, config.SEC_MAX_RETRIES)
        last_exc: Exception | None = None
        for attempt in range(1, max_retries + 1):
            self._throttle()
            try:
                resp = self.session.get(url, timeout=config.SEC_TIMEOUT_SECONDS)
            except requests.RequestException as exc:
                last_exc = exc
            else:
                if resp.status_code == 404:
                    raise SECAPIError(f"EDGAR has no data at {url}")
                if resp.status_code == 403:
                    raise SECAccessError(
                        f"EDGAR returned 403 for {url}. Check SEC_USER_AGENT is "
                        "'Name email@domain' and that you're under 10 requests/second."
                    )
                if resp.status_code == 429 or resp.status_code >= 500:
                    last_exc = SECAPIError(f"EDGAR returned {resp.status_code} for {url}")
                else:
                    try:
                        resp.raise_for_status()
                        data = resp.json()
                    except (requests.HTTPError, ValueError) as exc:
                        raise SECAPIError(f"Bad response from {url}: {exc}") from exc
                    self.cache.set(url, data)
                    return data

            if attempt < max_retries:
                wait = 2 ** attempt
                logger.warning("EDGAR request failed (%s), retry in %ss: %s", url, wait, last_exc)
                time.sleep(wait)

        raise SECAPIError(f"Failed to fetch {url} after {max_retries} attempts: {last_exc}")

    # ------------------------------------------------------------------- lookup
    def get_cik(self, symbol: str) -> int:
        if self._ticker_map is None:
            raw = self._get(TICKERS_URL)
            self._ticker_map = {
                normalize_symbol(row["ticker"]): int(row["cik_str"]) for row in raw.values()
            }
        cik = self._ticker_map.get(normalize_symbol(symbol))
        if cik is None:
            raise SECAPIError(f"Ticker {symbol} not found in EDGAR's ticker list")
        return cik

    def get_listed_companies(self) -> list[dict]:
        """Every ticker EDGAR knows, with its exchange: [{cik, name, ticker,
        exchange}], in EDGAR's order (roughly largest company first)."""
        raw = self._get(EXCHANGE_TICKERS_URL)
        return [dict(zip(raw["fields"], row)) for row in raw["data"]]

    def has_cached_facts(self, cik: int) -> bool:
        return self.cache.has(FACTS_URL.format(cik=f"{cik:010d}"))

    # -------------------------------------------------------------- submissions
    def get_filings(self, cik: int) -> list[dict]:
        """Full filing history as a list of dicts (one per filing), oldest
        pages included -- the `recent` block only covers ~1000 filings."""
        root = self._get(SUBMISSIONS_URL.format(name=f"CIK{cik:010d}.json"))
        pages = [root["filings"]["recent"]]
        for extra in root["filings"].get("files", []):
            pages.append(self._get(SUBMISSIONS_URL.format(name=extra["name"])))

        filings = []
        for page in pages:
            keys = list(page.keys())
            for values in zip(*(page[k] for k in keys)):
                filings.append(dict(zip(keys, values)))
        return filings

    def files_10q(self, cik: int) -> bool:
        """Whether the company's recent filings include a 10-Q or 10-K
        (foreign issuers file 20-F/40-F instead, which the pipeline can't use)."""
        root = self._get(SUBMISSIONS_URL.format(name=f"CIK{cik:010d}.json"))
        return bool({"10-Q", "10-K"} & set(root["filings"]["recent"].get("form", [])))

    def get_company_profile(self, cik: int) -> dict:
        """Name and SIC industry code from the submissions header (same
        cached response as get_filings, so no extra request)."""
        root = self._get(SUBMISSIONS_URL.format(name=f"CIK{cik:010d}.json"))
        sic = root.get("sic")
        return {
            "name": root.get("name"),
            "sic": int(sic) if str(sic or "").isdigit() else None,
            "sic_description": root.get("sicDescription"),
        }

    # --------------------------------------------------------------------- XBRL
    def get_company_facts(self, cik: int) -> dict:
        return self._get(FACTS_URL.format(cik=f"{cik:010d}"))
