"""Tiny JSON-on-disk cache shared by the SEC and price clients, so historical
data is downloaded once and reused across training runs."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from .config import config

logger = logging.getLogger(__name__)


class DiskCache:
    def __init__(self, namespace: str, ttl_seconds: int | None = None):
        self.enabled = config.CACHE_ENABLED
        self.ttl_seconds = config.CACHE_TTL_SECONDS if ttl_seconds is None else ttl_seconds
        self.dir = Path(config.CACHE_DIR) / namespace
        if self.enabled:
            self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
        safe = "".join(c if c.isalnum() else "_" for c in key)[:60]
        return self.dir / f"{safe}_{digest}.json"

    def has(self, key: str) -> bool:
        """Whether a file exists for `key`, fresh or not (an expired file is
        overwritten in place, so re-fetching it costs no extra disk)."""
        return self.enabled and self._path(key).exists()

    def get(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        path = self._path(key)
        if not path.exists() or time.time() - path.stat().st_mtime > self.ttl_seconds:
            return None
        try:
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring unreadable cache file %s: %s", path, exc)
            return None

    def set(self, key: str, data: Any) -> None:
        if not self.enabled:
            return
        path = self._path(key)
        tmp_path = path.with_suffix(".tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False)
            os.replace(tmp_path, path)
        except OSError as exc:
            logger.warning("Could not write cache file %s: %s", path, exc)
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
