"""
Central configuration. Everything is loaded from environment variables
(see .env.example) so no secrets ever live in source code.
"""
import os
import tempfile
from dotenv import load_dotenv

load_dotenv()


def _require(name: str, default=None):
    val = os.environ.get(name, default)
    return val


class Config:
    # --- SEC EDGAR -----------------------------------------------------------------
    # SEC requires a User-Agent naming you with a contact email, e.g.
    # "Jane Doe jane@example.com". Requests without one get a 403.
    SEC_USER_AGENT = _require("SEC_USER_AGENT")
    SEC_MAX_RETRIES = int(_require("SEC_MAX_RETRIES", 3))
    # SEC allows at most 10 requests/second.
    SEC_REQUEST_INTERVAL_SECONDS = float(_require("SEC_REQUEST_INTERVAL_SECONDS", 0.15))
    SEC_TIMEOUT_SECONDS = int(_require("SEC_TIMEOUT_SECONDS", 30))

    # --- Disk cache for SEC + price downloads ------------------------------------
    CACHE_ENABLED = _require("CACHE_ENABLED", "true").lower() not in {"0", "false", "no", "off"}
    CACHE_TTL_SECONDS = int(_require("CACHE_TTL_SECONDS", 86400))
    CACHE_DIR = _require("CACHE_DIR", os.path.join("data", "cache"))

    # --- Supabase ------------------------------------------------------------
    SUPABASE_URL = _require("SUPABASE_URL")
    # Use the service_role key (server-side only, never exposed to a browser).
    SUPABASE_SERVICE_KEY = _require("SUPABASE_SERVICE_KEY")
    SUPABASE_MODEL_BUCKET = _require("SUPABASE_MODEL_BUCKET", "models")

    # --- Flask -----------------------------------------------------------------
    SECRET_KEY = _require("SECRET_KEY", "dev-secret-change-me")
    # Debug mode exposes the Werkzeug debugger (arbitrary code execution), so
    # it's off by default and the server only listens on localhost.
    FLASK_DEBUG = _require("FLASK_DEBUG", "false").lower() == "true"
    FLASK_HOST = _require("FLASK_HOST", "127.0.0.1")
    FLASK_PORT = int(_require("FLASK_PORT", 5000))

    # --- Modeling ----------------------------------------------------------------
    # Forward-looking windows expressed in *trading days*, not calendar days.
    HORIZONS = {
        "1d": 1,
        "1w": 5,
        "1m": 21,
        "1y": 252,
    }
    # Targets are returns in excess of this benchmark over the same days.
    BENCHMARK_SYMBOL = _require("BENCHMARK_SYMBOL", "SPY")
    # Minimum number of usable rows before we'll bother training.
    MIN_TRAINING_ROWS = 30
    # Walk-forward evaluation: each of the last N calendar years is predicted
    # by a model trained only on earlier earnings (with overlapping labels
    # purged). Falls back to one chronological split when there's too little
    # history for that.
    WALK_FORWARD_YEARS = int(_require("WALK_FORWARD_YEARS", 4))
    MIN_FOLD_TEST_ROWS = 25
    # Chronological holdout fraction for the fallback split, and the slice of
    # each training set used for neural-net early stopping.
    VALIDATION_FRACTION = 0.15
    DEFAULT_EPOCHS = 150
    DEFAULT_BATCH_SIZE = 64
    # Networks trained with different random seeds and averaged.
    ENSEMBLE_SEEDS = int(_require("ENSEMBLE_SEEDS", 5))
    # "blend" (average of network ensemble + gradient-boosted trees), "nn", or "gbm".
    DEFAULT_MODEL_TYPE = "blend"
    MODEL_TYPES = ("blend", "nn", "gbm")
    # Predicted moves within this many standard deviations of zero are HOLD.
    HOLD_BAND_STD = float(_require("HOLD_BAND_STD", 0.25))
    # Companies needed in a calendar quarter before peer percentiles are used.
    MIN_PEERS_PER_QUARTER = 8
    MAX_TRAINING_TICKERS = 600
    MAX_BATCH_TICKERS = 200

    # Local scratch space for models before they're uploaded to Supabase Storage.
    LOCAL_MODEL_DIR = _require(
        "LOCAL_MODEL_DIR", os.path.join(tempfile.gettempdir(), "earnings_predictor_models")
    )

    @classmethod
    def validate(cls):
        missing = [
            name
            for name in ("SEC_USER_AGENT", "SUPABASE_URL", "SUPABASE_SERVICE_KEY")
            if not getattr(cls, name)
        ]
        return missing


config = Config()
