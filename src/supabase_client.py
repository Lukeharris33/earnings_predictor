"""
Everything that touches Supabase lives here: training run bookkeeping, the
full training-data audit trail, error/warning logs, and saved model
artifacts (zipped and pushed to Supabase Storage, with a row in `models`
pointing at the storage path).

See schema.sql for the table definitions this expects.
"""
from __future__ import annotations

import io
import logging
import os
import zipfile
from datetime import datetime, timezone

from supabase import create_client, Client

from .config import config

logger = logging.getLogger(__name__)

_BATCH_SIZE = 200  # rows per insert call, keeps request payloads reasonable


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SupabaseLogger:
    def __init__(self):
        missing = config.validate()
        if missing:
            raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")
        self.client: Client = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY)

    # ------------------------------------------------------------- training runs
    def create_training_run(self, symbols: list[str], run_config: dict) -> str:
        res = (
            self.client.table("training_runs")
            .insert({
                "status": "running",
                "symbols": symbols,
                "config": run_config,
                "started_at": _now_iso(),
            })
            .execute()
        )
        return res.data[0]["id"]

    def update_training_run(self, run_id: str, **fields):
        self.client.table("training_runs").update(fields).eq("id", run_id).execute()

    def finish_training_run(self, run_id: str, status: str, metrics: dict | None = None, error_count: int = 0):
        self.update_training_run(
            run_id,
            status=status,
            metrics=metrics or {},
            error_count=error_count,
            finished_at=_now_iso(),
        )

    # ------------------------------------------------------------------- events
    def log_events(self, run_id: str | None, events: list) -> int:
        """events: list of PipelineEvent-like objects with .level/.symbol/.stage/.message/.traceback"""
        rows = [
            {
                "run_id": run_id,
                "symbol": e.symbol,
                "stage": e.stage,
                "level": e.level,
                "message": e.message,
                "traceback": e.traceback,
            }
            for e in events
        ]
        error_count = sum(1 for e in events if e.level == "error")
        for i in range(0, len(rows), _BATCH_SIZE):
            batch = rows[i : i + _BATCH_SIZE]
            if batch:
                self.client.table("errors").insert(batch).execute()
        return error_count

    def log_error(self, run_id: str | None, symbol: str, stage: str, message: str, tb: str | None = None):
        self.client.table("errors").insert({
            "run_id": run_id, "symbol": symbol, "stage": stage,
            "level": "error", "message": message, "traceback": tb,
        }).execute()

    # ------------------------------------------------------------- training data
    def save_training_data(self, run_id: str, df, feature_columns: list[str], horizon_labels: list[str]):
        rows = []
        for _, r in df.iterrows():
            features = {c: (None if _is_nan(r[c]) else float(r[c])) for c in feature_columns}
            targets = {h: (None if _is_nan(r[f"target_{h}"]) else float(r[f"target_{h}"])) for h in horizon_labels}
            rows.append({
                "run_id": run_id,
                "symbol": r["symbol"],
                "earnings_date": r["earnings_date"],
                "features": features,
                "targets": targets,
            })
        for i in range(0, len(rows), _BATCH_SIZE):
            batch = rows[i : i + _BATCH_SIZE]
            if batch:
                self.client.table("training_data").insert(batch).execute()
        return len(rows)

    # ------------------------------------------------------------------- models
    def save_model(self, run_id: str, name: str, local_model_dir: str, metadata: dict, storage_name: str | None = None) -> str:
        zip_bytes = _zip_directory(local_model_dir)
        storage_path = f"{storage_name or name}.zip"

        self.client.storage.from_(config.SUPABASE_MODEL_BUCKET).upload(
            storage_path,
            zip_bytes,
            {"content-type": "application/zip", "upsert": "true"},
        )

        res = (
            self.client.table("models")
            .insert({
                "run_id": run_id,
                "name": name,
                "storage_path": storage_path,
                "feature_columns": metadata["feature_columns"],
                "target_horizons": metadata["target_horizons"],
                "target_scale": metadata["target_scale"],
                "metrics": metadata["metrics"],
                "symbols": metadata["symbols"],
            })
            .execute()
        )
        return res.data[0]["id"]

    def update_model_metrics(self, model_id: str, metrics: dict):
        self.client.table("models").update({"metrics": metrics}).eq("id", model_id).execute()

    def run_configs(self, run_ids: list[str]) -> dict[str, dict]:
        """run id -> the config it was trained with."""
        ids = [i for i in run_ids if i]
        if not ids:
            return {}
        res = self.client.table("training_runs").select("id, config").in_("id", ids).execute()
        return {r["id"]: r.get("config") or {} for r in res.data}

    def get_training_run(self, run_id: str) -> dict:
        res = self.client.table("training_runs").select("*").eq("id", run_id).single().execute()
        return res.data

    def fetch_training_data(self, run_id: str, page: int = 1000, progress=None) -> list[dict]:
        """Every stored training row for a run (the API returns at most 1000 per request)."""
        rows, start = [], 0
        while True:
            res = (
                self.client.table("training_data")
                .select("symbol, earnings_date, features, targets")
                .eq("run_id", run_id).order("id").range(start, start + page - 1).execute()
            )
            rows.extend(res.data)
            if progress:
                progress(len(rows))
            if len(res.data) < page:
                return rows
            start += page

    def fetch_prediction_outcomes(self, page: int = 1000) -> list[dict]:
        """Every saved prediction with just what's needed to score it later."""
        rows, start = [], 0
        cols = ("id, model_id, symbol, earnings_date, decisions, created_at, "
                "baseline_date:details->source->>baseline_price_date, day1_date:details->source->>day1_date")
        while True:
            res = (
                self.client.table("predictions").select(cols)
                .order("id").range(start, start + page - 1).execute()
            )
            rows.extend(res.data)
            if len(res.data) < page:
                return rows
            start += page

    def list_models(self) -> list[dict]:
        res = self.client.table("models").select("*").order("created_at", desc=True).execute()
        return res.data

    def get_model(self, model_id: str) -> dict:
        res = self.client.table("models").select("*").eq("id", model_id).single().execute()
        return res.data

    def download_model(self, model_id: str, dest_dir: str) -> str:
        row = self.get_model(model_id)
        zip_bytes = self.client.storage.from_(config.SUPABASE_MODEL_BUCKET).download(row["storage_path"])
        os.makedirs(dest_dir, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            zf.extractall(dest_dir)
        return dest_dir

    # -------------------------------------------------------------- predictions
    def save_prediction(
        self,
        model_id: str,
        symbol: str,
        earnings_date: str | None,
        predicted: dict,
        decisions: dict,
        name: str | None = None,
        batch_id: str | None = None,
        details: dict | None = None,
    ) -> int:
        row = {
            "model_id": model_id,
            "symbol": symbol,
            "earnings_date": earnings_date,
            "predicted": predicted,
            "decisions": decisions,
            "name": name,
            "batch_id": batch_id,
            "details": details,
        }
        try:
            res = self.client.table("predictions").insert(row).execute()
        except Exception as exc:
            _raise_if_missing_migration(exc)
            raise
        return res.data[0]["id"]

    def list_predictions(self, limit: int = 30) -> list[dict]:
        try:
            res = (
                self.client.table("predictions")
                .select("id, name, symbol, earnings_date, decisions, batch_id, model_id, created_at, models(name)")
                .order("created_at", desc=True).limit(limit).execute()
            )
        except Exception as exc:
            _raise_if_missing_migration(exc)
            raise
        return res.data

    # ------------------------------------------------------ prediction batches
    def create_batch(self, name: str, model_id: str, symbols: list[str]) -> str:
        try:
            res = self.client.table("prediction_batches").insert({
                "name": name, "model_id": model_id, "symbols": symbols, "status": "running",
            }).execute()
        except Exception as exc:
            _raise_if_missing_migration(exc)
            raise
        return res.data[0]["id"]

    def finish_batch(self, batch_id: str, status: str, summary: dict):
        self.client.table("prediction_batches").update({
            "status": status, "summary": summary, "finished_at": _now_iso(),
        }).eq("id", batch_id).execute()

    def list_batches(self, limit: int = 30) -> list[dict]:
        try:
            res = (
                self.client.table("prediction_batches")
                .select("id, name, model_id, symbols, status, summary, created_at, finished_at, models(name)")
                .order("created_at", desc=True).limit(limit).execute()
            )
        except Exception as exc:
            _raise_if_missing_migration(exc)
            raise
        return res.data

    def get_batch(self, batch_id: str) -> dict:
        batch = (
            self.client.table("prediction_batches")
            .select("*, models(name)").eq("id", batch_id).single().execute()
        ).data
        batch["predictions"] = (
            self.client.table("predictions")
            .select("id, name, symbol, earnings_date, predicted, decisions, details, created_at")
            .eq("batch_id", batch_id).order("symbol").execute()
        ).data
        return batch

    def list_training_runs(self) -> list[dict]:
        res = self.client.table("training_runs").select("*").order("started_at", desc=True).execute()
        return res.data


def _is_nan(v) -> bool:
    if v is None:
        return True
    try:
        return v != v  # NaN != NaN
    except Exception:
        return False


class MigrationRequired(RuntimeError):
    pass


def _raise_if_missing_migration(exc: Exception):
    text = str(exc)
    markers = ("prediction_batches", "batch_id", "details", "PGRST204", "PGRST205", "42P01", "42703")
    if any(m in text for m in markers):
        raise MigrationRequired(
            "Supabase is missing the named/batch prediction tables. Run "
            "migrations/002_named_and_batch_predictions.sql in the Supabase SQL editor."
        ) from exc


def _zip_directory(local_dir: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(local_dir):
            for f in files:
                full_path = os.path.join(root, f)
                arcname = os.path.relpath(full_path, local_dir)
                zf.write(full_path, arcname)
    buf.seek(0)
    return buf.read()
