import logging
import traceback

from flask import Flask, jsonify, render_template, request

from src import batch, diagnostics, ranking, testing_rig, trainer
from src.config import config
from src.data_pipeline import annotate_windows
from src.predictor import predict_for_symbol
from src.price_client import PriceClient
from src.supabase_client import MigrationRequired, SupabaseLogger
from src.universe import HOLDOUT_TICKERS, STARTER_UNIVERSE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["SECRET_KEY"] = config.SECRET_KEY


def _db_or_none():
    try:
        return SupabaseLogger()
    except Exception as exc:
        logger.warning("Supabase not configured: %s", exc)
        return None


def _log_route_error(stage: str, symbol: str, exc: Exception):
    db = _db_or_none()
    if db:
        try:
            db.log_error(None, symbol, stage, str(exc), traceback.format_exc())
        except Exception:
            logger.exception("Failed to log route error to Supabase")


def _parse_symbols(raw) -> list[str]:
    if isinstance(raw, list):
        raw = ",".join(raw)
    seen, out = set(), []
    for s in (raw or "").replace("\n", ",").replace(" ", ",").split(","):
        s = s.strip().upper()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


@app.route("/")
def index():
    missing_config = config.validate()
    models, runs, batches, predictions = [], [], [], []
    migration_needed = False
    if not missing_config:
        db = _db_or_none()
        if db:
            try:
                models = db.list_models()
                runs = db.list_training_runs()[:20]
            except Exception:
                logger.exception("Failed to load dashboard data")
            try:
                batches = db.list_batches(20)
                predictions = db.list_predictions(20)
                for p in predictions:
                    annotate_windows(p.get("decisions"), p.get("earnings_date"))
            except MigrationRequired:
                migration_needed = True
            except Exception:
                logger.exception("Failed to load predictions")
    return render_template(
        "index.html",
        missing_config=missing_config,
        migration_needed=migration_needed,
        models=models,
        runs=runs,
        batches=batches,
        predictions=predictions,
        horizons=list(config.HORIZONS.keys()),
        model_types=config.MODEL_TYPES,
        universe_size=len(STARTER_UNIVERSE),
        holdout_size=len(HOLDOUT_TICKERS),
        benchmark=config.BENCHMARK_SYMBOL,
    )


@app.route("/testing")
def testing():
    missing_config = config.validate()
    models, batches = [], []
    migration_needed = False
    if not missing_config:
        db = _db_or_none()
        if db:
            try:
                models = db.list_models()
                batches = db.list_batches(50)
            except MigrationRequired:
                migration_needed = True
            except Exception:
                logger.exception("Failed to load testing data")
    return render_template(
        "testing.html",
        missing_config=missing_config,
        migration_needed=migration_needed,
        models=models,
        batches=batches,
        horizons=list(config.HORIZONS.keys()),
        benchmark=config.BENCHMARK_SYMBOL,
    )


def _ranking_entries(db) -> list[dict]:
    models = db.list_models()
    return ranking.summarize(models, db.run_configs([m.get("run_id") for m in models]))


@app.route("/ranking")
def ranking_page():
    missing_config = config.validate()
    entries, boards, common = [], {}, {}
    if not missing_config:
        db = _db_or_none()
        if db:
            try:
                entries = _ranking_entries(db)
                boards = ranking.leaderboards(entries)
                common = {f: ranking.commonalities(entries, f) for f in ranking.FOCUSES}
            except Exception:
                logger.exception("Failed to build the ranking")
    return render_template(
        "ranking.html",
        missing_config=missing_config,
        payload={"entries": entries, "boards": boards, "commonalities": common,
                 "horizons": list(config.HORIZONS), "group_labels": ranking.GROUP_LABELS},
        focuses=ranking.FOCUSES,
    )


@app.route("/api/ranking/compare")
def api_ranking_compare():
    db = _db_or_none()
    if not db:
        return jsonify({"error": "Supabase is not configured."}), 500
    focus = request.args.get("focus", "overall")
    if focus not in ranking.FOCUSES:
        return jsonify({"error": f"focus must be one of {', '.join(ranking.FOCUSES)}"}), 400
    try:
        byid = {e["id"]: e for e in _ranking_entries(db)}
        a, b = byid.get(request.args.get("a")), byid.get(request.args.get("b"))
        if not a or not b:
            return jsonify({"error": "Pick two saved models."}), 400
        return jsonify(ranking.compare(a, b, focus))
    except Exception as exc:
        logger.exception("Comparison failed")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/ranking/track-record")
def api_track_record():
    db = _db_or_none()
    if not db:
        return jsonify({"error": "Supabase is not configured."}), 500
    try:
        return jsonify(ranking.track_record(db, PriceClient()))
    except Exception as exc:
        logger.exception("Track record failed")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/models/<model_id>/diagnostics", methods=["POST"])
def api_start_diagnostics(model_id):
    try:
        return jsonify({"job_id": diagnostics.start_backfill(model_id)})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/diagnostics/<job_id>")
def api_diagnostics_status(job_id):
    status = diagnostics.get_status(job_id)
    if status is None:
        return jsonify({"error": "Unknown diagnostics job"}), 404
    status.pop("traceback", None)
    return jsonify(status)


@app.route("/api/testing/batch/<batch_id>")
def api_testing_batch(batch_id):
    db = _db_or_none()
    if not db:
        return jsonify({"error": "Supabase is not configured."}), 500
    try:
        return jsonify(testing_rig.batch_payload(batch_id, db, PriceClient()))
    except Exception as exc:
        logger.exception("Testing payload failed for batch %s", batch_id)
        return jsonify({"error": str(exc)}), 500


@app.route("/api/testing/live")
def api_testing_live():
    symbols = _parse_symbols(request.args.get("symbols", ""))[: config.MAX_BATCH_TICKERS]
    if not symbols:
        return jsonify({"error": "symbols is required"}), 400
    try:
        return jsonify(testing_rig.live_payload(symbols, PriceClient()))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 502


@app.route("/api/universe")
def api_universe():
    return jsonify({"universe": STARTER_UNIVERSE, "holdout": HOLDOUT_TICKERS})


@app.route("/api/train", methods=["POST"])
def api_train():
    payload = request.get_json(force=True, silent=True) or {}
    symbols = _parse_symbols(payload.get("symbols", ""))
    if not symbols:
        return jsonify({"error": "Provide at least one ticker symbol."}), 400
    if len(symbols) > config.MAX_TRAINING_TICKERS:
        return jsonify({"error": f"Train on at most {config.MAX_TRAINING_TICKERS} tickers at a time."}), 400
    try:
        epochs = int(payload.get("epochs") or config.DEFAULT_EPOCHS)
        batch_size = int(payload.get("batch_size") or config.DEFAULT_BATCH_SIZE)
    except (TypeError, ValueError):
        return jsonify({"error": "epochs and batch_size must be whole numbers."}), 400
    epochs = min(max(epochs, 1), 1000)
    batch_size = min(max(batch_size, 1), 1024)
    model_type = payload.get("model_type") or config.DEFAULT_MODEL_TYPE
    if model_type not in config.MODEL_TYPES:
        return jsonify({"error": f"model_type must be one of {', '.join(config.MODEL_TYPES)}."}), 400
    name = (payload.get("name") or "").strip()[:120] or None

    try:
        run_id = trainer.start_training_run(symbols, epochs, batch_size, name=name, model_type=model_type)
    except Exception as exc:
        _log_route_error("flask_route:train", ",".join(symbols), exc)
        return jsonify({"error": str(exc)}), 500

    return jsonify({"run_id": run_id})


@app.route("/api/status/<run_id>")
def api_status(run_id):
    status = trainer.get_status(run_id)
    if status is None:
        return jsonify({"error": "Unknown run id"}), 404
    return jsonify(status)


@app.route("/api/models")
def api_models():
    db = _db_or_none()
    if not db:
        return jsonify({"error": "Supabase is not configured."}), 500
    try:
        return jsonify(db.list_models())
    except Exception as exc:
        _log_route_error("flask_route:list_models", "*", exc)
        return jsonify({"error": str(exc)}), 500


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(force=True, silent=True) or {}
    model_id = payload.get("model_id")
    symbol = (payload.get("symbol") or "").strip().upper()
    name = (payload.get("name") or "").strip()[:120] or None
    if not model_id or not symbol:
        return jsonify({"error": "model_id and symbol are both required."}), 400

    try:
        result = predict_for_symbol(model_id, symbol, name=name)
        return jsonify(result)
    except Exception as exc:
        logger.exception("Prediction failed for %s", symbol)
        _log_route_error("flask_route:predict", symbol, exc)
        return jsonify({"error": str(exc)}), 500


@app.route("/api/predictions")
def api_predictions():
    db = _db_or_none()
    if not db:
        return jsonify({"error": "Supabase is not configured."}), 500
    try:
        predictions = db.list_predictions(30)
        for p in predictions:
            annotate_windows(p.get("decisions"), p.get("earnings_date"))
        return jsonify(predictions)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ------------------------------------------------------------------ batches
@app.route("/api/batch", methods=["POST"])
def api_batch():
    payload = request.get_json(force=True, silent=True) or {}
    model_id = payload.get("model_id")
    name = (payload.get("name") or "").strip()[:120]
    symbols = _parse_symbols(payload.get("symbols", ""))
    if not model_id or not name or not symbols:
        return jsonify({"error": "model_id, a batch name, and at least one ticker are required."}), 400
    if len(symbols) > config.MAX_BATCH_TICKERS:
        return jsonify({"error": f"At most {config.MAX_BATCH_TICKERS} tickers per batch."}), 400
    try:
        job_id = batch.start_batch(model_id, symbols, name)
    except Exception as exc:
        _log_route_error("flask_route:batch", ",".join(symbols), exc)
        return jsonify({"error": str(exc)}), 500
    return jsonify({"job_id": job_id})


@app.route("/api/batch/status/<job_id>")
def api_batch_status(job_id):
    status = batch.get_status(job_id)
    if status is None:
        return jsonify({"error": "Unknown batch job"}), 404
    return jsonify(status)


@app.route("/api/batches")
def api_batches():
    db = _db_or_none()
    if not db:
        return jsonify({"error": "Supabase is not configured."}), 500
    try:
        return jsonify(db.list_batches(30))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/batches/<batch_id>")
def api_batch_detail(batch_id):
    db = _db_or_none()
    if not db:
        return jsonify({"error": "Supabase is not configured."}), 500
    try:
        b = db.get_batch(batch_id)
        for p in b["predictions"]:
            annotate_windows(p.get("decisions"), p.get("earnings_date"))
        return jsonify(b)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=config.FLASK_DEBUG)
