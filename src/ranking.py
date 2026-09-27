"""
Ranks models by the strength of their out-of-sample evidence and explains
what separates them.

Score: a model's rank correlation (IC) between predicted and actual moves,
taken at the *lower end* of its 95% confidence range. A high IC measured on
a few hundred predictions has a wide range and so a low floor; a slightly
lower IC measured on thousands has a narrow range and a higher floor. That
stops a small, lucky test set from topping the table. "Overall" averages
the floors across horizons.

Head-to-head findings are associations between two trained models, not
proof of cause; each comes with the experiment that would test it.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd

from .config import config
from .data_pipeline import ET, annotate_windows
from .price_client import PriceClient

HORIZONS = list(config.HORIZONS)
FOCUSES = ["overall"] + HORIZONS
Z95 = 1.96

GROUP_LABELS = {
    "profitability": "profitability", "growth": "growth", "cost_structure": "cost structure",
    "cash_flow": "cash flow", "balance_sheet": "balance sheet", "size": "company size",
    "valuation": "valuation", "vs_own_trend": "results vs the company's own trend",
    "trend_quality": "multi-year trend quality", "price_momentum": "pre-release price momentum",
    "release_timing": "release timing", "first_day_reaction": "the first-day reaction", "sector": "sector",
}


# ---------------------------------------------------------------- statistics
def ic_interval(ic: float | None, n: int | None) -> tuple[float | None, float | None]:
    """95% range for a correlation via Fisher's z transform."""
    if ic is None or not n or n <= 10:
        return None, None
    z = math.atanh(max(min(ic, 0.999), -0.999))
    se = 1 / math.sqrt(n - 3)
    return math.tanh(z - Z95 * se), math.tanh(z + Z95 * se)


def ic_difference(a: dict, b: dict) -> tuple[float, float]:
    """(difference in IC, z-score) for two independent correlations."""
    za, zb = math.atanh(max(min(a["ic"], 0.999), -0.999)), math.atanh(max(min(b["ic"], 0.999), -0.999))
    se = math.sqrt(1 / (a["n"] - 3) + 1 / (b["n"] - 3))
    return a["ic"] - b["ic"], (za - zb) / se


def _horizon_stats(hm: dict) -> dict:
    ic, n = hm.get("ic"), hm.get("n")
    lo, hi = ic_interval(ic, n)
    acc, base = hm.get("directional_accuracy"), hm.get("baseline_majority_accuracy")
    edge = edge_margin = None
    if acc is not None and base is not None and n:
        edge = acc - base
        edge_margin = Z95 * math.sqrt(max(acc * (1 - acc), 1e-9) / n)
    folds = hm.get("folds") or []
    fold_ics = [f["ic"] for f in folds if f.get("ic") is not None]
    return {
        "ic": ic, "ic_lo": lo, "ic_hi": hi, "n": n,
        "accuracy": acc, "baseline": base, "edge": edge, "edge_margin": edge_margin,
        "years_positive": sum(v > 0 for v in fold_ics), "years": len(fold_ics),
        "fold_ics": {f["test_period"]: f.get("ic") for f in folds},
        "by_family": {k: v.get("ic") for k, v in (hm.get("by_family") or {}).items()},
        "spread": hm.get("top_minus_bottom_quintile"),
    }


# ------------------------------------------------------------------- summary
def summarize(models: list[dict], run_configs: dict[str, dict]) -> list[dict]:
    out = []
    for m in models:
        metrics = m.get("metrics") or {}
        hs = {
            h: _horizon_stats(metrics[h]) for h in HORIZONS
            if isinstance(metrics.get(h), dict) and metrics[h].get("ic") is not None and metrics[h].get("n")
        }
        diag = metrics.get("_diagnostics") or {}
        cfg = run_configs.get(m.get("run_id")) or {}
        v2 = isinstance(m.get("feature_columns"), dict)
        entry = {
            "id": m["id"],
            "name": m["name"],
            "created_at": m.get("created_at"),
            "tickers": len(m.get("symbols") or []),
            "symbols": m.get("symbols") or [],
            "version": 2 if v2 else 1,
            "config": {k: cfg.get(k) for k in ("model_type", "epochs", "batch_size")},
            "horizons": hs,
            "profile": diag.get("profile"),
            "importance": diag.get("importance"),
            "diagnostics_method": diag.get("method"),
            "rankable": bool(hs),
            "unrankable_reason": None if hs else (
                "Trained before walk-forward testing: its accuracy came from one small holdout, with no "
                "rank correlation or baseline, so it can't be compared fairly." if not v2
                else "No out-of-sample results were recorded."
            ),
            "v1_accuracy": {
                h: (metrics.get(h) or {}).get("directional_accuracy") for h in HORIZONS
            } if not v2 else None,
        }
        folds = [s for s in hs.values() if s["years"]]
        entry["consistency"] = (
            sum(s["years_positive"] for s in folds) / sum(s["years"] for s in folds) if folds else None
        )
        entry["scores"] = {f: _score(entry, f) for f in FOCUSES}
        out.append(entry)
    return out


def _score(entry: dict, focus: str) -> dict | None:
    hs = entry["horizons"]
    if focus != "overall":
        s = hs.get(focus)
        if not s or s["ic_lo"] is None:
            return None
        return {"score": s["ic_lo"], "ic": s["ic"], "lo": s["ic_lo"], "hi": s["ic_hi"], "n": s["n"], "partial": False}
    usable = [hs[h] for h in HORIZONS if h in hs and hs[h]["ic_lo"] is not None]
    if not usable:
        return None
    return {
        "score": float(np.mean([s["ic_lo"] for s in usable])),
        "ic": float(np.mean([s["ic"] for s in usable])),
        "lo": float(np.mean([s["ic_lo"] for s in usable])),
        "hi": float(np.mean([s["ic_hi"] for s in usable])),
        "n": int(sum(s["n"] for s in usable)),
        "partial": len(usable) < len(HORIZONS),
    }


def leaderboards(entries: list[dict]) -> dict[str, list[str]]:
    """focus -> model ids, best first; unrankable models last."""
    boards = {}
    for f in FOCUSES:
        ranked = sorted((e for e in entries if e["scores"].get(f)), key=lambda e: -e["scores"][f]["score"])
        rest = [e for e in entries if not e["scores"].get(f)]
        boards[f] = [e["id"] for e in ranked + rest]
    return boards


# ------------------------------------------------------------ commonalities
PROFILE_ATTRS = [
    ("tickers_used", "Tickers trained on"),
    ("rows", "Training rows"),
    ("years_of_history", "Years of history"),
    ("sectors_covered", "Sectors covered"),
    ("loss_making_share", "Share of loss-making quarters"),
    ("median_market_cap", "Median market cap"),
    ("small_cap_share", "Share under $2B market cap"),
    ("epochs", "Max epochs"),
    ("batch_size", "Batch size"),
]


def commonalities(entries: list[dict], focus: str = "overall") -> dict:
    """Rank correlation between each training attribute and the score, across
    ranked models that have a profile. Needs a handful of models to mean anything."""
    rows = [e for e in entries if e["scores"].get(focus) and e["profile"]]
    if len(rows) < 4:
        return {"enough": False, "n": len(rows), "needed": 4}
    scores = pd.Series([e["scores"][focus]["score"] for e in rows])
    out = []
    for key, label in PROFILE_ATTRS:
        vals = pd.Series([e["profile"].get(key) for e in rows], dtype="float64")
        if vals.notna().sum() < 4 or vals.nunique() < 2:
            continue
        rho = vals.corr(scores, method="spearman")
        if np.isfinite(rho):
            out.append({"attribute": key, "label": label, "rho": round(float(rho), 3)})
    out.sort(key=lambda r: -abs(r["rho"]))
    return {"enough": True, "n": len(rows), "correlations": out}


# ---------------------------------------------------------------- head-to-head
def _ic(v: float) -> str:
    """Signed IC to 3 places, without a "-0.000"."""
    return f"{0.0 if abs(v) < 0.0005 else v:+.3f}"


def _fmt_money(v: float) -> str:
    return f"${v / 1e12:.1f}T" if v >= 1e12 else f"${v / 1e9:.0f}B" if v >= 1e9 else f"${v / 1e6:.0f}M"


def compare(a: dict, b: dict, focus: str = "overall") -> dict:
    """Findings for why `a` scores differently from `b`, most important first."""
    findings, experiments = [], []
    sa, sb = a["scores"].get(focus), b["scores"].get(focus)
    label = "overall" if focus == "overall" else f"at {focus}"

    if not (sa and sb):
        missing = a if not sa else b
        findings.append({"kind": "info", "text": f"{missing['name']} has no out-of-sample score {label}, so there's nothing to compare. {missing['unrankable_reason'] or ''}".strip()})
        return {"findings": findings, "experiments": experiments}

    lead, trail = (a, b) if sa["score"] >= sb["score"] else (b, a)
    findings.append({
        "kind": "headline",
        "text": f"{lead['name']} ranks higher {label}: score {_ic(lead['scores'][focus]['score'])} vs "
                f"{_ic(trail['scores'][focus]['score'])} (the low end of each model's 95% IC range).",
    })

    # 1. Is the gap real? Per-horizon significance.
    real, noise = [], []
    for h in HORIZONS if focus == "overall" else [focus]:
        ha, hb = lead["horizons"].get(h), trail["horizons"].get(h)
        if not (ha and hb):
            continue
        diff, z = ic_difference(ha, hb)
        (real if abs(z) >= Z95 else noise).append((h, diff, z, ha, hb))
    for h, diff, z, ha, hb in real:
        better = lead if diff > 0 else trail
        findings.append({
            "kind": "significant",
            "text": f"{h}: {better['name']} has the clearly better IC ({_ic(max(ha['ic'], hb['ic']))} vs "
                    f"{_ic(min(ha['ic'], hb['ic']))}); a gap this size is unlikely to be luck (z = {abs(z):.1f}).",
        })
    for h, diff, z, ha, hb in noise:
        findings.append({
            "kind": "noise",
            "text": f"{h}: the IC gap ({_ic(ha['ic'])} vs {_ic(hb['ic'])}) is within noise (z = {abs(z):.1f}). "
                    f"The ranges are ±{(ha['ic_hi'] - ha['ic_lo']) / 2:.3f} and ±{(hb['ic_hi'] - hb['ic_lo']) / 2:.3f}.",
        })
    if noise and not real:
        experiments.append("Neither model is reliably better yet. Test both on the same new batches each earnings season; the live record will separate them.")

    # 2. Evidence size.
    na, nb = lead["scores"][focus]["n"], trail["scores"][focus]["n"]
    if min(na, nb) / max(na, nb) < 0.25:
        small = lead if na < nb else trail
        findings.append({
            "kind": "evidence",
            "text": f"{small['name']} was tested on far fewer predictions ({min(na, nb):,} vs {max(na, nb):,}), "
                    f"so its numbers are much less certain. The score already discounts that.",
        })

    # 3. Consistency across test years.
    if lead.get("consistency") is not None and trail.get("consistency") is not None:
        ca, cb = lead["consistency"], trail["consistency"]
        if abs(ca - cb) >= 0.15:
            steadier = lead if ca > cb else trail
            findings.append({
                "kind": "consistency",
                "text": f"{steadier['name']} is steadier: its IC was positive in {max(ca, cb):.0%} of test years "
                        f"across horizons, vs {min(ca, cb):.0%}.",
            })

    # 4. Which model family carries the difference.
    for h in (HORIZONS if focus == "overall" else [focus]):
        ha, hb = lead["horizons"].get(h), trail["horizons"].get(h)
        if not (ha and hb and ha["by_family"] and hb["by_family"]):
            continue
        gaps = {f: ha["by_family"][f] - hb["by_family"][f] for f in ("nn", "gbm")
                if ha["by_family"].get(f) is not None and hb["by_family"].get(f) is not None}
        if len(gaps) == 2 and abs(gaps["nn"] - gaps["gbm"]) >= 0.05:
            f = max(gaps, key=lambda k: abs(gaps[k]))
            other = "gbm" if f == "nn" else "nn"
            name = {"nn": "network ensemble", "gbm": "boosted trees"}
            findings.append({
                "kind": "family",
                "text": f"{h}: the {name[f]} accounts for more of the gap (IC {_ic(ha['by_family'][f])} vs "
                        f"{_ic(hb['by_family'][f])}, a {_ic(gaps[f])} difference) than the {name[other]} "
                        f"({_ic(gaps[other])}).",
            })

    # 5. Training data differences.
    pa, pb = lead.get("profile"), trail.get("profile")
    if pa and pb:
        if max(pa["tickers_used"], pb["tickers_used"]) >= 2 * min(pa["tickers_used"], pb["tickers_used"]):
            findings.append({
                "kind": "data",
                "text": f"Breadth: {lead['name']} learned from {pa['tickers_used']} tickers / {pa['rows']:,} reports; "
                        f"{trail['name']} from {pb['tickers_used']} / {pb['rows']:,}.",
            })
        sectors = set(pa["sector_mix"]) | set(pb["sector_mix"])
        diffs = sorted(((s, pa["sector_mix"].get(s, 0) - pb["sector_mix"].get(s, 0)) for s in sectors), key=lambda x: -abs(x[1]))
        big = [(s, d) for s, d in diffs if abs(d) >= 0.08][:3]
        if big:
            parts = [f"{s.replace('_', ' ')} {pa['sector_mix'].get(s, 0):.0%} vs {pb['sector_mix'].get(s, 0):.0%}" for s, _ in big]
            findings.append({"kind": "data", "text": f"Sector mix differs ({lead['name']} vs {trail['name']}): " + "; ".join(parts) + "."})
        if pa.get("median_market_cap") and pb.get("median_market_cap"):
            ratio = pa["median_market_cap"] / pb["median_market_cap"]
            if ratio >= 3 or ratio <= 1 / 3:
                findings.append({
                    "kind": "data",
                    "text": f"Company size: median market cap {_fmt_money(pa['median_market_cap'])} vs "
                            f"{_fmt_money(pb['median_market_cap'])}. Big, heavily covered companies tend to react "
                            f"to earnings more predictably than small ones.",
                })
        la, lb = pa.get("loss_making_share"), pb.get("loss_making_share")
        if la is not None and lb is not None and abs(la - lb) >= 0.08:
            findings.append({
                "kind": "data",
                "text": f"Loss-makers: {la:.0%} of {lead['name']}'s training quarters were unprofitable vs {lb:.0%} for "
                        f"{trail['name']}. Unprofitable companies make results harder to read.",
            })
        for p, e in ((pa, lead), (pb, trail)):
            if p.get("thin_tickers"):
                findings.append({
                    "kind": "data",
                    "text": f"{e['name']}: {', '.join(p['thin_tickers'][:6])} contributed fewer than 8 usable reports "
                            f"each, so they barely shaped the model.",
                })
        overlap = set(lead["symbols"]) & set(trail["symbols"])
        if overlap and min(len(lead["symbols"]), len(trail["symbols"])) and \
                len(overlap) / min(len(lead["symbols"]), len(trail["symbols"])) >= 0.8:
            findings.append({"kind": "data", "text": f"Their tickers overlap heavily ({len(overlap)} shared), so the data is similar; settings are the likelier cause."})

    ca_, cb_ = lead["config"], trail["config"]
    changed = [k for k in ("model_type", "epochs", "batch_size") if ca_.get(k) != cb_.get(k)]
    lead_settings = ", ".join(f"{k.replace('_', ' ')} {ca_.get(k)}" for k in changed)
    if changed:
        findings.append({
            "kind": "settings",
            "text": "Settings differ: " + "; ".join(f"{k.replace('_', ' ')} {ca_.get(k)} vs {cb_.get(k)}" for k in changed) + ".",
        })

    # 6. What each model leans on (shuffle importance).
    ia, ib = lead.get("importance"), trail.get("importance")
    if ia and ib:
        horizons = HORIZONS if focus == "overall" else [focus]
        best_h = max(
            (h for h in horizons if h in ia and h in ib and lead["horizons"].get(h) and trail["horizons"].get(h)),
            key=lambda h: lead["horizons"][h]["ic"] - trail["horizons"][h]["ic"], default=None,
        )
        if best_h:
            ga = {g: v["ic_drop"] for g, v in ia[best_h].items() if v.get("ic_drop") is not None}
            gb = {g: v["ic_drop"] for g, v in ib[best_h].items() if v.get("ic_drop") is not None}
            top_a = sorted(ga, key=lambda g: -ga[g])[:3]
            top_b = sorted(gb, key=lambda g: -gb[g])[:3]
            findings.append({
                "kind": "importance",
                "text": f"{best_h} (where {lead['name']}'s lead is largest): it leans most on "
                        + ", ".join(f"{GROUP_LABELS[g]} ({_ic(ga[g])})" for g in top_a)
                        + f"; {trail['name']} leans on " + ", ".join(f"{GROUP_LABELS[g]} ({_ic(gb[g])})" for g in top_b)
                        + ". Numbers are the IC lost when that input group is scrambled.",
            })
            shared = set(ga) & set(gb)
            gaps = sorted(((g, ga[g] - gb[g]) for g in shared), key=lambda x: -x[1])
            if gaps and gaps[0][1] >= 0.02:
                g, d = gaps[0]
                findings.append({
                    "kind": "importance",
                    "text": f"The biggest difference in reliance is {GROUP_LABELS[g]}: {lead['name']} gets "
                            f"{_ic(ga[g])} IC from it, {trail['name']} {_ic(gb[g])}. Its data or settings let it use that signal better.",
                })
    else:
        missing = [e["name"] for e in (lead, trail) if not e.get("importance")]
        findings.append({"kind": "info", "text": f"Run diagnostics for {', '.join(missing)} to see which inputs each model relies on."})

    # Experiments that would isolate a cause.
    if pa and pb:
        if changed:
            experiments.append(
                f"Retrain {trail['name']}'s tickers with {lead['name']}'s settings "
                f"({lead_settings}). If the score rises, the settings were the reason."
            )
        if pa["tickers_used"] != pb["tickers_used"] or set(lead["symbols"]) != set(trail["symbols"]):
            experiments.append(
                f"Retrain with {lead['name']}'s settings on {trail['name']}'s tickers. If it now scores like "
                f"{trail['name']}, the ticker set, not the model, explains the gap."
            )
    return {"findings": findings, "experiments": experiments, "leader": lead["id"]}


# --------------------------------------------------------------- live record
def track_record(db, price_client: PriceClient) -> dict:
    """Per model and horizon: how many batch/single predictions have closed
    windows, and how many called the direction right. Each (model, ticker,
    report) counts once however many times it was predicted. Only predictions
    that recorded their price basis (v2) are scored."""
    rows = db.fetch_prediction_outcomes()
    latest: dict[tuple, dict] = {}
    for r in rows:
        if not r.get("baseline_date"):
            continue
        key = (r["model_id"], r["symbol"], r["earnings_date"])
        if key not in latest or r["id"] > latest[key]["id"]:
            latest[key] = r
    if not latest:
        return {}

    start = (pd.Timestamp(min(r["baseline_date"] for r in latest.values())) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    end = (pd.Timestamp.now(tz=ET) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    closes: dict[str, pd.Series] = {}

    def close_on(sym: str, date: str) -> float:
        if sym not in closes:
            try:
                closes[sym] = price_client.get_prices(sym, start, end)["close"]
            except Exception:
                closes[sym] = pd.Series(dtype="float64")
        s = closes[sym]
        v = s.asof(pd.Timestamp(date)) if len(s) else np.nan
        return float(v) if v == v else np.nan

    bench = config.BENCHMARK_SYMBOL
    out: dict[str, dict] = defaultdict(lambda: {h: {"n": 0, "hits": 0} for h in HORIZONS + ["all"]})
    for r in latest.values():
        decisions = annotate_windows(r.get("decisions") or {}, r["earnings_date"])
        for h, d in decisions.items():
            if h not in config.HORIZONS or not d.get("closed"):
                continue
            actual = d.get("actual_pct_change")
            if actual is None:
                s0 = r["baseline_date"] if h == "1d" else r.get("day1_date")
                if not s0:
                    continue
                e0 = d["window_end"]
                stock = close_on(r["symbol"], e0) / close_on(r["symbol"], s0) - 1
                market = close_on(bench, e0) / close_on(bench, s0) - 1
                actual = (stock - market) * 100
            if not np.isfinite(actual):
                continue
            hit = int(np.sign(d["predicted_pct_change"]) == np.sign(actual))
            for k in (h, "all"):
                out[r["model_id"]][k]["n"] += 1
                out[r["model_id"]][k]["hits"] += hit
    return dict(out)
