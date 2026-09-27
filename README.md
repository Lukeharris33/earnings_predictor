# Earnings Signal

A Flask app that trains models to predict how a stock will do *relative to
the market* after an earnings report — at 1 day, 1 week, 1 month, and 1
year out — and turns that into weighted buy/sell/hold calls. Training data,
every error/warning, every trained model, and every (named or batched)
prediction are saved to Supabase.

## How it works

1. **Data.** For each ticker you give it, the app pulls from [SEC EDGAR](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)
   (free, no API key):
   - `submissions` — every filing, plus the company's SIC industry code.
     Each 10-Q/10-K is one quarterly report; its announcement time is the
     8-K *Item 2.02* earnings press release filed between the quarter end
     and the 10-Q (falling back to the 10-Q's own filing time).
   - `companyfacts` — every XBRL-tagged number the company has filed. Each
     value comes from the *original* filing for that quarter, so later
     restatements can't leak in.

   Daily prices and split history come from Yahoo Finance via `yfinance`.

2. **Features.** Per quarterly report:
   - *Fundamentals*: margins, YoY/QoQ growth, balance-sheet and cash-flow ratios.
   - *Valuation*: market cap (price x reported shares, corrected for later
     splits), sales/earnings/free-cash-flow yield, book-to-market.
   - *Sector*: one of ~16 groups from the SIC code.
   - *Vs. own trend* (all horizons): revenue growth acceleration, and growth
     and margins vs. the previous four quarters. EDGAR has no analyst
     estimates, so this stands in for "surprise".
   - *Trend quality* (1m/1y only): 8-quarter growth consistency, margin
     slope, 2-year revenue CAGR, accruals.
   - *Price*: returns vs. the market over the 1/3/12 months before the
     release, volatility, and — for the 1w/1m/1y models — the first-day
     reaction to the release (post-earnings drift).

3. **Targets.** Return in excess of `SPY` over the same days. The 1d target
   is the release reaction (last close before the announcement to the next
   close). The 1w/1m/1y targets start at the first post-release close,
   since those models see the first-day reaction as an input. Each horizon
   only needs its own window, so recent reports still train the short
   horizons.

4. **Preprocessing** (per horizon, fitted on training rows only): features
   are capped at their 1st/99th percentile, missing values are
   median-filled with a 0/1 "was missing" flag, selected features also get
   a percentile against all companies reporting the same quarter, and
   everything is standard-scaled. Targets are winsorized at 1%/99%.

5. **Models.** One per horizon: a small L2-regularized network trained with
   5 random seeds and averaged, gradient-boosted trees, or a blend of both
   (the default).

6. **Evaluation.** Walk-forward: each of the last 4 calendar years is
   predicted by a model trained only on earlier reports whose labels had
   fully resolved before that year (purged). Reported per horizon, next to
   naive baselines: directional accuracy vs. "always guess the more common
   direction", MAE vs. guessing zero, rank correlation (IC), and how much
   the top-fifth picks beat the bottom-fifth. The 1d model is also scored
   with and without the vs-trend features, to check they actually help.
   The saved model is then retrained on all rows.

7. **Decisions.** A predicted move more than 0.25 standard deviations (of
   that horizon's typical move) above zero is BUY, below is SELL, in
   between is HOLD. Each call gets a **confidence weight (0-1)** from the
   predicted move's size relative to the typical move.

8. **Breakdown.** Every prediction shows the filing it used (dates, link,
   sector, market cap), the raw XBRL figures vs. a year earlier, and every
   model input with its training median, peer percentile, z-score, and
   whether it was missing or capped. This is saved with the prediction.

## Setup

### 1. Set an SEC User-Agent
EDGAR needs no key, but the SEC requires every request to carry a
User-Agent with your name and a contact email (e.g.
`Jane Doe jane@example.com`), and allows at most 10 requests/second. The
client paces itself and caches responses under `data/cache/` for a day.

### 2. Create a Supabase project
- Create a project at [supabase.com](https://supabase.com).
- Open **SQL Editor**, paste in `schema.sql`, and run it. (Upgrading an
  existing database? Run `migrations/002_named_and_batch_predictions.sql`
  instead — it adds named and batch predictions.)
- Open **Storage**, create a new bucket named `models` (private is fine).
- Open **Project Settings → API** and copy the **Project URL** and the
  **service_role** key (not the anon key — this app is a trusted backend).

### 3. Configure the app
```bash
cp .env.example .env
# then fill in SEC_USER_AGENT, SUPABASE_URL, SUPABASE_SERVICE_KEY
```

### 4. Install and run
```bash
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
python app.py
```
Open `http://localhost:5000`.

## Using it

- **Train a model**: name it, enter tickers (or click *Load the starter
  universe* for ~400 companies across every sector), pick blend / trees /
  network, and hit "Start training". The first run on the starter universe
  downloads a lot from EDGAR (about 20 minutes) and then trains for a while;
  later runs reuse the cache. When it finishes you get the walk-forward
  table per horizon.
- **Predict one ticker**: pick a model and a ticker, optionally name the
  prediction, and hit "Get decision". It scores the ticker's most recent
  quarterly report and shows the calls plus the full breakdown.
- **Batch predict**: give the batch a name, pick a model, list tickers (or
  *Load the holdout tickers* — 20 companies the starter universe leaves
  out, so the test is out-of-sample) and run it. Every prediction is saved
  under that name; the batch row stores BUY/SELL/HOLD counts per horizon.
  *Recent batches* lets you reopen any earlier batch.
- **Model ranking** tab: models ranked by the low end of their out-of-sample
  IC's 95% range, so a lucky result from a small test set can't top the
  table. It shows what the better models have in common, and a head-to-head
  that explains why one model beats another: whether the gap is real, how
  their training data and settings differ, and which input groups each relies
  on (IC lost when that group is shuffled in the test years). New models
  record this at training time; older v2 models can backfill it with *Run
  diagnostics*, which replays their stored training rows through walk-forward.
- **Saved models / Recent runs / Recent predictions**: what's been trained,
  how each model did out of sample, and what's been predicted.
- Models from before these changes (v1) still load, and predict raw rather
  than market-relative moves.

## Notes and limitations

- **This is not financial advice** and the model has no idea about macro
  conditions, sector moves, guidance language, or anything outside the raw
  numbers in the statements. Treat it as a research toy.
- There are no analyst estimates in EDGAR, so EPS/revenue *surprise* is not
  a feature. Growth and margins vs. the company's own trend stand in for it.
- The starter universe is today's listed companies, so it has survivorship
  bias: firms that went bust or were acquired are missing, which flatters
  backtests somewhat.
- XBRL numbers come from the 10-Q, which is usually filed a few days after
  the press release. The headline numbers match, so they're treated as
  known at announcement time.
- Only filings from ~2009 on carry XBRL data, so older quarters are skipped.
- `yfinance` scrapes Yahoo Finance and is unofficial; if it breaks, only
  `src/price_client.py` needs changing.
- The dev server (`python app.py`) runs training in a background thread
  in-process. That's fine for local/single-user use; for anything else, put
  a real task queue (Celery, RQ) in front of `src/trainer.py`'s `_run()`.
- `predictor.py` caches loaded models in memory per process, so the first
  prediction against a given model is slower (downloads + unzips from
  Supabase Storage) and subsequent ones are fast.

## Project layout

```
app.py                  Flask routes
src/config.py            Env-driven settings
src/universe.py          Starter training universe + holdout tickers
src/sec_client.py         SEC EDGAR wrapper (User-Agent, pacing, retries)
src/price_client.py       Adjusted daily closes via yfinance
src/disk_cache.py         JSON disk cache for both clients
src/data_pipeline.py      Earnings events, XBRL/valuation/trend features, excess-return labels
src/preprocessing.py      Capping, imputation + missing flags, peer percentiles, scaling
src/model.py               Networks, boosted trees, evaluation, decision logic
src/trainer.py              Background-thread training orchestration
src/predictor.py            Load a saved model and score a ticker
src/batch.py                Background batch predictions
src/supabase_client.py      All Supabase reads/writes
schema.sql                Supabase table definitions
migrations/               Upgrades for existing Supabase databases
templates/, static/        Flask frontend
```
