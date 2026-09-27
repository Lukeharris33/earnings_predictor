-- Run this once in the Supabase SQL editor (Project -> SQL Editor -> New query).
-- This app is a trusted, single-user backend that authenticates with the
-- service_role key, so RLS is left disabled on these tables. If you deploy
-- this somewhere multi-tenant, add RLS policies before exposing it further.

create extension if not exists "pgcrypto";

create table if not exists training_runs (
  id uuid primary key default gen_random_uuid(),
  status text not null default 'pending',       -- pending | running | completed | failed
  symbols text[] not null,
  config jsonb,
  metrics jsonb,
  error_count integer default 0,
  started_at timestamptz default now(),
  finished_at timestamptz
);

create table if not exists training_data (
  id bigint generated always as identity primary key,
  run_id uuid references training_runs(id) on delete cascade,
  symbol text not null,
  earnings_date date not null,
  features jsonb not null,   -- XBRL-derived fundamentals + pre-announcement price features
  targets jsonb not null,    -- {"1d": pct, "1w": pct, "1m": pct, "1y": pct}
  created_at timestamptz default now()
);
create index if not exists training_data_run_id_idx on training_data(run_id);
create index if not exists training_data_symbol_idx on training_data(symbol);

create table if not exists errors (
  id bigint generated always as identity primary key,
  run_id uuid references training_runs(id) on delete set null,
  symbol text,
  stage text,               -- fetch_filings | fetch_prices | assemble | train | flask_route:*
  level text not null default 'error',  -- info | warning | error
  message text,
  traceback text,
  created_at timestamptz default now()
);
create index if not exists errors_run_id_idx on errors(run_id);

create table if not exists models (
  id uuid primary key default gen_random_uuid(),
  run_id uuid references training_runs(id) on delete set null,
  name text not null,
  storage_path text not null,          -- path inside the `models` storage bucket
  feature_columns jsonb not null,
  target_horizons text[] not null,
  target_scale jsonb,                  -- per-horizon mean/std used to un-scale predictions
  metrics jsonb,                       -- per-horizon walk-forward metrics vs. baselines
  symbols text[],
  created_at timestamptz default now()
);

create table if not exists prediction_batches (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  model_id uuid references models(id) on delete cascade,
  symbols text[] not null,
  status text not null default 'running',   -- running | completed | failed
  summary jsonb,                            -- per-horizon BUY/SELL/HOLD counts, failures
  created_at timestamptz default now(),
  finished_at timestamptz
);
create index if not exists prediction_batches_name_idx on prediction_batches(name);

create table if not exists predictions (
  id bigint generated always as identity primary key,
  model_id uuid references models(id) on delete cascade,
  name text,
  batch_id uuid references prediction_batches(id) on delete cascade,
  symbol text not null,
  earnings_date date,
  predicted jsonb not null,
  decisions jsonb not null,
  details jsonb,                            -- filing, reported figures, model inputs
  created_at timestamptz default now()
);
create index if not exists predictions_batch_id_idx on predictions(batch_id);
create index if not exists predictions_name_idx on predictions(name);

-- Upgrading a database created before named/batch predictions? Run
-- migrations/002_named_and_batch_predictions.sql instead of this file.

-- After running this file, also create a Storage bucket named "models"
-- (Project -> Storage -> New bucket -> name it "models", private is fine)
-- so save_model() has somewhere to upload the zipped model artifacts.
