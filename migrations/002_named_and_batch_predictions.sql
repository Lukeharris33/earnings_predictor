-- Named predictions and prediction batches.
-- Run once in the Supabase SQL editor (Project -> SQL Editor -> New query)
-- on a database created from an older schema.sql. Safe to re-run.

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

alter table predictions add column if not exists name text;
alter table predictions add column if not exists batch_id uuid references prediction_batches(id) on delete cascade;
alter table predictions add column if not exists details jsonb;   -- filing, reported figures, model inputs
create index if not exists predictions_batch_id_idx on predictions(batch_id);
create index if not exists predictions_name_idx on predictions(name);

-- Make the Supabase API pick up the new table and columns immediately.
notify pgrst, 'reload schema';
