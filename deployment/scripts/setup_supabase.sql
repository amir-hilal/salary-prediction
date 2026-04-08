-- =============================================================================
-- Salary Prediction — Supabase Schema
-- Run this once against your Supabase project via the SQL editor or psql.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- predictions
-- Stores every call to POST /api/v1/predict.
-- ---------------------------------------------------------------------------
create table if not exists public.predictions (
    id               uuid primary key default gen_random_uuid(),
    created_at       timestamptz not null default now(),
    features         jsonb not null,          -- encoded feature dict from request
    predicted_salary numeric(12, 2) not null, -- point estimate in USD
    salary_range_low numeric(12, 2),          -- Q25 of leaf-node peer group
    salary_range_high numeric(12, 2),         -- Q75 of leaf-node peer group
    model_version    text not null,           -- artifact timestamp from registry
    currency         text not null default 'USD'
);

-- Index for dashboard time-series queries
create index if not exists predictions_created_at_idx
    on public.predictions (created_at desc);

-- ---------------------------------------------------------------------------
-- narratives
-- Stores the LLM-generated narrative for each prediction.
-- ---------------------------------------------------------------------------
create table if not exists public.narratives (
    id             uuid primary key default gen_random_uuid(),
    prediction_id  uuid not null references public.predictions (id) on delete cascade,
    created_at     timestamptz not null default now(),
    summary        text not null,
    uncertainty    text not null,   -- mandatory error/range disclosure
    insights       jsonb not null,  -- list[str] of bullet-point insights
    recommendation text not null,
    chart_spec     jsonb not null,  -- ChartSpec fields: type, title, x_label, y_label, data_key
    raw_response   text            -- full LLM output kept for debugging
);

-- Index for joining narratives to their predictions
create index if not exists narratives_prediction_id_idx
    on public.narratives (prediction_id);

-- ---------------------------------------------------------------------------
-- Row Level Security
-- Anon key: read only.
-- Service-role key (used by the API): full access.
-- ---------------------------------------------------------------------------
alter table public.predictions enable row level security;
alter table public.narratives  enable row level security;

-- Allow anyone with the anon key to read predictions (dashboard)
create policy "predictions: anon read"
    on public.predictions for select
    using (true);

-- Allow the service role to insert predictions (API writes)
create policy "predictions: service insert"
    on public.predictions for insert
    with check (true);

-- Allow anyone with the anon key to read narratives (dashboard)
create policy "narratives: anon read"
    on public.narratives for select
    using (true);

-- Allow the service role to insert narratives (API writes)
create policy "narratives: service insert"
    on public.narratives for insert
    with check (true);

-- ---------------------------------------------------------------------------
-- Realtime
-- Enable realtime publication so the Streamlit dashboard receives live inserts.
-- ---------------------------------------------------------------------------
alter publication supabase_realtime add table public.predictions;
alter publication supabase_realtime add table public.narratives;
