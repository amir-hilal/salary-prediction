# Supabase Setup Guide

How to create a Supabase project, locate your credentials, run the schema,
and verify the setup.

---

## 1. Create a Supabase Project

1. Go to [supabase.com](https://supabase.com) and sign in (or create a free account).
2. Click **New project**.
3. Choose an organisation, give the project a name (e.g. `salary-prediction`), set a strong database password, and pick the region closest to you.
4. Click **Create new project** and wait ~60 seconds for provisioning.

---

## 2. Find Your `SUPABASE_URL`

1. In your project dashboard, click **Project Settings** (gear icon in the left sidebar).
2. Click **API** under the Settings section.
3. Under **Project URL**, copy the value — it looks like:
   ```
   https://abcdefghijklm.supabase.co
   ```
4. Paste it into your `.env` file:
   ```
   SUPABASE_URL=https://abcdefghijklm.supabase.co
   ```

---

## 3. Find Your Keys

On the same **Settings → API** page, scroll down to **Project API keys**:

| Key name | Where to find it | Use |
|---|---|---|
| `anon` / `public` | Visible by default | Read-only dashboard access (respects RLS) |
| `service_role` | Click **Reveal** | Server-side writes from the API (bypasses RLS) |

Add both to your `.env`:
```
SUPABASE_ANON_KEY=eyJ...         # the anon key
SUPABASE_SERVICE_ROLE_KEY=eyJ... # the service_role key — keep this secret
```

> **Security note:** Never commit the service-role key to version control. It is listed in `.gitignore` via `.env`.

---

## 4. Run the Schema in the SQL Editor

1. In the left sidebar, click **SQL Editor** (the `</>` icon).
2. Click **New query** (top-right of the editor).
3. Open `deployment/scripts/setup_supabase.sql` in your editor, copy the entire contents.
4. Paste into the Supabase SQL editor and click **Run** (or press `Ctrl+Enter`).
5. You should see a success message. The following objects will be created:
   - Table `public.predictions`
   - Table `public.narratives`
   - Indexes on `created_at` and `prediction_id`
   - RLS policies (anon read / service insert / service update / service delete)
   - Realtime publication for both tables

---

## 5. Verify the Setup

Still in the SQL editor, run:

```sql
-- Check tables exist
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
ORDER BY table_name;

-- Check RLS is enabled
SELECT tablename, rowsecurity
FROM pg_tables
WHERE schemaname = 'public';

-- Check indexes
SELECT indexname, tablename
FROM pg_indexes
WHERE schemaname = 'public'
ORDER BY tablename, indexname;
```

Expected output — tables: `narratives`, `predictions`. RLS: `rowsecurity = true` for both. Indexes: at least `predictions_created_at_idx`, `narratives_prediction_id_idx`, and the primary key indexes.

---

## 6. Enable Realtime (if not automatic)

The SQL script runs `ALTER PUBLICATION supabase_realtime ADD TABLE ...` which should
enable realtime for both tables. First, verify the publication actually has the tables:

```sql
SELECT pubname, tablename
FROM pg_publication_tables
WHERE pubname = 'supabase_realtime';
```

Expected output: two rows — one for `predictions`, one for `narratives`.

If the query returns nothing, run the statements manually:

```sql
alter publication supabase_realtime add table public.predictions;
alter publication supabase_realtime add table public.narratives;
```

To verify via the UI: go to **Database → Publications** in the left sidebar, click
`supabase_realtime`, and confirm `predictions` and `narratives` are listed.

---

## 7. Local Development Tips

- Copy `.env.example` to `.env` and fill in the three Supabase variables before starting.
- The API uses the `service_role` key (writes bypass RLS — intentional for server-side).
- The Streamlit dashboard uses the `anon` key (read-only, RLS enforced).
- If you need to reset the schema during development, run in the SQL editor:
  ```sql
  drop table if exists public.narratives cascade;
  drop table if exists public.predictions cascade;
  ```
  Then re-run `setup_supabase.sql`.
