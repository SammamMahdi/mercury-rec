-- Runs once, on first initialisation of an empty data directory.
-- Schema itself is managed by Alembic migrations, not here: init scripts do
-- not re-run on an existing volume, so using them for schema would silently
-- skip every change after the first start.

-- pg_stat_statements makes "which query is slow" answerable without adding
-- instrumentation to the application.
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
