#!/bin/bash
# Database migration script
# Run this after updating schema changes

set -e

POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_DB="${POSTGRES_DB:-openclaw}"
POSTGRES_USER="${POSTGRES_USER:-openclaw}"

export PGPASSWORD="${POSTGRES_PASSWORD:-openclaw123}"

echo "Running database migrations..."

psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" <<'SQL'
-- Add migration SQL statements here
-- Example:
-- ALTER TABLE chunks ADD COLUMN IF NOT EXISTS version INTEGER DEFAULT 1;
SQL

echo "Migrations completed successfully."
unset PGPASSWORD
