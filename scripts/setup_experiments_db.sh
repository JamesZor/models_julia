#!/usr/bin/env bash
set -euo pipefail

# Provision (or migrate) the PostgreSQL 16 experiment database on mcmc-beast.
# Override MCMC_DB_HOST=local to run directly on the current machine.

CONTAINER="${MCMC_DB_CONTAINER:-mcmc_experiments_postgres}"
DATABASE="${MCMC_DB_NAME:-mcmc_experiments}"
DB_USER="${MCMC_DB_USER:-postgres}"
DB_PASSWORD="${MCMC_DB_PASSWORD:-football_mcmc_secure}"
DATA_DIR="${MCMC_DB_DATA_DIR:-/root/postgres_experiments_data}"
TARGET="${MCMC_DB_HOST:-root@mcmc-beast}"
PORT="${MCMC_DB_PORT:-5432}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCHEMA="${SCRIPT_DIR}/../src/training/inference/db/schema.sql"

[[ -f "$SCHEMA" ]] || { echo "Schema not found: $SCHEMA" >&2; exit 1; }

if [[ "$TARGET" == "local" ]] || [[ "$(uname -n | cut -d. -f1)" == "mcmc-beast" ]]; then
    remote() { "$@"; }
    apply_schema() {
        docker exec -i -e PGPASSWORD="$DB_PASSWORD" "$CONTAINER" \
            psql -v ON_ERROR_STOP=1 -U "$DB_USER" -d "$DATABASE" < "$SCHEMA"
    }
else
    remote() {
        local command_line
        printf -v command_line '%q ' "$@"
        ssh "$TARGET" "$command_line"
    }
    apply_schema() {
        ssh "$TARGET" docker exec -i -e PGPASSWORD="$DB_PASSWORD" "$CONTAINER" \
            psql -v ON_ERROR_STOP=1 -U "$DB_USER" -d "$DATABASE" < "$SCHEMA"
    }
fi

remote mkdir -p "$DATA_DIR"

if remote docker container inspect "$CONTAINER" >/dev/null 2>&1; then
    if [[ "$(remote docker inspect -f '{{.State.Running}}' "$CONTAINER")" != "true" ]]; then
        remote docker start "$CONTAINER" >/dev/null
    fi
else
    remote docker run -d \
        --name "$CONTAINER" \
        --restart unless-stopped \
        -e "POSTGRES_USER=$DB_USER" \
        -e "POSTGRES_PASSWORD=$DB_PASSWORD" \
        -v "$DATA_DIR:/var/lib/postgresql/data" \
        -p "$PORT:5432" \
        postgres:16-alpine >/dev/null
fi

printf 'Waiting for PostgreSQL'
for _ in $(seq 1 60); do
    if remote docker exec "$CONTAINER" pg_isready -U "$DB_USER" >/dev/null 2>&1; then
        echo ' ready.'
        break
    fi
    printf '.'
    sleep 1
done
remote docker exec "$CONTAINER" pg_isready -U "$DB_USER" >/dev/null

if ! remote docker exec -e PGPASSWORD="$DB_PASSWORD" "$CONTAINER" \
    psql -U "$DB_USER" -d postgres -tAc \
    "SELECT 1 FROM pg_database WHERE datname = '$DATABASE'" | grep -q 1; then
    remote docker exec -e PGPASSWORD="$DB_PASSWORD" "$CONTAINER" \
        createdb -U "$DB_USER" "$DATABASE"
fi

apply_schema

echo "Experiment database is ready: postgresql://${DB_USER}:***@mcmc-beast:${PORT}/${DATABASE}"
