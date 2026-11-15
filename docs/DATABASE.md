# DGB AI database

Source of truth for accounts (role, active flag) and the audit log. SQLAlchemy 2 + Alembic migrations; the schema is
created and upgraded automatically the first time the API or `tools/manage_users.py` connects.

| Environment | Database | How it is chosen |
| --- | --- | --- |
| Local development (Docker) | PostgreSQL 17 container `dgb-postgres`, host port 5434 | `DGB_DATABASE_URL=postgresql+psycopg://dgb:<password>@127.0.0.1:5434/dgb` |
| Production | PostgreSQL | same variable, managed or self-hosted server |
| Fallback / quick tests | SQLite file `instance/dgb.sqlite3` | used only when `DGB_DATABASE_URL` is not set |

`postgresql://...` is accepted and rewritten to `postgresql+psycopg://...` (driver: `psycopg[binary]`, in `requirements/api.txt`).

## Start the local PostgreSQL
1. Put `DGB_POSTGRES_PASSWORD=<letters and digits>` in `.env` (git-ignored) in the project root.
2. `docker compose -f docker-compose.db.yml up -d` (creates databases `dgb` and `dgb_test`, data kept in volume `dgb_pgdata`).
3. `$env:DGB_DATABASE_URL = "postgresql+psycopg://dgb:<password>@127.0.0.1:5434/dgb"` before starting the API or running `tools\manage_users.py`.

The server only listens on 127.0.0.1. `docker compose -f docker-compose.db.yml down` stops it and keeps the data;
add `-v` only if you want to delete the data.

## Tests
`python -m pytest tests\unit -q` uses a throw-away SQLite file per test. To run the same tests on PostgreSQL:
`$env:DGB_TEST_DATABASE_URL = "postgresql+psycopg://dgb:<password>@127.0.0.1:5434/dgb_test"`. That database is wiped
before each test and the suite refuses to run unless its name ends in `_test`.

## Migrations
`database/migrations/versions/`. Add a new revision for every schema change; never edit an applied one. The drift
test fails if the models and the migrations disagree.
