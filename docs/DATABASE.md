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

## Production requirements

- Set `DGB_DATABASE_URL` through the deployment platform's secret manager. Do not copy the development `.env` or
  commit database credentials. Use a dedicated database role with only the permissions DGB needs.
- Use a supported PostgreSQL service on a private network, restrict inbound database access to the API service,
  and require TLS for connections when supported by the provider.
- Run Alembic schema upgrades in a controlled release step and verify the resulting revision before serving traffic.
  This version also upgrades the schema during API startup; serialize migrations during deployment so multiple
  workers or replicas do not attempt concurrent upgrades.
- Configure automated backups and point-in-time recovery where available, restrict backup access, and test a restore
  before relying on the database for production accounts and audit records.
- Supply `DGB_SECRET_KEY` and the initial `DGB_BOOTSTRAP_ADMIN_USERNAME` / `DGB_BOOTSTRAP_ADMIN_PASSWORD` through
  secret management. Bootstrap only creates the account if it is absent; changing those variables does not reset its
  password. Use the account-management tool to rotate credentials after bootstrap.
- Verify connectivity, migrations, login, role checks, audit writes, backup/restore procedures, and the deployed
  commit against the production environment before calling the release ready. Never point the destructive
  `DGB_TEST_DATABASE_URL` test suite at a production database.

## Tests
`python -m pytest tests\unit -q` uses a throw-away SQLite file per test. To run the same tests on PostgreSQL:
`$env:DGB_TEST_DATABASE_URL = "postgresql+psycopg://dgb:<password>@127.0.0.1:5434/dgb_test"`. That database is wiped
before each test and the suite refuses to run unless its name ends in `_test`.

## Migrations
`database/migrations/versions/`. Add a new revision for every schema change; never edit an applied one. The drift
test fails if the models and the migrations disagree.
