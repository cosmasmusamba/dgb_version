#!/usr/bin/env python3
r"""
tools/manage_users.py (db-foundation) - manage DGB accounts from the command line.

    python tools\manage_users.py list
    python tools\manage_users.py create alice --role user            # asks for the password
    python tools\manage_users.py create bob --role admin --email bob@example.com
    python tools\manage_users.py set-role alice admin
    python tools\manage_users.py disable alice
    python tools\manage_users.py enable alice
    python tools\manage_users.py set-password alice
    python tools\manage_users.py audit --limit 20

Uses the database from DGB_DATABASE_URL (default instance\dgb.sqlite3). For scripts, the password can
come from an environment variable: --password-env NAME. Passwords are never printed or logged.
The last active administrator cannot be demoted or disabled.
"""
import argparse
import getpass
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def read_password(args) -> str:
    if getattr(args, "password_env", None):
        pw = os.environ.get(args.password_env, "")
        if not pw:
            raise SystemExit(f"ERROR: environment variable {args.password_env} is empty")
        return pw
    first = getpass.getpass("Password: ")
    if first != getpass.getpass("Repeat password: "):
        raise SystemExit("ERROR: passwords do not match")
    return first


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Manage DGB accounts")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    c = sub.add_parser("create"); c.add_argument("username"); c.add_argument("--role", default="user")
    c.add_argument("--email"); c.add_argument("--password-env")
    r = sub.add_parser("set-role"); r.add_argument("username"); r.add_argument("role")
    sub.add_parser("disable").add_argument("username")
    sub.add_parser("enable").add_argument("username")
    p = sub.add_parser("set-password"); p.add_argument("username"); p.add_argument("--password-env")
    a = sub.add_parser("audit"); a.add_argument("--limit", type=int, default=30)
    args = ap.parse_args(argv)

    from api_manager.auth.api_security import hash_password
    from database import get_session, database_url
    from database import audit, users as repo

    try:
        with get_session() as s:
            if args.cmd == "list":
                rows = repo.list_users(s)
                print(f"database: {database_url().split('@')[-1]}")
                print(f"{'username':<24}{'role':<10}{'active':<8}{'email':<28}last login")
                for u in rows:
                    print(f"{u['username']:<24}{u['role']:<10}{str(u['is_active']):<8}{(u['email'] or '-'):<28}{u['last_login_at'] or '-'}")
                print(f"{len(rows)} account(s)")
            elif args.cmd == "create":
                pw = read_password(args)
                repo.validate_password(pw)
                u = repo.create_user(s, args.username, hash_password(pw), args.role, args.email)
                audit.record(s, "user_created", actor="cli", target=u.username, detail={"role": u.role})
                print(f"created {u.username} ({u.role})")
            elif args.cmd == "set-role":
                u = repo.set_role(s, args.username, args.role)
                audit.record(s, "role_changed", actor="cli", target=u.username, detail={"role": u.role})
                print(f"{u.username} is now {u.role}")
            elif args.cmd in ("disable", "enable"):
                u = repo.set_active(s, args.username, args.cmd == "enable")
                audit.record(s, f"user_{args.cmd}d", actor="cli", target=u.username)
                print(f"{u.username} {'enabled' if u.is_active else 'disabled'}")
            elif args.cmd == "set-password":
                pw = read_password(args)
                repo.validate_password(pw)
                u = repo.set_password_hash(s, args.username, hash_password(pw))
                audit.record(s, "password_changed", actor="cli", target=u.username)
                print(f"password updated for {u.username}")
            elif args.cmd == "audit":
                for e in audit.recent(s, args.limit):
                    print(f"{e['ts']}  {'ok ' if e['ok'] else 'ERR'}  {e['action']:<18} actor={e['actor'] or '-':<12} target={e['target'] or '-'}")
    except repo.AccountError as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
