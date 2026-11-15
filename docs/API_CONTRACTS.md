# DGB Backend API Contracts and Implementation Order

Owner: Claude (backend). Consumer: Copilot (`ui/portal`). Approver: project owner.
Status: **PROPOSED** unless marked IMPLEMENTED. Last updated 2026-11-13. Nothing here is integrated until Copilot confirms impact and
timing and the owner approves the merge (see `DEVELOPMENT_RESPONSIBILITY_MATRIX.md`).

Conventions: errors keep the existing FastAPI shape `{"detail": "<text>"}`. Where a machine-readable reason is
needed, the response also carries the header `X-DGB-Error-Code`. Token pairs always have the existing shape
`{access_token, refresh_token, token_type, expires_in}`.

---------------------------------------------------------------------------------------------------------
## 0. Role claim (IMPLEMENTED, merged `208f4bf`)

`role` in tokens and in `GET /auth/me` is exactly `"admin" | "user" | "readonly"`. It used to be the text
`"UserRole.ADMIN"`, which made every admin fail `require_role`. Old tokens are read as the role they name and
repaired on refresh. Unknown roles -> 401. The frontend compares with `"admin"` and normalizes nothing.

---------------------------------------------------------------------------------------------------------
## 0b. Accounts and sessions (IMPLEMENTED in `backend/db-foundation`, pending merge)

No route, payload or error shape changes. Behaviour that the frontend can rely on:
- Accounts live in the database. `role` and `is_active` are read from the account on **every** protected request and on
  refresh; the token proves identity only. A demoted user gets `403` on admin routes and a disabled or deleted user
  gets `401` immediately, not when the token expires.
- Usernames are case-insensitive (`Boss` logs in as `boss`); `/auth/me` returns the normalized name in `sub`.
- `POST /auth/refresh` returns `401` (`Account is not available`) if the account is disabled or gone, and issues the
  role currently stored for the account.
- `POST /auth/register` stays disabled. Accounts are created with `tools/manage_users.py` today; admin HTTP endpoints for
  users follow in `backend/admin-users`.
- Failed and successful logins and refresh refusals are written to an audit log (hashed client key, never passwords).

---------------------------------------------------------------------------------------------------------
## 1. Anonymous (guest) chat  - PROPOSED, ships disabled

Goal: a visitor can try the chat without an account, without creating a way to run up our compute bill.

**Route.** Same route as signed-in chat: `POST /v1/chat/completions`. Request and SSE response shapes are
unchanged. No `Authorization` header = guest request. No guest token or cookie is ever issued.

**Switch.** Off unless `DGB_GUEST_CHAT=on` (environment). While off, behaviour is today's: `401`
(`X-DGB-Error-Code: auth_required`). `GET /v1/guest/status` (public) returns
`{"enabled": bool, "limits": {...}}` so the UI can show or hide guest chat from the server's answer, not a guess.

**Limits for guests** (all server-side, configurable, defaults shown):

| Limit | Default | If exceeded |
|---|---|---|
| Requests per client per minute / per hour / per day | 5 / 20 / 60 | `429` + `Retry-After`, `X-DGB-Error-Code: guest_rate_limited` |
| Request body size | 16 KB | `413` |
| Messages per request / characters per message | 20 / 2,000 | `422` |
| `max_new_tokens` | capped at 256 (silently lowered, reported in a response header `X-DGB-Max-New-Tokens`) | - |
| Concurrent streams per client / globally | 1 / 8 | `429` / `503` (`guest_unavailable`) |
| Models | the single public default model only; any `model` field is ignored | - |
| Admin kill switch | `DGB_GUEST_CHAT=off` stops guests immediately | `401` |

"Client" = a keyed hash of the network address (salted, rotated daily). Raw addresses are never stored.
Behind a proxy the API only trusts `X-Forwarded-For` from `DGB_TRUSTED_PROXIES`.
Optional later hardening: CAPTCHA (e.g. Cloudflare Turnstile) required via header `X-Turnstile-Token` when
`DGB_GUEST_CAPTCHA=on`; the UI gets that from `/v1/guest/status`.

**Retention.** The server stores no guest message content and no guest conversation. Guest history lives only in
the visitor's browser (as the UI already does). Server logs for guests hold only: timestamp, hashed client key,
token counts, status code, latency. Message text is never logged. Metadata kept 30 days.

**Guests cannot:** select models, use any `/v1/admin/*`, see other conversations, or call any route other than
chat and `/v1/guest/status`.

**Not before:** guest chat stays OFF until there is a model good enough to show the public (an evaluation
threshold set by the owner, see `tools/eval_model.py`). A 2-8M-parameter test model produces nonsense.

---------------------------------------------------------------------------------------------------------
## 2. Google and GitHub sign-in  - PROPOSED

Backend-driven authorization-code flow with PKCE and a one-time exchange code. Provider secrets live only on
the server (`DGB_OAUTH_GOOGLE_CLIENT_ID/SECRET`, `DGB_OAUTH_GITHUB_CLIENT_ID/SECRET`, `DGB_PUBLIC_BASE_URL`).
The browser never sees a provider token or secret, and no JWT is ever placed in a URL.

1. `GET /auth/providers` (public) -> `{"providers":[{"id":"google","enabled":true},{"id":"github","enabled":false}]}`
   The UI enables a button only when its provider is `enabled`.
2. `GET /auth/oauth/{provider}/start?return_to=/ui/` -> `302` to the provider. The server creates a random `state`,
   a PKCE verifier and a nonce, stores them server-side for 10 minutes (single use), and validates `return_to`
   against an allow-list of same-site paths (no open redirect).
3. Provider redirects to `GET /auth/oauth/{provider}/callback?code=...&state=...`. The server checks `state`,
   exchanges the code (with the PKCE verifier), and requires a **verified** email (Google: `email_verified`;
   GitHub: primary verified address from `/user/emails`).
4. Success -> `302` to `{return_to}#oauth_code=<one-time code>` (valid 60 s, single use).
   Failure -> `302` to `/ui/#oauth_error=<access_denied|state_invalid|email_unverified|provider_unavailable|account_disabled>`.
5. `POST /auth/oauth/exchange` body `{"code": "<one-time code>"}` -> normal token pair (same shape as `/auth/login`).
   Invalid, expired or reused code -> `401`.

Account rules: users are matched by provider identity first, then by verified email. A new account is always role
`user`. OAuth can never create or promote an `admin` (admins come from bootstrap or admin tooling only). Disabled
accounts are refused. Every sign-in and failure is written to the audit log.

---------------------------------------------------------------------------------------------------------
## 3. Decisions the backend work depends on

- D1 **Persistent users and audit log first.** Today users exist only in memory (bootstrap admin). OAuth, user
  management, API keys and audit logs all need a database. Choice: SQLAlchemy 2 + Alembic migrations, PostgreSQL
  (Docker container for local development, same in production; URL from `DGB_DATABASE_URL`), SQLite only as a no-setup fallback. See `docs/DATABASE.md`.
- D2 **Passwords** stay bcrypt via the existing hashing; no new auth mechanism for the frontend.
- D3 **Everything protected is enforced on the server**; the UI only reflects what `/auth/me` returns.

## 4. Implementation order (backend)

| # | Work | Branch | Needs from Copilot |
|---|---|---|---|
| 1 | Role claim fix | `backend/fix-admin-role-claim` (merged) | verified |
| 2 | Byte-level tokenizer | `task/T1.1-byte-tokenizer` (merged) | none |
| 3 | Database foundation: users, audit log, migrations; bootstrap admin seeds the DB | `backend/db-foundation` (in review) | none |
| 4 | Admin: user list, role change, disable/enable, invitations; audit-log read API | `backend/admin-users` | wire screens |
| 5 | API keys and usage reporting | `backend/api-keys` | wire screens |
| 6 | Guest chat (section 1), disabled by default | `backend/guest-chat` | read `/v1/guest/status`, show limits |
| 7 | Google/GitHub OAuth (section 2) | `backend/oauth` | enable buttons from `/auth/providers`, handle `#oauth_code` |
| 8 | Model registry, real deploy/rollback, active-model API; training-launch honesty | `backend/model-registry` | wire model picker |
| 9 | Dataset inventory and per-source status | `backend/dataset-status` | wire screen |

Parallel model track (independent of the web work): T1.2 token shards and packed loader, then cloud GPU
pretraining (roadmap Phase 2). The website is only as good as the model behind it.
