# DGB Web Portal

The portal provides a guest-first user chat workspace and a separate, role-gated administration workspace. It uses the existing API authentication and endpoints; it does not implement a second auth service or weaken server authorization.

## Development and build

```powershell
cd ui/portal
npm install
npm run dev
npm run build
```

The Vite development server proxies API requests to `http://127.0.0.1:8000`. The production build is served by the API at `/ui/`.

The development login screen includes user and administrator previews. These use synthetic data, do not authenticate, and never make API requests. They are excluded from the production build. Use them to exercise UI navigation only, not to validate API authorization.

## Integration

- Login and session restoration use `POST /auth/login`, `POST /auth/refresh`, and `GET /auth/me`.
- User chat streams from `POST /v1/chat/completions`.
- Administration calls the protected `/v1/admin/*` routes and authenticated `/stream/metrics`.
- Conversation history and projects are stored per account in browser local storage. Access and refresh tokens are stored in session storage.
- The administrator workspace is only rendered when the authenticated API user has the exact role `admin`. The client does not infer, convert, or grant roles.
- Visitors enter a browser-local guest workspace without signing in. Guest history and projects are stored under a separate browser-local guest ID; no guest tokens or account credentials are created.
- Guest inference requires explicit backend support for unauthenticated chat, including rate/abuse limits and data-retention semantics. Until that contract is implemented, a guest API 401 is shown as a clear service limitation rather than being bypassed.
- Google and GitHub sign-in controls remain disabled until Claude publishes the backend OAuth contract and provider configuration. Do not invent callback routes or handle provider secrets in the browser.

## Backend work required

The UI deliberately reports missing capabilities instead of implying unsupported operations are live. The following backend contracts are needed to complete those screens:

- Return a consistent role value from `/auth/me` and make the bootstrap administrator's role accepted by protected admin routes. The current bootstrap login's `/auth/me` role is `UserRole.ADMIN`, while `/v1/admin/health` rejects that identity with `403`; the UI does not work around this authorization failure.
- Add protected user listing, role-management, invitation, and account lifecycle endpoints.
- Add API-key lifecycle and token/usage reporting endpoints.
- Add protected security and audit-log endpoints.
- Add an authenticated user-facing model listing/selection contract.
- Add dataset inventory and source-level status endpoints.
- Make model deploy/rollback activate the requested checkpoint and report the resulting active model. The current endpoint only acknowledges that the request is queued.
- Ensure training launch uses the settings accepted from the UI, or expose the effective settings and clearly report that the request values are ignored.
- Define and implement guest chat access for `POST /v1/chat/completions`: anonymous request policy, rate limiting and abuse controls, model access, conversation retention, and error responses.
- Define secure Google and GitHub OAuth flows: provider availability/configuration, authorization initiation and callback/redirect behavior, state/CSRF and PKCE protections as applicable, account linking/provisioning and default role, token/session issuance, and failure responses. Coordinate exact routes and payloads before enabling the frontend buttons.

Do not expose these endpoints without server-side authorization. Once implemented, wire them through the shared client in `src/api.ts`; do not add a parallel auth or API layer.
