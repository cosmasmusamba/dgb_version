# Development Responsibility Matrix

This document records the agreed division of development work for DGB. Claude and Copilot are separate contributors; these responsibilities have not been delegated or acknowledged by either agent unless the project owner explicitly confirms that handover.

## Responsibility allocation

| Area | Owner | Scope |
|---|---|---|
| Backend | Claude | Backend architecture and implementation; database design, migrations, and integrity; authentication and authorization; user/admin roles and permissions; APIs; AI model integration and inference; tokenizer and transformer integration; training and dataset pipelines; backend security and performance; API documentation and backend tests. |
| Frontend | Copilot | DGB user chat workspace; administrator dashboard; UI/UX, branding, themes, and design system; frontend components and navigation; responsive layouts; client-side state; integration with agreed backend APIs; browser testing and frontend validation. |

Ownership applies to implementation and maintenance, not just initial delivery. A task crossing the boundary must be coordinated before changes begin.

## Ownership boundaries

- Claude owns backend source, database schemas and migrations, authorization policy, API contracts and their documentation, and backend/model/data-pipeline behavior.
- Copilot owns the frontend under `ui/portal/`, including browser-side API integration. Copilot must not compensate for backend authorization or contract defects by weakening client checks, fabricating authenticated responses, or adding duplicate backend functionality.
- The frontend may show clearly labeled development-only fixtures when a backend capability is unavailable. Fixtures must not be used as production data or presented as a completed backend operation.
- Neither agent modifies files owned by the other without explicit coordination and approval from the project owner. Shared documentation or configuration changes should be discussed when they affect both sides.
- Existing backend APIs and database contracts should be preserved unless Claude communicates a proposed change and Copilot has had a chance to update and validate the frontend integration.

## Shared integration contracts

- The backend is the source of truth for identity, roles, permissions, validation, and operation results. The frontend may hide or show UI based on returned identity data, but the backend must enforce authorization on every protected request.
- Before changing an API request or response, authentication behavior, event-stream format, error shape, or persistence contract, the owner communicates the proposed contract to the other agent and the project owner. Include affected routes, payloads, response examples, authorization requirements, error cases, and rollout or compatibility expectations.
- The receiving agent confirms impact and implementation timing before the contract change is treated as ready for integration. Keep interfaces backward-compatible where practical; document any intentional breaking change and coordinate both sides before merging.
- The frontend uses the existing authentication and API services. Do not introduce a second authentication mechanism or duplicate business logic in the browser.
- Record integration assumptions, unavailable endpoints, and blockers in this document or the relevant project documentation. Update the record when a blocker is resolved or the contract changes.

## Communication and handover procedures

1. For work that crosses ownership boundaries, send a concise handover before implementation: context, owner, exact files or endpoints affected, proposed contract, acceptance criteria, tests, and known risks.
2. The receiving agent acknowledges the handover and raises questions or conflicts before changing the shared interface. Do not assume an agent has seen or accepted a message that the project owner has not delivered.
3. During implementation, report contract deviations and newly discovered blockers promptly. Keep changes within the assigned ownership boundary unless the project owner explicitly authorizes an exception.
4. At handover, provide changed files, branch and commit status, verification performed, unresolved issues, and steps needed by the other owner. Do not commit, push, merge, or deploy without project-owner approval.

## Current blockers and integration dependencies

### Bootstrap administrator authorization mismatch

The frontend integration check found:

- Bootstrap administrator login succeeds.
- `GET /auth/me` returns role `UserRole.ADMIN`.
- `GET /v1/admin/health` returns HTTP `403` for that authenticated administrator.

**Owner: Claude.** Investigate and resolve the backend role serialization/authorization inconsistency, preserving server-side authorization. Do not ask the frontend to normalize the role or bypass the protected route. After a backend fix, jointly verify the returned identity and protected admin endpoints using an authenticated administrator session.

### Additional backend capabilities needed by the UI

The current frontend documents or handles these as unavailable rather than pretending they are live. Claude owns backend/API design and implementation; Copilot owns wiring the agreed contracts into the UI after coordination:

- User listing, role management, invitations, and account lifecycle operations.
- API-key lifecycle and inference/token-usage reporting.
- Security and audit-log endpoints.
- An authenticated user-facing active-model listing and selection contract.
- Dataset inventory and source-level status.
- Model deploy/rollback that actually activates a checkpoint and reports the active model; current acknowledgements indicate the action is queued.
- Clear training-launch semantics: apply submitted settings or report the effective settings and explain any ignored values.

Endpoint additions must remain protected by server-side authorization and be documented and tested by the backend owner.

## Git and change management

- Frontend work remains isolated on `feature/dgb-ui-workspaces`.
- Claude performs backend work on a separate backend feature branch, not on the frontend branch.
- Neither agent modifies the T0.1 test branch.
- No commits, pushes, merges, or deployments without the project owner's explicit approval.
- Preserve rollback capability: keep branch boundaries, avoid overwriting unrelated work, and identify a known-good checkpoint or commit before integrating coordinated changes.
- Keep generated files, local data, credentials, and unrelated worktree changes out of a task's changes. Do not revert or delete another contributor's work.

## Testing and release responsibilities

| Verification | Owner |
|---|---|
| Backend unit/integration tests, database migration and integrity checks, auth/authorization tests, API contract tests, model/inference and pipeline tests, backend security/performance checks | Claude |
| Frontend type-check/build, component behavior, responsive and accessibility checks, browser navigation/forms/dialogs, client state, API integration behavior, fixture exclusion from production bundle | Copilot |
| Cross-boundary contract verification, authenticated administrator flow, coordinated regression checks, release readiness and rollback decision | Claude and Copilot coordinate; project owner approves |

Each owner reports the exact checks run and their results. Do not treat frontend fixtures as evidence that backend authorization or API integration works. A release or merge requires owner approval and successful checks appropriate to both affected layers; neither agent independently deploys or merges.
