# Studio Naming Migration Plan

## Goal

Replace all `tts_project` naming with `studio` naming across backend and frontend for the unreleased Studio feature, without preserving legacy route compatibility.

## Final Contract

- Backend API module name: `api::studio`
- Backend route base: `/v1/studio`
- Backend route resources:
  - `GET, POST /v1/studio/projects`
  - `GET, PATCH, DELETE /v1/studio/projects/:project_id`
  - `GET /v1/studio/projects/:project_id/audio`
  - `GET, PATCH /v1/studio/projects/:project_id/meta`
  - `GET, POST /v1/studio/projects/:project_id/pronunciations`
  - `DELETE /v1/studio/projects/:project_id/pronunciations/:pronunciation_id`
  - `GET, POST /v1/studio/projects/:project_id/snapshots`
  - `POST /v1/studio/projects/:project_id/snapshots/:snapshot_id/restore`
  - `GET, POST /v1/studio/projects/:project_id/render-jobs`
  - `PATCH /v1/studio/projects/:project_id/render-jobs/:job_id`
  - `GET, PATCH, DELETE /v1/studio/projects/:project_id/segments/:segment_id`
  - `POST /v1/studio/projects/:project_id/segments/:segment_id/split`
  - `POST /v1/studio/projects/:project_id/segments/:segment_id/merge-next`
  - `PATCH /v1/studio/projects/:project_id/segments/reorder`
  - `POST /v1/studio/projects/:project_id/segments/bulk-delete`
  - `POST /v1/studio/projects/:project_id/segments/:segment_id/render`
  - `GET, POST /v1/studio/folders`
- Frontend API type names: `StudioProject*`
- Frontend API methods: `listStudioProjects`, `createStudioProject`, etc.
- Frontend Studio page route remains `/studio` and `/studio/:projectId`.
- Remove legacy compatibility route path `/tts-projects`.

## Non-Goals

- No requirement to preserve `/v1/tts-projects*` compatibility endpoints.
- No requirement to keep `TtsProject*` aliases in frontend or backend.

## Verification Gates

- Backend: `cargo check -p izwi-server`
- Frontend types: `cd ui && npm run typecheck`
- Frontend API tests: `cd ui && npm test -- src/shared/api/audio.test.ts`
- Studio route/tests: `cd ui && npm test -- src/features/PageHeaderHistoryButtons.test.tsx src/features/voice-studio/route.test.tsx`

