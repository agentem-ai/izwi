# Transcription Summary Contract

This document defines the implementation contract for AI-generated summaries on the `/transcription` route.

## Product Behavior

- Every saved transcription record should automatically start summary generation after transcript persistence succeeds.
- Summary generation must not block transcription delivery to the UI.
- Summary generation uses the chat model variant `Qwen3.5-4B`.
- The summary prompt must include the full transcription text.
- Users can manually regenerate a summary for an existing transcription record.

## Summary Lifecycle

Summary status values:

- `not_requested`: No summary has been queued (for legacy or empty transcripts).
- `pending`: Summary generation is in progress.
- `ready`: Summary text is available.
- `failed`: Summary generation failed; an error message is stored.

Lifecycle transitions:

- New non-empty transcript: `pending` -> (`ready` | `failed`)
- New empty transcript: `not_requested`
- Regenerate action: (`ready` | `failed` | `not_requested`) -> `pending` -> (`ready` | `failed`)

## Backend Requirements

- Persist summary state with each transcription record.
- Perform generation asynchronously in a background task after record creation.
- Persist the model id used for summary generation.
- Persist timestamp of last summary update.
- Persist a bounded error string when generation fails.
- Sanitize model output before persistence (for example remove `<think>` blocks and code fences).

## API Requirements

- Include summary fields on list and detail transcription endpoints.
- Add a dedicated regenerate endpoint:
  - `POST /v1/transcriptions/:record_id/summary/regenerate`
- Regenerate endpoint returns the updated record in `pending` state immediately.

## UI Requirements

- Show summary status and summary text in transcription review UI.
- Show summary status for active output and history detail views.
- Provide a regenerate summary action in the UI.
- Poll pending summaries until completion so status updates are reflected without manual refresh.

## Non-Goals (Initial Rollout)

- Custom summary templates.
- User-selectable summary model for transcription route.
- Cross-record batch summary jobs.
