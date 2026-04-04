# Diarization Alignment Plan

## Goal

Align the `/diarization` route with the current `/transcription` route UX and structure while preserving diarization-specific capabilities:

- multi-model pipeline requirements
- upload and microphone capture
- speaker correction workflow
- reruns from quality controls
- summary regeneration and polling

## What Changed On Transcription

- The route was split into a collection view and a dedicated record view.
- History loading and record loading were extracted into route-level hooks.
- The old inline setup flow was replaced by a dedicated `NewTranscriptionModal`.
- New jobs now navigate straight to `/transcription/:recordId`.
- History is now shown as a simple operational table instead of a drawer workflow.
- Record actions were concentrated in a dedicated detail page: back, delete, copy, export, summary regeneration, and polling.
- Model setup moved onto reusable route-selection patterns with grouped model sections in `RouteModelModal`.
- The last polish pass focused on scan speed, modal styling, delete confirmation styling, background polling, and live delta updates.

## Current Diarization Understanding

- `/diarization` is still a single-route experience driven by `ui/src/features/diarization/route.tsx` and the older shared `ui/src/components/DiarizationPlayground.tsx`.
- Job creation, session state, review workspace, and latest-record state all live inside the playground instead of the route.
- Saved history is still handled through `ui/src/components/DiarizationHistoryPanel.tsx`, which uses a drawer plus a custom full-screen modal instead of dedicated route pages.
- The persisted diarization API already supports the record-centric workflow we need: list, get, create, update speaker names, rerun, regenerate summary, delete, and fetch record audio.
- Diarization creation is synchronous for persisted records and does not support the transcription-style streaming create flow, so the alignment should copy the route pattern, not the streaming behavior.
- Diarization has extra feature depth that transcription does not: pipeline model readiness, load-all pipeline controls, speaker name correction, quality reruns that produce new records, and summary polling for completed records.

## Main Alignment Gaps

- No `/diarization/:recordId` route yet.
- No route-level history or record hooks.
- No dedicated diarization creation modal.
- No transcription-style history table.
- No dedicated diarization record detail page.
- Model-selection logic is still mostly route-local instead of reusing the newer route-selection pattern where it fits.
- Regression coverage is component-heavy, but route-level coverage is missing.

## Phased Plan

- [x] Phase 1: Establish route-owned diarization state and routing
  Scope:
  Add `/diarization/:recordId` routing, create `useDiarizationHistory` and `useDiarizationRecord` hooks, and move history/record data ownership out of the playground so the route becomes the orchestration layer.
  Notes:
  Keep the visible UX mostly unchanged in this phase if possible. The goal is to create the same architectural footing that transcription now has before swapping the UI patterns.
  Verification:
  Add route tests covering `/diarization` and `/diarization/:recordId`, plus hook-driven polling expectations for pending summaries.
  Commit:
  `feat(ui): scaffold diarization record routes and data hooks`

- [x] Phase 2: Replace drawer-first history with a table and dedicated record pages
  Scope:
  Introduce a `DiarizationHistoryTable` and a `DiarizationRecordDetail` flow modeled after transcription. Move history record opening from drawer/modal interactions to route navigation. Preserve delete, copy, export, summary regeneration, speaker corrections, rerun access, and audio playback from the dedicated record page.
  Notes:
  This is the biggest alignment step because it changes the mental model from “single playground plus popups” to “index page plus detail page”.
  Verification:
  Add route tests for opening records from the table, back navigation, delete navigation behavior, and record summary polling.
  Commit:
  `feat(ui): move diarization history into dedicated record pages`

- [x] Phase 3: Move diarization job creation into a dedicated modal
  Scope:
  Build a `NewDiarizationModal` that takes over the inline session setup currently embedded in `DiarizationPlayground`. The modal should keep diarization-specific controls: upload and microphone entry points, speaker range, timing windows, pipeline readiness, and route-model guidance.
  Notes:
  This should align to the transcription modal layout principles, not blindly duplicate its fields. Diarization still needs richer setup than transcription.
  Verification:
  Add tests for opening the modal from the page header, enforcing model/pipeline readiness, creating a record from upload, and navigating to the new diarization record after creation.
  Commit:
  `feat(ui): move diarization setup into a creation modal`

- [x] Phase 4: Polish the diarization record workspace to match transcription conventions
  Scope:
  Refine the new record page so it matches the transcription detail rhythm: cleaner header metadata, grouped actions, better empty/loading/error states, styled delete confirmation, summary guidance, and background-safe refresh behavior. Keep diarization-specific tabs for transcript, speakers, and quality.
  Notes:
  This is where the “creation modals etc” polish from transcription gets mirrored across the diarization detail experience.
  Verification:
  Add focused tests for record actions, summary regeneration errors, speaker-correction persistence, rerun navigation/selection behavior, and delete confirmation behavior.
  Commit:
  `feat(ui): polish the diarization record workspace`

- [x] Phase 5: Align model-management ergonomics and clean up legacy diarization UI
  Scope:
  Reuse the newer route model selection patterns where they fit, centralize preferred diarization model resolution, preserve pipeline sections and load-all behavior, and remove the now-obsolete drawer/modal history path and any dead playground-only orchestration.
  Notes:
  The expected end state is feature-scoped diarization route/components/hooks, with older shared-component ownership reduced or removed.
  Verification:
  Run the diarization route tests plus existing diarization component tests to confirm no regressions in export, review workspace, and quality workflows.
  Commit:
  `refactor(ui): align diarization model management and remove legacy history flow`

## Review

- The safest sequence is architecture first, then navigation, then creation modal, then polish, then cleanup.
- The main place to avoid false parity is streaming: transcription has persisted streaming creation, diarization does not.
- The main place to avoid regressions is reruns and speaker corrections, because both currently depend on state living inside the old playground/history modal stack.

# Settings Redesign Plan

## Goal

Redesign `/settings` to feel like a polished product surface instead of a stack of generic cards, using the OpenAI GPT-5.4 frontend guidance as the quality bar.

## Design Notes From OpenAI Reference

- App surfaces should default to restrained layout, strong typography, minimal chrome, and a single accent.
- Cards should only exist when they are the interaction container; layout sections should otherwise be plain structure.
- Utility copy should prioritize orientation, status, and action over marketing language.
- Dense information should remain readable through spacing, alignment, and hierarchy rather than decorative treatments.
- Motion and ornament should be restrained; product UI should avoid gradient-heavy dashboard styling.

## Plan

- [x] Audit the current `/settings` information hierarchy and identify where card styling can be removed.
- [x] Rebuild the page as a cleaner settings surface with section dividers, row-based controls, and switches instead of checkboxes.
- [x] Preserve update, theme, and analytics behavior while improving scan speed and state visibility.
- [x] Verify the route compiles and the new UI follows existing theme tokens on desktop and mobile.

## Review

- Replaced the stacked card layout with a single structured settings surface that uses section dividers and row alignment for faster scanning.
- Swapped the analytics checkbox for the shared switch control and kept the optimistic-save behavior intact.
- Kept theme and update functionality unchanged while presenting state through tighter utility copy, badges, and inline metadata instead of boxed panels.
- Follow-up refinement removed non-essential overview content, update diagnostics, and privacy explainer blocks to make the page calmer and denser.
- Verification: `npm run typecheck` and `npm run build` passed in `ui/`.


# Transcription History Actions Plan

## Goal

Add a standard row actions menu to the `/transcription` history table with a three-dot trigger, appropriate quick actions, and delete confirmation that refreshes the list after removal.

## Planned UX

- Add a trailing overflow menu on every history row using a vertical three-dot trigger.
- Keep row click to open the record, but stop propagation for menu interactions.
- Include only actions that fit a history list standard: open record, copy transcript, export, and delete.
- Keep summary regeneration on the detail page because it is a heavier, model-dependent action rather than a common list action.
- Show delete confirmation in a modal and refresh history after successful deletion.

## Plan

- [x] Audit current transcription history row capabilities and detail-page actions.
- [x] Add a row overflow menu with standard quick actions and correct interaction handling.
- [x] Implement delete confirmation modal and refresh the history data after deletion.
- [x] Add or update tests for row menu actions and delete refresh behavior.

## Review

- Added a trailing three-dot actions menu to each transcription history row with `Open record`, `Copy transcript`, `Export`, and `Delete`.
- Kept `Regenerate summary` on the detail page because it depends on summary-model readiness and is less appropriate as a routine list action.
- Added a row-level delete confirmation modal and refreshed the history list after successful deletion.
- Updated the shared dropdown primitive so interactive items show a pointer cursor.
- Verification: `npm run typecheck` and `npm run test -- src/features/transcription/route.test.tsx`.

# Diarization And TTS History Actions Plan

## Goal

Align `/diarization` and `/text-to-speech` with the standard row-actions pattern now used on `/transcription`, so each history row has a conventional overflow menu with only appropriate quick actions.

## Planned UX

- Add a trailing three-dot menu to every history row on both routes.
- Keep row click to open the record, while preventing the menu trigger and items from accidentally opening the row.
- Use only standard list-level actions:
  - `/diarization`: `Open record`, `Copy transcript`, `Export`, `Delete`
  - `/text-to-speech`: `Open record`, `Copy text`, `Download`, `Delete`
- Keep heavier or workflow-specific actions on the detail pages, such as summary regeneration, reruns, and speaker correction.
- Confirm deletion in a modal and refresh the history data after successful removal.

## Plan

- [x] Audit existing detail-page actions and shared helpers for both routes.
- [x] Add row overflow menus and action handlers to the diarization and TTS history tables.
- [x] Wire parent-owned delete callbacks so history refresh stays consistent after removal.
- [x] Extend any shared dialog component needed for row-triggered export flows.
- [x] Add focused route tests for menu items and delete-refresh behavior on both routes.
- [x] Verify with `npm run typecheck` and focused route tests.

## Review

- Added a three-dot actions menu to each diarization history row with `Open record`, `Copy transcript`, `Export`, and `Delete`.
- Added a three-dot actions menu to each text-to-speech history row with `Open record`, `Copy text`, `Download`, and `Delete`.
- Kept heavier actions on detail pages, including diarization summary regeneration, speaker correction, and rerun controls.
- Added delete confirmation modals on both history tables and refreshed the route-owned history after successful deletion.
- Extended `DiarizationExportDialog` so it can open from a controlled row action instead of only a trigger child.
- Verification: `npm run typecheck`, `npm run test -- src/features/diarization/route.test.tsx`, `npm run test -- src/features/text-to-speech/route.test.tsx`, and `npm run test -- src/components/DiarizationExportDialog.test.tsx`.

# UUID Rollout Plan

## Research Notes

- Most persisted application records already use `TEXT` primary keys in SQLite, so new UUID-based IDs can be introduced without schema migrations.
- The current server mostly generates prefixed IDs such as `thread_*`, `msg_*`, `txr_*`, `dir_*`, `ttsp_*`, and `agent_sess_*`.
- Several runtime/API IDs already use plain UUIDs today, including core engine request IDs and some request-context values.
- A few frontend-only IDs are still timestamp/random based, including render queue items, transcript entries, and toast IDs.
- `ui/src/types.ts` contains semantic parsing for Kokoro voice IDs. Those are model identifiers, not generated record IDs, and should not be converted as part of this rollout.
- The default voice profile ID is a fixed constant for bootstrapping existing installs. Leave it unchanged in this phase so old data remains addressable.

## Phases

- [x] Phase 1: Introduce shared UUID helpers and switch all newly persisted server records to plain UUIDs.
  Scope:
  `chat_store`, `voice_store`, `voice_observation_store`, `saved_voice_store`, `speech_history_store`, `transcription_store`, `diarization_store`, and `studio_project_store`.
  Deliverable:
  New rows created in storage use canonical UUID strings with no type-specific prefix, while existing rows remain readable.
  Verification:
  `cargo fmt --package izwi-server`, `cargo test -p izwi-server transcription_store -- --nocapture`, `cargo test -p izwi-server diarization_store -- --nocapture`, `cargo test -p izwi-server voice_store -- --nocapture`, `cargo test -p izwi-server voice_observation_store -- --nocapture`, `cargo test -p izwi-server studio_project_store -- --nocapture`, `cargo test -p izwi-server chat_store -- --nocapture`, `cargo test -p izwi-server speech_history_store -- --nocapture`, and `cargo test -p izwi-server saved_voice_store -- --nocapture`.
  Commit:
  `refactor(server): use uuid ids for newly persisted records`

- [x] Phase 2: Switch remaining newly created server/runtime session and API object IDs to UUIDs.
  Scope:
  agent session records, OpenAI-compatible chat completion IDs, OpenAI-compatible response/message/tool-call IDs, and any remaining server-generated IDs that are user-visible but not persisted in the main stores.
  Deliverable:
  New server-generated IDs are consistently UUIDs across storage and API surfaces.
  Verification:
  Run focused server tests for affected API modules.
  Commit:
  `refactor(server): use uuid ids for runtime and api objects`

- [x] Phase 3: Switch client-generated new record IDs to UUIDs and update tests.
  Scope:
  render queue item IDs, realtime transcript entry IDs, toast IDs, and any other client-generated record IDs discovered during implementation.
  Deliverable:
  Client-side new IDs use `crypto.randomUUID()` through a shared helper instead of timestamp/random concatenation.
  Verification:
  Run targeted UI tests for the affected helpers/components.
  Commit:
  `refactor(ui): use uuid ids for client-generated records`

## Review

- Phase 1 complete.
- Added a shared server UUID helper and switched all newly persisted record IDs in the main stores from prefixed IDs to canonical UUID strings.
- Phase 2 complete.
- Switched agent session IDs plus OpenAI-compatible completion, response, tool-call, and assistant message IDs to canonical UUID strings.
- Focused verification:
  - `cargo fmt --package izwi-server`
  - `cargo test -p izwi-server state::tests::trim_store_by_uses_updated_at_for_agent_sessions -- --nocapture`
  - `cargo test -p izwi-server parses_qwen_tool_call_output_into_openai_shape -- --nocapture`
  - `cargo test -p izwi-server builds_tool_call_finish_reason_when_tool_output_detected -- --nocapture`
  - `cargo test -p izwi-server normalizes_assistant_tool_calls_into_qwen_xml -- --nocapture`
- Broad verification note:
  - `cargo test -p izwi-server openai -- --nocapture` still fails in pre-existing content-flattening tests unrelated to the UUID edits:
    - `api::openai::chat::completions::tests::flattens_text_parts_content`
    - `api::openai::responses::handlers::tests::flattens_part_content`
- Phase 3 complete.
- Switched client-generated toast IDs, realtime transcript entry IDs, and studio render-queue item IDs to shared UUID generation.
- Focused verification:
  - `npm run typecheck`
  - `npm run test -- src/lib/ids.test.ts src/features/voice/realtime/support.test.ts`
- Scope note:
  - Remaining `Date.now()` usage in the UI is for timestamps, filenames, or visual randomness rather than application record IDs.

# Transcription Nested Modal Fix

## Plan

- [x] Confirm how the transcription route stacks `NewTranscriptionModal` and `RouteModelModal`.
- [x] Raise the model modal above the new transcription modal on `/transcription`.
- [x] Add a focused route test for opening model management from inside the new transcription modal.

## Review

- The transcription route now raises `RouteModelModal` to `z-[70]` while `NewTranscriptionModal` is open, so the model manager can sit above the creation dialog.
- Added a focused route test that verifies the model modal is promoted when the new transcription modal opens.
- Verification:
  - `npm run test -- src/features/transcription/route.test.tsx`
  - `npm run typecheck`
- Follow-up fix:
  - The underlying `NewTranscriptionModal` now prevents outside-dismiss interactions while the stacked model modal is open, so clicks in the top modal no longer close the transcription modal underneath.
  - The route test now verifies a click inside the stacked model modal still triggers model selection while keeping the transcription modal open.
