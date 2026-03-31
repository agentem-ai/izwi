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

- [ ] Phase 5: Align model-management ergonomics and clean up legacy diarization UI
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
