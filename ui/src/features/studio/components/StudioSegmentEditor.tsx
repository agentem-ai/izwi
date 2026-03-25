import {
  ArrowDown,
  ArrowUp,
  Link2,
  Loader2,
  MoreHorizontal,
  PencilLine,
  Play,
  Scissors,
  Settings,
  Trash2,
} from "lucide-react";
import type { StudioProjectRecord, StudioProjectSegmentRecord } from "@/api";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";

interface StudioSegmentEditorProps {
  project: StudioProjectRecord;
  segmentDrafts: Record<string, string>;
  segmentSelections: Record<string, number | null>;
  selectedSegmentIdSet: ReadonlySet<string>;
  selectedSegmentCount: number;
  queuedSegmentIdSet: ReadonlySet<string>;
  savingSegmentId: string | null;
  renderingSegmentId: string | null;
  onToggleSelectAll: () => void;
  onRenderSelected: () => void;
  onDeleteSelected: () => void;
  onToggleSegmentSelection: (segmentId: string, checked: boolean) => void;
  onSaveSegment: (segmentId: string) => void;
  onMoveSegment: (segmentId: string, direction: "up" | "down") => void;
  onMergeSegmentWithNext: (segmentId: string) => void;
  onSplitSegment: (segmentId: string) => void;
  onRenderSegment: (segmentId: string) => void;
  onDeleteSegment: (segmentId: string) => void;
  onOpenSegmentSettings: (segmentId: string) => void;
  onChangeSegmentDraft: (segmentId: string, value: string) => void;
  onChangeSegmentCursor: (segmentId: string, cursor: number | null) => void;
  audioUrlForRecordId: (recordId: string) => string;
}

function SegmentActionsMenu({
  segment,
  isFirst,
  isLast,
  canSplitSegment,
  canDeleteSegment,
  onMoveSegment,
  onMergeSegmentWithNext,
  onSplitSegment,
  onDeleteSegment,
}: {
  segment: StudioProjectSegmentRecord;
  isFirst: boolean;
  isLast: boolean;
  canSplitSegment: boolean;
  canDeleteSegment: boolean;
  onMoveSegment: (segmentId: string, direction: "up" | "down") => void;
  onMergeSegmentWithNext: (segmentId: string) => void;
  onSplitSegment: (segmentId: string) => void;
  onDeleteSegment: (segmentId: string) => void;
}) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="outline"
          size="icon"
          className="h-9 w-9 bg-[var(--bg-surface-0)]"
          aria-label={`More actions for segment ${segment.position + 1}`}
        >
          <MoreHorizontal className="h-4 w-4" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-44">
        {!isFirst ? (
          <DropdownMenuItem onSelect={() => onMoveSegment(segment.id, "up")}>
            <ArrowUp className="mr-2 h-4 w-4" />
            Move up
          </DropdownMenuItem>
        ) : null}
        {!isLast ? (
          <DropdownMenuItem onSelect={() => onMoveSegment(segment.id, "down")}>
            <ArrowDown className="mr-2 h-4 w-4" />
            Move down
          </DropdownMenuItem>
        ) : null}
        {!isLast ? (
          <DropdownMenuItem onSelect={() => onMergeSegmentWithNext(segment.id)}>
            <Link2 className="mr-2 h-4 w-4" />
            Merge next
          </DropdownMenuItem>
        ) : null}
        {canSplitSegment ? (
          <DropdownMenuItem onSelect={() => onSplitSegment(segment.id)}>
            <Scissors className="mr-2 h-4 w-4" />
            Split at cursor
          </DropdownMenuItem>
        ) : null}
        <DropdownMenuSeparator />
        <DropdownMenuItem
          onSelect={() => onDeleteSegment(segment.id)}
          disabled={!canDeleteSegment}
          className="text-[var(--danger-text)] focus:text-[var(--danger-text)]"
        >
          <Trash2 className="mr-2 h-4 w-4" />
          Delete
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

export function StudioSegmentEditor({
  project,
  segmentDrafts,
  segmentSelections,
  selectedSegmentIdSet,
  selectedSegmentCount,
  queuedSegmentIdSet,
  savingSegmentId,
  renderingSegmentId,
  onToggleSelectAll,
  onRenderSelected,
  onDeleteSelected,
  onToggleSegmentSelection,
  onSaveSegment,
  onMoveSegment,
  onMergeSegmentWithNext,
  onSplitSegment,
  onRenderSegment,
  onDeleteSegment,
  onOpenSegmentSettings,
  onChangeSegmentDraft,
  onChangeSegmentCursor,
  audioUrlForRecordId,
}: StudioSegmentEditorProps) {
  const allSelected =
    project.segments.length > 0 && selectedSegmentCount === project.segments.length;

  return (
    <>
      <div className="flex flex-wrap items-center gap-2">
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={onToggleSelectAll}
          className="bg-[var(--bg-surface-1)]"
        >
          {allSelected ? "Clear selection" : "Select all"}
        </Button>
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={onRenderSelected}
          disabled={selectedSegmentCount === 0}
          className="bg-[var(--bg-surface-1)]"
        >
          <Play className="h-4 w-4" />
          Render selected
        </Button>
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={onDeleteSelected}
          disabled={selectedSegmentCount === 0}
          className="bg-[var(--bg-surface-1)]"
        >
          <Trash2 className="h-4 w-4" />
          Delete selected
        </Button>
        {selectedSegmentCount > 0 ? (
          <span className="rounded-full border border-[var(--status-warning-border)] bg-[var(--status-warning-bg)] px-2.5 py-1 text-[11px] text-[var(--status-warning-text)]">
            {selectedSegmentCount} selected
          </span>
        ) : null}
      </div>

      <div className="mt-5 space-y-4">
        {project.segments.map((segment) => {
          const draft = segmentDrafts[segment.id] ?? segment.text;
          const segmentDirty = draft !== segment.text;
          const segmentNeedsRender = segmentDirty || !segment.speech_record_id;
          const isSaving = savingSegmentId === segment.id;
          const isRendering = renderingSegmentId === segment.id;
          const segmentQueued = queuedSegmentIdSet.has(segment.id);
          const splitIndex = segmentSelections[segment.id];
          const canSplitSegment =
            typeof splitIndex === "number" && splitIndex > 0 && splitIndex < draft.length;
          const isSelected = selectedSegmentIdSet.has(segment.id);
          const isFirst = segment.position === 0;
          const isLast = segment.position === project.segments.length - 1;
          const canDeleteSegment = project.segments.length > 1;
          const renderButtonLabel = segmentQueued
            ? "Queued"
            : segmentDirty
              ? "Save & render"
              : segmentNeedsRender
                ? "Render block"
                : "Re-render";

          return (
            <article
              key={segment.id}
              className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-4 sm:p-5"
            >
              <div className="grid gap-3 sm:grid-cols-[minmax(0,1fr)_auto] sm:items-start">
                <div className="min-w-0 space-y-1.5">
                  <div className="flex flex-wrap items-center gap-2 sm:flex-nowrap">
                    <div className="inline-flex items-center">
                      <Switch
                        checked={isSelected}
                        onCheckedChange={(checked) =>
                          onToggleSegmentSelection(segment.id, checked)
                        }
                        aria-label={`Select segment ${segment.position + 1}`}
                        className="h-5 w-9"
                      />
                    </div>
                    <span className="whitespace-nowrap text-xs font-semibold uppercase tracking-wider text-[var(--text-muted)]">
                      Segment {segment.position + 1}
                    </span>
                    <span
                      className={cn(
                        "whitespace-nowrap rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.16em]",
                        segmentDirty
                          ? "border-[var(--status-warning-border)] bg-[var(--status-warning-bg)] text-[var(--status-warning-text)]"
                          : segment.speech_record_id
                            ? "border-[var(--status-positive-border)] bg-[var(--status-positive-bg)] text-[var(--status-positive-text)]"
                            : "border-[var(--border-muted)] bg-[var(--bg-surface-0)] text-[var(--text-muted)]",
                      )}
                    >
                      {segmentDirty
                        ? "Edited"
                        : segment.speech_record_id
                          ? "Rendered"
                          : "Needs render"}
                    </span>
                  </div>
                  <div className="text-sm text-[var(--text-secondary)]">
                    {segment.input_chars} chars
                    {segment.audio_duration_secs
                      ? ` · ${segment.audio_duration_secs.toFixed(1)}s audio`
                      : ""}
                  </div>
                </div>

                <div className="flex w-full flex-wrap items-center gap-2 sm:w-auto sm:flex-nowrap sm:justify-end sm:gap-1.5 md:gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => onSaveSegment(segment.id)}
                    disabled={!segmentDirty || isSaving}
                    className="min-w-0 flex-1 basis-[calc(50%-0.25rem)] justify-center bg-[var(--bg-surface-0)] sm:basis-auto sm:flex-none"
                  >
                    {isSaving ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <PencilLine className="h-4 w-4" />
                    )}
                    Save draft
                  </Button>
                  <Button
                    size="sm"
                    onClick={() => onRenderSegment(segment.id)}
                    disabled={isRendering || segmentQueued}
                    className="min-w-0 flex-1 basis-[calc(50%-0.25rem)] justify-center sm:basis-auto sm:flex-none"
                  >
                    {isRendering ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Play className="h-4 w-4" />
                    )}
                    {renderButtonLabel}
                  </Button>
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={() => onOpenSegmentSettings(segment.id)}
                    className="order-2 ml-auto h-9 w-9 bg-[var(--bg-surface-0)] sm:order-none sm:ml-0"
                    aria-label={`Open settings for segment ${segment.position + 1}`}
                  >
                    <Settings className="h-4 w-4" />
                  </Button>
                  <div className="order-2 sm:order-none">
                    <SegmentActionsMenu
                      segment={segment}
                      isFirst={isFirst}
                      isLast={isLast}
                      canSplitSegment={canSplitSegment}
                      canDeleteSegment={canDeleteSegment}
                      onMoveSegment={onMoveSegment}
                      onMergeSegmentWithNext={onMergeSegmentWithNext}
                      onSplitSegment={onSplitSegment}
                      onDeleteSegment={onDeleteSegment}
                    />
                  </div>
                </div>
              </div>

              <Textarea
                className="mt-4 border-[var(--border-muted)] bg-[var(--bg-surface-0)]"
                value={draft}
                onChange={(event) => onChangeSegmentDraft(segment.id, event.target.value)}
                onSelect={(event) =>
                  onChangeSegmentCursor(segment.id, event.currentTarget.selectionStart)
                }
              />

              <div className="mt-2 text-xs text-[var(--text-muted)]">
                {canSplitSegment
                  ? "Split at cursor is ready in the actions menu."
                  : "Place the text cursor inside this block to enable split."}
              </div>

              {segmentDirty ? (
                <div className="mt-4 rounded-xl border border-[var(--status-warning-border)] bg-[var(--status-warning-bg)] px-3 py-2 text-xs text-[var(--status-warning-text)]">
                  This block has local edits. Rendering will save the latest text first
                  and then refresh the audio.
                </div>
              ) : null}

              {segment.speech_record_id ? (
                <div className="mt-4 space-y-2 rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-3">
                  <audio
                    src={audioUrlForRecordId(segment.speech_record_id)}
                    controls
                    preload="none"
                    className="h-10 w-full"
                  />
                  <div className="text-xs text-[var(--text-muted)]">
                    {segmentDirty
                      ? "Preview reflects the last rendered audio until you render this edited block again."
                      : `Linked generation: ${segment.speech_record_id}`}
                  </div>
                </div>
              ) : null}
            </article>
          );
        })}
      </div>
    </>
  );
}
