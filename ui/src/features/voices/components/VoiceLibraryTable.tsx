import { type KeyboardEvent } from "react";
import { Loader2 } from "lucide-react";

import { type VoicePickerItem } from "@/components/VoicePicker";
import { StatePanel } from "@/components/ui/state-panel";
import { StatusBadge } from "@/components/ui/status-badge";
import { cn } from "@/lib/utils";

interface VoiceLibraryTableProps {
  items: VoicePickerItem[];
  emptyTitle: string;
  emptyDescription: string;
  className?: string;
}

function renderPreviewCell(item: VoicePickerItem) {
  if (item.previewLoading) {
    return (
      <div className="inline-flex items-center gap-2 text-xs text-[var(--text-muted)]">
        <Loader2 className="h-3.5 w-3.5 animate-spin" />
        {item.previewMessage || "Generating preview..."}
      </div>
    );
  }

  if (item.previewUrl) {
    return (
      <audio
        controls
        src={item.previewUrl}
        className="h-9 w-full min-w-[14rem] max-w-[18rem]"
      />
    );
  }

  return (
    <p className="text-xs leading-5 text-[var(--text-muted)]">
      {item.previewMessage || "No preview available yet."}
    </p>
  );
}

export function VoiceLibraryTable({
  items,
  emptyTitle,
  emptyDescription,
  className,
}: VoiceLibraryTableProps) {
  if (items.length === 0) {
    return (
      <StatePanel
        title={emptyTitle}
        description={emptyDescription}
        align="center"
        dashed
        className={className}
      />
    );
  }

  return (
    <div
      className={cn(
        "overflow-hidden rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)]",
        className,
      )}
    >
      <div className="overflow-x-auto">
        <table className="min-w-[72rem] w-full border-collapse text-sm">
          <thead className="bg-[var(--bg-surface-1)] text-left text-xs uppercase tracking-[0.14em] text-[var(--text-muted)]">
            <tr>
              <th className="px-4 py-3 font-semibold">Voice</th>
              <th className="px-4 py-3 font-semibold">Type</th>
              <th className="px-4 py-3 font-semibold">Notes</th>
              <th className="px-4 py-3 font-semibold">Metadata</th>
              <th className="px-4 py-3 font-semibold">Preview</th>
              <th className="px-4 py-3 font-semibold text-right">Actions</th>
            </tr>
          </thead>
          <tbody>
            {items.map((item) => {
              const handleRowKeyDown = (event: KeyboardEvent<HTMLTableRowElement>) => {
                if (!item.onSelect) {
                  return;
                }
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  item.onSelect();
                }
              };

              return (
                <tr
                  key={item.id}
                  tabIndex={item.onSelect ? 0 : undefined}
                  onClick={item.onSelect}
                  onKeyDown={handleRowKeyDown}
                  data-testid={`voice-row-${item.id}`}
                  className={cn(
                    "border-t border-[var(--border-muted)] align-top",
                    item.onSelect &&
                      "cursor-pointer transition-colors hover:bg-[var(--bg-surface-1)] focus-visible:bg-[var(--bg-surface-1)] focus-visible:outline-none",
                  )}
                >
                  <td className="px-4 py-3">
                    <div className="font-semibold text-[var(--text-primary)]">
                      {item.name}
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <StatusBadge>{item.categoryLabel}</StatusBadge>
                  </td>
                  <td className="px-4 py-3 text-[var(--text-secondary)]">
                    <p className="line-clamp-2">
                      {item.description ||
                        "No reference notes were saved for this voice yet."}
                    </p>
                  </td>
                  <td className="px-4 py-3 text-[var(--text-secondary)]">
                    <div className="flex flex-wrap gap-1.5">
                      {(item.meta ?? []).map((meta) => (
                        <span
                          key={`${item.id}-${meta}`}
                          className="rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.08em]"
                        >
                          {meta}
                        </span>
                      ))}
                      {!item.meta?.length ? (
                        <span className="text-xs text-[var(--text-muted)]">n/a</span>
                      ) : null}
                    </div>
                  </td>
                  <td className="px-4 py-3">{renderPreviewCell(item)}</td>
                  <td className="px-4 py-3">
                    <div
                      className="flex flex-wrap justify-end gap-2"
                      onClick={(event) => event.stopPropagation()}
                    >
                      {item.actions}
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
