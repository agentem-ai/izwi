import {
  AlertTriangle,
  ExternalLink,
  Loader2,
  MoreVertical,
  Trash2,
} from "lucide-react";
import type { StudioProjectMetaRecord, StudioProjectSummary } from "@/api";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/utils";

interface StudioProjectHistoryTableProps {
  projects: StudioProjectSummary[];
  projectMetaById?: Record<string, StudioProjectMetaRecord>;
  loading?: boolean;
  error?: string | null;
  deletePending?: boolean;
  loadMore?: {
    canLoadMore: boolean;
    loading: boolean;
    onLoadMore: () => void;
  };
  onRefresh?: () => void;
  onCreateProject?: () => void;
  onOpenProject: (projectId: string) => void;
  onRequestDeleteProject?: (project: StudioProjectSummary) => void;
}

function formatCreatedAt(timestampMs: number): string {
  if (!Number.isFinite(timestampMs)) {
    return "Unknown time";
  }
  const value = new Date(timestampMs);
  if (Number.isNaN(value.getTime())) {
    return "Unknown time";
  }
  return value.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function rowLabel(project: StudioProjectSummary): string {
  return project.name || project.model_id || project.id;
}

export function StudioProjectHistoryTable({
  projects,
  projectMetaById = {},
  loading = false,
  error = null,
  deletePending = false,
  loadMore,
  onRefresh,
  onCreateProject,
  onOpenProject,
  onRequestDeleteProject,
}: StudioProjectHistoryTableProps) {
  if (loading) {
    return (
      <div className="mb-6 flex min-h-[20rem] items-center justify-center rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] text-sm text-[var(--text-muted)]">
        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
        Loading Studio projects...
      </div>
    );
  }

  if (error) {
    return (
      <div className="mb-6 rounded-2xl border border-[var(--danger-border)] bg-[var(--danger-bg)] p-4 text-sm text-[var(--danger-text)]">
        <div className="flex items-start justify-between gap-3">
          <div className="flex items-start gap-3">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
            <p>{error}</p>
          </div>
          {onRefresh ? (
            <Button type="button" variant="outline" size="sm" onClick={onRefresh}>
              Retry
            </Button>
          ) : null}
        </div>
      </div>
    );
  }

  if (projects.length === 0) {
    return (
      <div className="mb-6 rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-10 text-center">
        <h3 className="text-lg font-semibold text-[var(--text-primary)]">
          No Studio projects yet
        </h3>
        <p className="mt-2 text-sm text-[var(--text-muted)]">
          Saved long-form text-to-speech projects will appear here.
        </p>
        {onCreateProject ? (
          <div className="mt-4">
            <Button type="button" onClick={onCreateProject}>
              New project
            </Button>
          </div>
        ) : null}
      </div>
    );
  }

  return (
    <div className="mb-6 overflow-hidden rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)]">
      <div className="overflow-x-auto">
        <table className="min-w-full border-collapse text-sm">
          <thead className="bg-[var(--bg-surface-1)] text-left text-xs uppercase tracking-[0.14em] text-[var(--text-muted)]">
            <tr>
              <th className="px-4 py-3 font-semibold sm:px-5">Created</th>
              <th className="px-4 py-3 font-semibold">Project</th>
              <th className="px-4 py-3 font-semibold">Progress</th>
              <th className="px-4 py-3 font-semibold">Segments</th>
              <th className="px-4 py-3 font-semibold">Model</th>
              <th className="w-[56px] px-3 py-3 text-right font-semibold sm:px-4">
                <span className="sr-only">Actions</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {projects.map((project) => {
              const progressPercent =
                project.segment_count > 0
                  ? Math.round(
                      (project.rendered_segment_count / project.segment_count) * 100,
                    )
                  : 0;
              const statusLabel =
                progressPercent >= 100
                  ? "Ready"
                  : project.rendered_segment_count > 0
                    ? "In progress"
                    : "Not rendered";
              const statusClass =
                progressPercent >= 100
                  ? "border-[var(--status-positive-border)] bg-[var(--status-positive-bg)] text-[var(--status-positive-text)]"
                  : "border-[var(--status-warning-border)] bg-[var(--status-warning-bg)] text-[var(--status-warning-text)]";
              const tags = projectMetaById[project.id]?.tags ?? [];

              return (
                <tr
                  key={project.id}
                  aria-label={`Open Studio project ${rowLabel(project)}`}
                  className="cursor-pointer border-t border-[var(--border-muted)] transition-colors hover:bg-[var(--bg-surface-1)]"
                  onClick={(event) => {
                    if ((event.target as HTMLElement).closest("[data-row-action]")) {
                      return;
                    }
                    onOpenProject(project.id);
                  }}
                  onKeyDown={(event) => {
                    if ((event.target as HTMLElement).closest("[data-row-action]")) {
                      return;
                    }
                    if (event.key === "Enter" || event.key === " ") {
                      event.preventDefault();
                      onOpenProject(project.id);
                    }
                  }}
                  tabIndex={0}
                >
                  <td className="px-4 py-3 align-top text-[var(--text-secondary)] sm:px-5">
                    {formatCreatedAt(project.created_at)}
                  </td>
                  <td className="px-4 py-3 align-top">
                    <div className="font-medium text-[var(--text-primary)]">
                      {project.name}
                    </div>
                    <div className="mt-1 max-w-[28rem] text-xs text-[var(--text-muted)]">
                      {tags.length > 0
                        ? tags.slice(0, 3).join(" · ")
                        : project.source_filename || "Script project"}
                    </div>
                  </td>
                  <td className="px-4 py-3 align-top">
                    <div className="space-y-2">
                      <span
                        className={cn(
                          "inline-flex rounded-full border px-2 py-0.5 text-[10px] font-medium uppercase tracking-[0.12em]",
                          statusClass,
                        )}
                      >
                        {statusLabel}
                      </span>
                      <div className="flex items-center gap-2">
                        <div className="h-1.5 w-24 overflow-hidden rounded-full bg-[var(--bg-surface-2)]">
                          <div
                            className="h-full rounded-full bg-[var(--accent-solid)] transition-[width] duration-300"
                            style={{ width: `${progressPercent}%` }}
                          />
                        </div>
                        <span className="text-xs text-[var(--text-secondary)]">
                          {progressPercent}%
                        </span>
                      </div>
                    </div>
                  </td>
                  <td className="px-4 py-3 align-top text-[var(--text-secondary)]">
                    <div className="font-medium text-[var(--text-primary)]">
                      {project.rendered_segment_count}/{project.segment_count}
                    </div>
                    <div className="mt-1 text-xs text-[var(--text-muted)]">
                      {project.total_chars} chars
                    </div>
                  </td>
                  <td className="px-4 py-3 align-top text-[var(--text-secondary)]">
                    {project.model_id || "Not set"}
                  </td>
                  <td className="px-3 py-2 align-top text-right sm:px-4">
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button
                          type="button"
                          variant="ghost"
                          size="icon"
                          data-row-action
                          className="h-8 w-8 rounded-full text-[var(--text-muted)] hover:bg-[var(--bg-surface-2)] hover:text-[var(--text-primary)]"
                          aria-label={`More actions for ${rowLabel(project)}`}
                          onClick={(event) => event.stopPropagation()}
                          onKeyDown={(event) => event.stopPropagation()}
                        >
                          <MoreVertical className="h-4 w-4" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent
                        align="end"
                        className="w-48"
                        onClick={(event) => event.stopPropagation()}
                      >
                        <DropdownMenuItem onSelect={() => onOpenProject(project.id)}>
                          <ExternalLink className="mr-2 h-4 w-4" />
                          Open project
                        </DropdownMenuItem>
                        <DropdownMenuSeparator />
                        <DropdownMenuItem
                          disabled={!onRequestDeleteProject || deletePending}
                          onSelect={() => onRequestDeleteProject?.(project)}
                          className="text-[var(--danger-text)] focus:text-[var(--danger-text)]"
                        >
                          <Trash2 className="mr-2 h-4 w-4" />
                          Delete
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {loadMore?.canLoadMore ? (
        <div className="flex justify-center border-t border-[var(--border-muted)] px-4 py-3 sm:px-5">
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="h-9 gap-2"
            onClick={loadMore.onLoadMore}
            disabled={loadMore.loading}
          >
            {loadMore.loading ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
            Load more
          </Button>
        </div>
      ) : null}
    </div>
  );
}
