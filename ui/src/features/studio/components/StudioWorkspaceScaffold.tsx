import type { ReactNode } from "react";
import { Card } from "@/components/ui/card";

interface StudioWorkspaceScaffoldProps {
  library: ReactNode;
  editor: ReactNode;
  delivery: ReactNode;
}

export function StudioWorkspaceScaffold({
  library,
  editor,
  delivery,
}: StudioWorkspaceScaffoldProps) {
  return (
    <div className="space-y-5">
      <Card
        data-testid="studio-library-pane"
        className="rounded-2xl border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-5 shadow-none sm:p-6"
      >
        {library}
      </Card>

      <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_360px]">
        <Card
          data-testid="studio-editor-pane"
          className="rounded-2xl border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-5 shadow-none sm:p-6"
        >
          {editor}
        </Card>

        <aside
          data-testid="studio-delivery-pane"
          className="space-y-5"
        >
          {delivery}
        </aside>
      </div>
    </div>
  );
}
