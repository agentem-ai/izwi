import { useMemo } from "react";
import { useSearchParams } from "react-router-dom";
import { Library, Sparkles, Users } from "lucide-react";
import { PageHeader, PageShell } from "@/components/PageShell";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { WorkspacePanel } from "@/components/ui/workspace";

type VoiceStudioTab = "library" | "clone" | "design";

const TAB_LABELS: Record<VoiceStudioTab, string> = {
  library: "Library",
  clone: "Clone",
  design: "Design",
};

function resolveTab(value: string | null): VoiceStudioTab {
  if (value === "clone" || value === "design" || value === "library") {
    return value;
  }
  return "library";
}

function tabIcon(tab: VoiceStudioTab) {
  switch (tab) {
    case "clone":
      return Users;
    case "design":
      return Sparkles;
    default:
      return Library;
  }
}

export function VoiceStudioPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const activeTab = resolveTab(searchParams.get("tab"));

  const title = useMemo(() => TAB_LABELS[activeTab], [activeTab]);
  const Icon = useMemo(() => tabIcon(activeTab), [activeTab]);

  const handleTabChange = (nextValue: string) => {
    const nextTab = resolveTab(nextValue);
    const nextSearchParams = new URLSearchParams(searchParams);
    nextSearchParams.set("tab", nextTab);
    setSearchParams(nextSearchParams, { replace: true });
  };

  return (
    <PageShell>
      <PageHeader
        title="Voice Studio"
        description="Manage reusable voices and switch between cloning and design workflows in one place."
      />

      <Tabs value={activeTab} onValueChange={handleTabChange} className="w-full">
        <TabsList className="grid h-10 w-full max-w-[22rem] grid-cols-3 overflow-hidden rounded-[var(--radius-pill)] border-[var(--border-strong)] bg-[var(--bg-surface-2)] p-[2px] shadow-none">
          <TabsTrigger
            value="library"
            className="h-full rounded-[var(--radius-pill)] px-3 text-[13px] font-semibold text-[var(--text-muted)] data-[state=active]:bg-[var(--bg-surface-1)] data-[state=active]:text-[var(--text-primary)] data-[state=active]:shadow-none"
          >
            Library
          </TabsTrigger>
          <TabsTrigger
            value="clone"
            className="h-full rounded-[var(--radius-pill)] px-3 text-[13px] font-semibold text-[var(--text-muted)] data-[state=active]:bg-[var(--bg-surface-1)] data-[state=active]:text-[var(--text-primary)] data-[state=active]:shadow-none"
          >
            Clone
          </TabsTrigger>
          <TabsTrigger
            value="design"
            className="h-full rounded-[var(--radius-pill)] px-3 text-[13px] font-semibold text-[var(--text-muted)] data-[state=active]:bg-[var(--bg-surface-1)] data-[state=active]:text-[var(--text-primary)] data-[state=active]:shadow-none"
          >
            Design
          </TabsTrigger>
        </TabsList>
      </Tabs>

      <WorkspacePanel className="mt-5 p-5 sm:p-6">
        <div className="flex items-start gap-3">
          <div className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-2)] p-2 text-[var(--text-secondary)]">
            <Icon className="h-4 w-4" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-[var(--text-primary)]">
              {title} workspace
            </h3>
            <p className="mt-1 text-sm text-[var(--text-secondary)]">
              This foundation route is live. Full tab content is added in
              subsequent rollout commits.
            </p>
          </div>
        </div>
      </WorkspacePanel>
    </PageShell>
  );
}
