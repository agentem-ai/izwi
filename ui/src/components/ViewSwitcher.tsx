import { motion } from "framer-motion";
import { Volume2, Users, Wand2, AudioWaveform, FileText } from "lucide-react";
import { ViewMode, VIEW_CONFIGS } from "../types";
import clsx from "clsx";

interface ViewSwitcherProps {
  currentView: ViewMode;
  onViewChange: (view: ViewMode) => void;
  modelCounts: Record<ViewMode, { total: number; ready: number }>;
}

const ICONS = {
  Volume2,
  Users,
  Wand2,
  AudioWaveform,
  FileText,
};

export function ViewSwitcher({
  currentView,
  onViewChange,
  modelCounts,
}: ViewSwitcherProps) {
  const views = Object.values(VIEW_CONFIGS).filter((v) => !v.disabled);

  return (
    <div className="flex flex-col sm:flex-row gap-2 p-1 bg-[#0d0d0d] rounded-xl border border-[var(--border-muted)]">
      {views.map((view) => {
        const Icon = ICONS[view.icon as keyof typeof ICONS];
        const isActive = currentView === view.id;
        const counts = modelCounts[view.id];
        const hasReadyModel = counts.ready > 0;

        return (
          <button
            key={view.id}
            onClick={() => onViewChange(view.id)}
            className={clsx(
              "relative flex-1 flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200",
              isActive
                ? "bg-[var(--bg-surface-1)] border border-[var(--border-muted)]"
                : "hover:bg-[var(--bg-surface-2)] border border-transparent",
            )}
          >
            {isActive && (
              <motion.div
                layoutId="activeViewIndicator"
                className="absolute inset-0 bg-[var(--bg-surface-1)] rounded-lg border border-[var(--border-muted)]"
                transition={{ type: "spring", bounce: 0.2, duration: 0.6 }}
              />
            )}

            <div className="relative flex items-center gap-3">
              <div
                className={clsx(
                  "p-2 rounded-lg transition-colors",
                  isActive
                    ? "bg-white text-black"
                    : "bg-[var(--bg-surface-3)] text-gray-400",
                )}
              >
                <Icon className="w-4 h-4" />
              </div>

              <div className="text-left">
                <div className="flex items-center gap-2">
                  <span
                    className={clsx(
                      "text-sm font-medium transition-colors",
                      isActive ? "text-white" : "text-gray-400",
                    )}
                  >
                    {view.label}
                  </span>
                  {hasReadyModel && (
                    <span className="w-1.5 h-1.5 rounded-full bg-white" />
                  )}
                </div>
                <span className="text-xs text-gray-400 hidden sm:block">
                  {view.description}
                </span>
              </div>
            </div>
          </button>
        );
      })}
    </div>
  );
}
