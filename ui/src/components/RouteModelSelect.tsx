import { useEffect, useMemo, useRef, useState, type ComponentProps } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  AlertTriangle,
  ArrowDownToLine,
  Check,
  CheckCircle2,
  ChevronDown,
  CircleDashed,
  Loader2,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { StatusBadge } from "@/components/ui/status-badge";
import { cn } from "@/lib/utils";

interface RouteModelSelectOption {
  value: string;
  label: string;
  statusLabel: string;
  isReady: boolean;
}

interface RouteModelSelectProps {
  value: string | null;
  options: RouteModelSelectOption[];
  onSelect?: (value: string) => void;
  placeholder?: string;
  className?: string;
  disabled?: boolean;
  menuPlacement?: "top" | "bottom";
}

function getStatusTone(
  option: RouteModelSelectOption,
): ComponentProps<typeof StatusBadge>["tone"] {
  const normalizedStatus = option.statusLabel.toLowerCase();

  if (option.isReady) {
    return "success";
  }
  if (
    normalizedStatus.includes("downloading") ||
    normalizedStatus.includes("loading")
  ) {
    return "warning";
  }
  if (normalizedStatus.includes("error")) {
    return "danger";
  }
  return "neutral";
}

function getStatusPresentation(option: RouteModelSelectOption): {
  icon: typeof CheckCircle2;
  className: string;
} {
  const normalizedStatus = option.statusLabel.toLowerCase();

  if (option.isReady) {
    return {
      icon: CheckCircle2,
      className: "text-[var(--status-positive-text)]",
    };
  }
  if (normalizedStatus.includes("downloaded")) {
    return {
      icon: ArrowDownToLine,
      className: "text-[var(--status-info-text)]",
    };
  }
  if (
    normalizedStatus.includes("downloading") ||
    normalizedStatus.includes("loading")
  ) {
    return {
      icon: Loader2,
      className: "text-[var(--status-warning-text)]",
    };
  }
  if (normalizedStatus.includes("error")) {
    return {
      icon: AlertTriangle,
      className: "text-[var(--danger-text)]",
    };
  }

  return {
    icon: CircleDashed,
    className: "text-[var(--text-muted)]",
  };
}

export function RouteModelSelect({
  value,
  options,
  onSelect,
  placeholder = "Select model",
  className,
  disabled = false,
  menuPlacement = "bottom",
}: RouteModelSelectProps) {
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const selectedOption = useMemo(
    () => options.find((option) => option.value === value) ?? null,
    [options, value],
  );
  const selectedStatus = selectedOption
    ? getStatusPresentation(selectedOption)
    : null;

  useEffect(() => {
    const handlePointerDown = (event: MouseEvent) => {
      if (
        containerRef.current &&
        event.target instanceof Node &&
        !containerRef.current.contains(event.target)
      ) {
        setIsOpen(false);
      }
    };

    window.addEventListener("mousedown", handlePointerDown);
    return () => window.removeEventListener("mousedown", handlePointerDown);
  }, []);

  return (
    <div ref={containerRef} className={cn("relative", className)}>
      <Button
        type="button"
        variant="outline"
        onClick={() => {
          if (!disabled && options.length > 0) {
            setIsOpen((current) => !current);
          }
        }}
        disabled={disabled || options.length === 0}
        className={cn(
          "h-10 w-full justify-between rounded-[var(--radius-md)] border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-3.5 font-normal text-[var(--text-primary)] shadow-[var(--shadow-soft)] transition-[border-color,background-color,box-shadow] hover:border-[var(--border-strong)] hover:bg-[var(--bg-surface-1)]",
          isOpen && "border-ring/50 ring-2 ring-ring/35",
        )}
      >
        <div className="min-w-0 flex flex-1 items-center gap-2">
          {selectedStatus ? (
            <selectedStatus.icon
              className={cn(
                "h-3.5 w-3.5 shrink-0",
                selectedStatus.className,
                selectedOption?.statusLabel.toLowerCase().includes("loading") &&
                  "animate-spin",
              )}
            />
          ) : null}
          <span
            className="min-w-0 flex-1 truncate text-left text-sm font-medium text-[var(--text-primary)]"
            title={selectedOption?.label || placeholder}
          >
            {selectedOption?.label || placeholder}
          </span>
        </div>
        <ChevronDown
          className={cn(
            "h-3.5 w-3.5 shrink-0 text-[var(--text-muted)] transition-transform",
            isOpen && "rotate-180",
          )}
        />
      </Button>

      <AnimatePresence>
        {isOpen ? (
          <motion.div
            initial={{ opacity: 0, y: 6, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 6, scale: 0.98 }}
            transition={{ duration: 0.16 }}
            className={cn(
              "absolute left-0 z-[90] min-w-full max-w-[min(36rem,calc(100vw-2rem))] rounded-[var(--radius-lg)] border border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-1.5 shadow-[var(--shadow-overlay)]",
              menuPlacement === "top" ? "bottom-[calc(100%+8px)]" : "top-[calc(100%+8px)]",
            )}
          >
            <div className="max-h-72 overflow-y-auto">
              {options.map((option) => (
                <button
                  key={option.value}
                  type="button"
                  onClick={() => {
                    onSelect?.(option.value);
                    setIsOpen(false);
                  }}
                  className={cn(
                    "relative flex min-w-[18rem] items-start gap-3 rounded-[var(--radius-sm)] px-3 py-2.5 text-left transition-colors hover:bg-[var(--bg-surface-1)]",
                    selectedOption?.value === option.value &&
                      "bg-[var(--bg-surface-1)]",
                  )}
                  title={option.label}
                >
                  <div className="flex min-w-0 flex-1 items-start gap-2.5">
                    <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-1)]">
                      {(() => {
                        const status = getStatusPresentation(option);
                        const StatusIcon = status.icon;
                        return (
                          <StatusIcon
                            className={cn(
                              "h-4 w-4",
                              status.className,
                              option.statusLabel.toLowerCase().includes("loading") &&
                                "animate-spin",
                            )}
                          />
                        );
                      })()}
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="text-sm font-medium leading-5 text-[var(--text-primary)] break-words">
                        {option.label}
                      </div>
                      <div className="mt-1 text-xs text-[var(--text-muted)]">
                        {option.statusLabel}
                      </div>
                    </div>
                  </div>
                  {selectedOption?.value === option.value ? (
                    <div className="flex shrink-0 items-center gap-2 pl-2">
                      <StatusBadge
                        tone={getStatusTone(option)}
                        className="px-2 py-0.5 text-[9px] tracking-[0.14em]"
                      >
                        Current
                      </StatusBadge>
                      <Check className="h-3.5 w-3.5 shrink-0 text-[var(--text-primary)]" />
                    </div>
                  ) : null}
                </button>
              ))}
            </div>
          </motion.div>
        ) : null}
      </AnimatePresence>
    </div>
  );
}
