import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface CursorPaginationControlsProps {
  page: number;
  canPrevious: boolean;
  canNext: boolean;
  loading?: boolean;
  className?: string;
  onPrevious: () => void;
  onNext: () => void;
}

export function CursorPaginationControls({
  page,
  canPrevious,
  canNext,
  loading = false,
  className,
  onPrevious,
  onNext,
}: CursorPaginationControlsProps) {
  return (
    <div
      className={cn(
        "flex items-center justify-between gap-3 border-t border-[var(--border-muted)] px-4 py-3 sm:px-5",
        className,
      )}
    >
      <p className="text-xs font-medium text-[var(--text-muted)]">Page {page}</p>
      <div className="flex items-center gap-2">
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={onPrevious}
          disabled={loading || !canPrevious}
          aria-label="Previous page"
        >
          Previous
        </Button>
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={onNext}
          disabled={loading || !canNext}
          aria-label="Next page"
        >
          Next
        </Button>
      </div>
    </div>
  );
}
