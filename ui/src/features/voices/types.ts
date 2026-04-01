import type { ReactNode } from "react";

export interface VoiceLibraryItem {
  id: string;
  name: string;
  secondaryLabel?: string;
  categoryLabel: string;
  description?: string;
  previewUrl?: string | null;
  previewMessage?: string | null;
  previewLoading?: boolean;
  selected?: boolean;
  onSelect?: () => void;
  actions?: ReactNode;
}
