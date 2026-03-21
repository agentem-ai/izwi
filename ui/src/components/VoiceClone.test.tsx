import { act, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { VoiceClone } from "./VoiceClone";

vi.mock("../api", () => ({
  api: {
    listSavedVoices: vi.fn().mockResolvedValue([]),
    createSavedVoice: vi.fn(),
    getSavedVoice: vi.fn(),
    savedVoiceAudioUrl: vi.fn(),
  },
}));

describe("VoiceClone", () => {
  it("hides saved-voice tools in capture workflow mode", () => {
    render(
      <VoiceClone
        workflowMode="capture"
        onVoiceCloneReady={vi.fn()}
        onClear={vi.fn()}
      />,
    );

    expect(screen.queryByRole("button", { name: "Saved Voice" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Save New Voice" })).not.toBeInTheDocument();
  });

  it("keeps saved-voice tools in full workflow mode", async () => {
    await act(async () => {
      render(
        <VoiceClone
          onVoiceCloneReady={vi.fn()}
          onClear={vi.fn()}
        />,
      );
    });

    expect(screen.getByRole("button", { name: "Saved Voice" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Save New Voice" })).toBeInTheDocument();
  });
});
