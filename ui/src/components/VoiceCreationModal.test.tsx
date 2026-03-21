import { fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { VoiceCreationModal } from "./VoiceCreationModal";

function VoiceCreationModalHarness() {
  const [open, setOpen] = useState(true);
  return (
    <VoiceCreationModal open={open} onOpenChange={setOpen} />
  );
}

describe("VoiceCreationModal", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("moves from flow choice into clone and supports back navigation", () => {
    render(<VoiceCreationModalHarness />);

    fireEvent.click(screen.getByRole("button", { name: /Clone Voice/i }));

    expect(screen.getByRole("dialog", { name: "Clone Voice" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Back" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Back" }));

    expect(screen.getByRole("dialog", { name: "New Voice" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Design Voice/i })).toBeInTheDocument();
  });

  it("protects in-progress draft when closing from clone step", () => {
    const confirmMock = vi.spyOn(window, "confirm").mockReturnValue(false);
    render(<VoiceCreationModalHarness />);

    fireEvent.click(screen.getByRole("button", { name: /Clone Voice/i }));
    fireEvent.click(screen.getByRole("button", { name: "Close" }));

    expect(confirmMock).toHaveBeenCalledTimes(1);
    expect(screen.getByRole("dialog", { name: "Clone Voice" })).toBeInTheDocument();
  });
});
