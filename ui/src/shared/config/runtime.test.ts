import { describe, expect, it } from "vitest";
import { resolveApiBaseUrl } from "./runtime";

describe("runtime config", () => {
  it("prefers the desktop window server override when present", () => {
    expect(
      resolveApiBaseUrl({
        envBaseUrl: "https://api.example.com/v1",
        windowServerUrl: "http://127.0.0.1:8080/",
      }),
    ).toBe("http://127.0.0.1:8080/v1");
  });

  it("uses the explicit Vite api base url when no desktop override exists", () => {
    expect(
      resolveApiBaseUrl({
        envBaseUrl: "https://api.example.com/v1/",
      }),
    ).toBe("https://api.example.com/v1");
  });

  it("falls back to the default browser-relative api path", () => {
    expect(resolveApiBaseUrl()).toBe("/v1");
  });
});
