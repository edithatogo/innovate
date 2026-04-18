import { describe, expect, it } from "vitest";

import { KernelBridgeError, kernelExtractDiagnostics } from "../src";
import { kernelPythonCommand } from "../src/kernel";

describe("TypeScript kernel bridge helpers", () => {
  it("defaults to the uv python launcher", () => {
    expect(kernelPythonCommand()).toBe("uv run python");
  });

  it("normalizes bridge errors with fallback metadata", () => {
    const error = KernelBridgeError.fromResponse(
      {
        operation: "fit_model",
        error: {
          code: "invalid_request",
          message: "bad request",
          details: null,
        },
      },
      "predict_model",
    );

    expect(error.code).toBe("invalid_request");
    expect(error.message).toBe("bad request");
    expect(error.operation).toBe("predict_model");
    expect(error.retryable).toBe(false);
    expect(error.details).toEqual({});
  });

  it("returns null when diagnostics are absent", () => {
    expect(kernelExtractDiagnostics(null)).toBeNull();
    expect(kernelExtractDiagnostics({})).toBeNull();
  });
});
