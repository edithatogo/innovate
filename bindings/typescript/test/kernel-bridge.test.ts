import { readFileSync } from "node:fs";
import { join } from "node:path";

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

  it("normalizes diagnostics artifact fixture payloads", () => {
    const fixture = JSON.parse(
      readFileSync(
        join(process.cwd(), "..", "..", "tests", "fixtures", "diagnostics_artifact_payload.json"),
        "utf8",
      ),
    );

    const diagnostics = kernelExtractDiagnostics(fixture.result);

    expect(diagnostics?.artifacts).toMatchObject({
      schema_version: "1.0",
      model_name: "BassModel",
      artifacts: {
        residuals: {
          kind: "residual_diagnostics",
          columns: ["index", "residual", "standardized_residual"],
        },
        model_comparison: {
          kind: "model_comparison",
          columns: ["metric", "value"],
        },
      },
    });
  });
});
