import { describe, expect, it } from "vitest";

import {
  kernelBindingsRoot,
  kernelBridgeScript,
  kernelRequest,
  kernelSchemaVersion,
} from "../src/kernel";

describe("TypeScript kernel architecture", () => {
  it("exports the shared kernel schema version", () => {
    expect(kernelSchemaVersion()).toBe("1.0");
  });

  it("resolves the bridge script inside the TypeScript bindings tree", () => {
    expect(kernelBridgeScript()).toContain("bindings/typescript/inst/python/kernel_bridge.py");
  });

  it("builds discover-models requests with the shared schema envelope", () => {
    expect(kernelRequest("discover_models")).toEqual({
      schema_version: "1.0",
      operation: "discover_models",
      model_key: null,
      payload: {},
      metadata: {},
    });
  });

  it("anchors the bindings package under bindings/typescript", () => {
    expect(kernelBindingsRoot()).toContain("bindings/typescript");
  });
});
