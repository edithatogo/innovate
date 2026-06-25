import { describe, expect, it } from "vitest";

import { formatDiagnosticsWorkflow, runDiagnosticsWorkflow } from "../examples/diagnostics-workflow";

describe("Diagnostics workflow example", () => {
  it("runs the end-to-end stable kernel diagnostics example", { timeout: 120000 }, () => {
    const workflow = runDiagnosticsWorkflow();

    expect(workflow.modelKey).toBe("bass");
    expect(workflow.family).toBe("diffusion");
    expect(workflow.discoveryCount).toBeGreaterThan(0);
    expect(workflow.predictionCount).toBeGreaterThan(0);
    expect(workflow.simulationCount).toBeGreaterThan(0);
    expect(workflow.fitDiagnostics).toMatchObject({
      support_level: "supported",
    });
    expect(workflow.summaryDiagnostics).toMatchObject({
      support_level: "supported",
    });
    expect(workflow.diagnose.diagnostics).toMatchObject({
      support_level: "supported",
    });
  });

  it("formats a concise diagnostics report for users", { timeout: 120000 }, () => {
    const report = formatDiagnosticsWorkflow(runDiagnosticsWorkflow());

    expect(report).toContain("Model: bass (diffusion)");
    expect(report).toContain("Fit support: supported");
    expect(report).toContain("Summary support: supported");
    expect(report).toContain("Diagnose support: supported");
    expect(report).toContain("Simulation count:");
  });
});
