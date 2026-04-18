import { pathToFileURL } from "node:url";

import {
  kernelDiagnoseModel,
  kernelDiscoverModels,
  kernelExtractDiagnostics,
  kernelFitModel,
  kernelPredictModel,
  kernelSimulateModel,
  kernelRequest,
  kernelSummarizeModel,
  type KernelDiagnoseResponse,
  type KernelFitResponse,
  type KernelJSONValue,
  type KernelSummaryResponse,
} from "../src";

const time = [0, 1, 2, 3, 4];
const observed = [0.02, 0.06, 0.12, 0.25, 0.41];

export interface DiagnosticsWorkflowExample {
  modelKey: string;
  family: string;
  discoveryCount: number;
  predictionCount: number;
  simulationCount: number;
  fit: KernelFitResponse;
  summary: KernelSummaryResponse;
  diagnose: KernelDiagnoseResponse;
  fitDiagnostics: Record<string, KernelJSONValue> | null;
  summaryDiagnostics: Record<string, KernelJSONValue> | null;
}

export function runDiagnosticsWorkflow(): DiagnosticsWorkflowExample {
  const discovery = kernelDiscoverModels();
  const bass = discovery.find((record) => record.key === "bass");
  if (!bass) {
    throw new Error("Expected bass to be discoverable for the diagnostics example");
  }

  const fit = kernelFitModel(
    kernelRequest("fit_model", bass.key, {
      inputs: { time, observed },
      model_kwargs: {},
    }),
  );

  const summary = kernelSummarizeModel(
    kernelRequest("summarize_model", bass.key, {
      inputs: { time, observed },
      state: fit.state,
    }),
  );

  const diagnose = kernelDiagnoseModel(
    kernelRequest("diagnose_model", bass.key, {
      inputs: { time, observed },
      state: fit.state,
    }),
  );

  const predict = kernelPredictModel(
    kernelRequest("predict_model", bass.key, {
      inputs: { time },
      state: fit.state,
    }),
  );
  const simulate = kernelSimulateModel(
    kernelRequest("simulate_model", bass.key, {
      inputs: { time },
      state: fit.state,
    }),
  );

  return {
    modelKey: bass.key,
    family: bass.family,
    discoveryCount: discovery.length,
    predictionCount: Array.isArray(predict) ? predict.length : 0,
    simulationCount: Array.isArray(simulate) ? simulate.length : 0,
    fit,
    summary,
    diagnose,
    fitDiagnostics: kernelExtractDiagnostics(fit),
    summaryDiagnostics: kernelExtractDiagnostics(summary),
  };
}

export function formatDiagnosticsWorkflow(result: DiagnosticsWorkflowExample): string {
  return [
    `Model: ${result.modelKey} (${result.family})`,
    `Models discovered: ${result.discoveryCount}`,
    `Prediction count: ${result.predictionCount}`,
    `Simulation count: ${result.simulationCount}`,
    `Fit support: ${String(result.fitDiagnostics?.support_level ?? "unknown")}`,
    `Summary support: ${String(result.summaryDiagnostics?.support_level ?? "unknown")}`,
    `Diagnose support: ${String(result.diagnose.diagnostics.support_level ?? "unknown")}`,
  ].join("\n");
}

function main(): void {
  console.log(formatDiagnosticsWorkflow(runDiagnosticsWorkflow()));
}

const entrypoint = process.argv[1];
if (entrypoint && import.meta.url === pathToFileURL(entrypoint).href) {
  main();
}
