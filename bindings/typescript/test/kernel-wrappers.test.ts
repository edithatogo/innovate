import { describe, expect, it } from "vitest";

import {
  KernelBridgeError,
  kernelDiagnoseModel,
  kernelDiscoverModels,
  kernelExtractDiagnostics,
  kernelFitModel,
  kernelPredictModel,
  kernelRequest,
  kernelSimulateModel,
  kernelSummarizeModel,
} from "../src";

describe("TypeScript kernel wrappers", () => {
  const time = [0, 1, 2, 3, 4];
  const observed = [0.02, 0.06, 0.12, 0.25, 0.41];

  it("discovers the stable model registry through the bridge", () => {
    const discovery = kernelDiscoverModels();

    expect(Array.isArray(discovery)).toBe(true);
    const bass = discovery.find((record) => record.key === "bass");
    expect(bass).toMatchObject({
      key: "bass",
      family: "diffusion",
    });
  });

  it("fits a stable model and returns structured predictions and diagnostics", () => {
    const discovery = kernelDiscoverModels();
    const bass = discovery.find((record) => record.key === "bass");
    if (!bass) {
      throw new Error("Expected bass to be discoverable");
    }

    const fit = kernelFitModel(
      kernelRequest("fit_model", bass.key, {
        inputs: { time, observed },
        model_kwargs: {},
      }),
    );

    expect(fit).toMatchObject({
      model_key: "bass",
      family: "diffusion",
    });
    expect(Array.isArray(fit.predictions)).toBe(true);
    expect(fit.diagnostics).toMatchObject({
      support_level: "supported",
    });

    expect(kernelExtractDiagnostics(fit)).toMatchObject({
      support_level: "supported",
    });
  });

  it("predicts, simulates, summarizes, and diagnoses from a fitted state", { timeout: 120000 }, () => {
    const discovery = kernelDiscoverModels();
    const bass = discovery.find((record) => record.key === "bass");
    if (!bass) {
      throw new Error("Expected bass to be discoverable");
    }
    const fit = kernelFitModel(
      kernelRequest("fit_model", bass.key, {
        inputs: { time, observed },
        model_kwargs: {},
      }),
    );

    const state = fit.state;
    const predict = kernelPredictModel(
      kernelRequest("predict_model", bass.key, {
        inputs: { time },
        state,
      }),
    );
    const simulate = kernelSimulateModel(
      kernelRequest("simulate_model", bass.key, {
        inputs: { time },
        state,
      }),
    );
    const summarize = kernelSummarizeModel(
      kernelRequest("summarize_model", bass.key, {
        inputs: { time, observed },
        state,
      }),
    );
    const diagnose = kernelDiagnoseModel(
      kernelRequest("diagnose_model", bass.key, {
        inputs: { time, observed },
        state,
      }),
    );

    expect(Array.isArray(predict)).toBe(true);
    expect(Array.isArray(simulate)).toBe(true);
    expect(summarize).toMatchObject({
      model_key: "bass",
      family: "diffusion",
    });
    expect(summarize.diagnostics).toMatchObject({
      support_level: "supported",
    });
    expect(diagnose).toMatchObject({
      state: expect.any(Object),
      diagnostics: {
        support_level: "supported",
      },
    });
  });

  it("maps kernel failures to a typed bridge error", () => {
    const discovery = kernelDiscoverModels();
    const bass = discovery.find((record) => record.key === "bass");
    if (!bass) {
      throw new Error("Expected bass to be discoverable");
    }

    expect(() =>
      kernelFitModel(
        kernelRequest("fit_model", bass.key, {
          inputs: { time },
          model_kwargs: {},
        }),
      ),
    ).toThrowError(KernelBridgeError);
  });
});
