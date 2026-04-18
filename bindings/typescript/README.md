# innovate TypeScript bindings

This package provides a thin TypeScript-facing adapter over the Python `innovate` functional kernel.

## Installation

From the repository root:

```bash
cd bindings/typescript
npm install
```

## Invocation path

The package shells out to the shared kernel bridge at `inst/python/kernel_bridge.py` and keeps the
runtime surface thin. The bridge is intentionally Python-backed so the TypeScript layer does not
reimplement model logic.

## Current scaffold

- Package metadata in `package.json`
- TypeScript compiler settings in `tsconfig.json`
- Kernel contract helpers in `src/kernel.ts`
- Public entrypoint re-exports in `src/index.ts`
- Contract and scaffold tests under `test/`
- A runnable end-to-end diagnostics workflow example in `examples/diagnostics-workflow.ts`

## Example workflow

Run the stable-kernel diagnostics example from the package root:

```bash
npx tsx examples/diagnostics-workflow.ts
```

The example discovers a stable model, fits it on a small observed time series, then summarizes and
diagnoses the fitted state through the TypeScript wrapper layer. The package stays thin and does
not duplicate model semantics.

## Package scripts

- `npm test` runs the TypeScript test suite.
- `npm run coverage` runs the suite with V8 coverage enabled.
- `npm run typecheck` validates the package with `tsc --noEmit`.
- `npm run schema:check` verifies the TypeScript kernel schema version matches the Python kernel.

## Public surface

The package exports:

- `kernelSchemaVersion()` and `KERNEL_SCHEMA_VERSION`
- request helpers from `src/kernel.ts`
- stable wrapper helpers for discovery, fit, predict, simulate, summarize, and diagnose
- `KernelBridgeError` for typed bridge failures

The public API is intentionally narrow so the TypeScript layer stays a thin adapter over the shared
kernel implementation.

## Usage

```ts
import { kernelDiscoverModels, kernelFitModel, kernelRequest } from "innovate-typescript-bindings";

const discovery = kernelDiscoverModels();
const bass = discovery.find((record) => record.key === "bass");

if (!bass) {
  throw new Error("Bass model is not available");
}

const fit = kernelFitModel(
  kernelRequest("fit_model", bass.key, {
    inputs: { time: [0, 1, 2, 3], observed: [0.02, 0.06, 0.12, 0.25] },
    model_kwargs: {},
  }),
);
```

The wrapper layer converts the kernel bridge responses into idiomatic TypeScript objects while
preserving the underlying kernel contract.

## Backend expectations

- Node.js 22 or newer is expected for the TypeScript test harness.
- The TypeScript package expects the repository checkout to contain the Python `src/` tree.
- The default Python launcher is `uv`; set `INNOVATE_PYTHON_COMMAND` only if you need an override.

The TypeScript package remains a thin adapter over the shared kernel and does not duplicate model
logic.
