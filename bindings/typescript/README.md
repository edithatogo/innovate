# innovate.ts TypeScript bindings

This package provides a thin TypeScript-facing adapter over the Python `innovate` functional kernel.
Promoted native slices execute in the shared kernel contract, unsupported promoted payloads
return explicit native errors, and bridge fallback remains available only for explicitly non-native
model families.

## Installation

Install the published package from npm:

```bash
npm install innovate.ts
```

## Invocation path

The package shells out to the shared kernel bridge shipped at `inst/python/kernel_bridge.py` and
keeps the runtime surface thin. The bridge is intentionally Python-backed for explicitly
non-native model families so the TypeScript layer does not reimplement model logic.

## Current scaffold

- Package metadata in `package.json`
- TypeScript compiler settings in `tsconfig.json`
- Published entrypoint and declarations in `dist/`
- Kernel contract helpers in `src/kernel.ts`
- Public entrypoint re-exports in `src/index.ts`
- Contract and scaffold tests under `test/`
- A runnable end-to-end diagnostics workflow example in `examples/diagnostics-workflow.ts`

## Example workflow

Run the stable-kernel diagnostics example from the package root:

```bash
cd bindings/typescript
npm install
npx tsx examples/diagnostics-workflow.ts
```

The example discovers a stable model, fits it on a small observed time series, then summarizes and
diagnoses the fitted state through the TypeScript wrapper layer. The package stays thin and does
not duplicate model semantics.

## Package scripts

- `npm test` runs the TypeScript test suite.
- Bridge-backed Vitest checks run serially with a 120 second timeout because
  they launch the shared Python kernel bridge.
- `npm run build` compiles the published JavaScript and declaration files.
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
import { kernelDiscoverModels, kernelFitModel, kernelRequest } from "innovate.ts";

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

- Node.js 26 or newer is required.
- The Python `innovate` package must be available to the bridge command.
- The default Python launcher is `uv run python`; set `INNOVATE_PYTHON_COMMAND` when the backend is
  installed in another environment.
- Repository checkouts are still supported. Set `INNOVATE_REPO_ROOT` to force discovery of a local
  checkout and prepend its Python `src/` tree to the bridge process.
- Native promoted slices do not silently fall back to Python; only explicitly non-native
  families use the bridge.

The TypeScript package remains a thin adapter over the shared kernel and does not duplicate model
logic.
