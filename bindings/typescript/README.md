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

## Backend expectations

- Node.js 22 or newer is expected for the TypeScript test harness.
- The TypeScript package expects the repository checkout to contain the Python `src/` tree.
- The default Python launcher is `uv`; set `INNOVATE_PYTHON_COMMAND` only if you need an override.

The TypeScript package remains a thin adapter over the shared kernel and does not duplicate model
logic.
