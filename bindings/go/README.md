# innovate Go bindings

This package provides a thin Go-facing adapter over the Python `innovate`
functional kernel.

## Installation

From the repository root:

```bash
cd bindings/go
go test ./...
```

The Go module is intended for direct repository use. It is not published as a
standalone registry package.

## Architecture

- `bindings/go` is the Go module root.
- The package shells out to `bindings/go/inst/python/kernel_bridge.py` to keep
  the Go layer contract-driven and free of model logic.
- Kernel request and response envelopes mirror the Python kernel contract so Go
  can exchange structured data without reimplementing diffusion semantics.

## Planned mapping rules

- Kernel schema version becomes a package-level constant and helper.
- Kernel request envelopes map to exported Go structs with explicit JSON tags.
- Kernel responses map to exported Go structs with nested error and diagnostics
  payloads.
- Kernel operation names remain stable strings shared with the Python kernel.

## Expected surface

- Schema/version helpers
- Bridge path helpers
- Request and response envelope types
- Stable operation wrappers for discovery, fit, predict, simulate, summarize,
  and diagnose

## Compatibility checks

- `KernelSchemaVersion()` exposes the shared kernel schema version.
- `KernelOperations()` lists the stable operations the Go layer supports.
- The test suite checks that the bridge discovery response and the exported
  schema version stay aligned.
- The test suite also verifies the supported operation list and exercises an
  end-to-end example.

## Runtime expectations

- The Go package remains a thin wrapper.
- The Python source tree must be available in the repository checkout.
- The default bridge launcher should be `uv run python` unless explicitly
  overridden by the environment.
- `INNOVATE_PYTHON_COMMAND` can override the Python launcher when `uv` is not
  available.

## Support boundaries

- The package only wraps the stable kernel contract.
- It does not reimplement diffusion semantics in Go.
- It is designed for local development and in-repo testing, not external
  registry distribution.
