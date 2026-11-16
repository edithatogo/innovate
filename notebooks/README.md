# Native marimo learning notebooks

This optional environment keeps notebook tooling outside Innovate's runtime
dependencies. Run commands from the repository root with the supported Python
and Rust toolchains described by the installation guide:

```sh
uv sync --project notebooks --locked
uv run --project notebooks marimo edit notebooks/public_data_fit.py
```

`public_data_fit.py` calls the same `run_comparison` function as
`examples/native_public_data.py`. The default cutoff is 2015; the slider changes
only the last training year. Results are recreated without mutating a fitted
envelope or cached shared data. The frozen public dataset is read locally.

The notebook needs native Python and the installed Rust extension. Browser
WebAssembly/Pyodide execution has not been qualified for this package. A
server-rendered or static export is a separate mode; a static marimo export
needs its JavaScript renderer and is not an interactive Python kernel.

The ordinary Astro tutorial remains readable without running a notebook. To
check the default and changed-control computation and export the current app:

```sh
uv run --project notebooks python scripts/check_learning_notebooks.py --execute
```

The checker rejects an export process failure even if an HTML file was written.
It compares actual notebook results with the ordinary example, records source
and environment identities, and stores the export under
`docs/astro-site/.reports/notebooks/`. These outputs are local run evidence, not
automatically published pages. Any later source edit requires a fresh run.
