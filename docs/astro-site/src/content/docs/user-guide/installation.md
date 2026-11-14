---
title: Install the native development package
description: Build the current Rust and Polars development API using the repository's locked tools.
---

Innovate 0.7 is a development line. Use a checkout to reproduce the current
documentation. A registry's latest published package can be an earlier API;
availability of a current registry artifact is a separate release gate.

## Required tools

The repository currently requires Python **3.15.2** and Rust/Cargo **1.99.1**.
Its locked Python runtime dependency is Polars **2.1.0**. Inspect
`pyproject.toml`, `Cargo.toml` and the lockfiles when changing revisions.
These requirements come from the checkout, not from the documentation theme.

## Build from a checkout

Run these commands in a fresh checkout. If an environment already belongs to an
active operation, finish that operation before changing its dependencies.

```sh
git clone https://github.com/edithatogo/innovate.git
cd innovate
git rev-parse HEAD
rustup toolchain install 1.99.1 --profile minimal
rustup override set 1.99.1
uv venv .venv-native --python 3.15.2
UV_PROJECT_ENVIRONMENT=.venv-native uv sync --locked --python 3.15.2
.venv-native/bin/python -c 'import sys, innovate; print(sys.version); print(innovate.__version__)'
.venv-native/bin/python -I examples/native_forecast.py
```

Retain the printed Git revision with results. The final command executes the
[quickstart](/innovate/user-guide/getting-started/) and checks its expected
numerical result. On Windows, use the environment's `Scripts/python.exe` and
set `UV_PROJECT_ENVIRONMENT` using your shell's environment syntax.

## Troubleshooting

A missing `innovate._native` extension means that the selected interpreter
does not have the built native package. Confirm the interpreter path, the Rust
toolchain and successful completion of `uv sync`. Installing only Python source
cannot provide the scientific runtime.

Compiler or package-resolution failures remain failures. Keep the locked
versions while diagnosing them, and retain the command and error log. Do not
substitute an earlier Python numerical backend to obtain a passing example.

## Other bindings and release scope

Python, Julia and R are the supported binding direction. Julia and R call the
Rust C ABI; the [native migration guide](/innovate/native-migration/#r-and-julia)
describes their current build and library requirements. Installed artifacts and
platform support are qualified by their own tests.

The scientific migration ledger remains open. A successful local installation
establishes that this checkout built in this environment, not full model parity
or a qualified release.
