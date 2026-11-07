FROM rust:1.99.1-slim AS rust-toolchain
FROM python:3.15.2-slim AS build-source
COPY --from=rust-toolchain /usr/local/cargo /usr/local/cargo
COPY --from=rust-toolchain /usr/local/rustup /usr/local/rustup
ENV CARGO_HOME=/usr/local/cargo RUSTUP_HOME=/usr/local/rustup
ENV PATH=/usr/local/cargo/bin:$PATH
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends build-essential pkg-config && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml uv.lock Cargo.toml Cargo.lock README.md LICENSE ./
COPY rust/ rust/
COPY bindings/python/ bindings/python/
ENV CARGO_BUILD_JOBS=2
FROM build-source AS remote-worker-build
RUN cargo build --locked --release -p innovate-core --bin innovate-remote-worker

FROM python:3.15.2-slim AS remote-worker
COPY --from=remote-worker-build /app/target/release/innovate-remote-worker /usr/local/bin/
USER 65532:65532
ENTRYPOINT ["/usr/local/bin/innovate-remote-worker"]

FROM build-source AS build
RUN uv venv --python /usr/local/bin/python .venv && uv sync --locked --no-dev --no-editable
RUN .venv/bin/python -c 'import sys, innovate; assert sys.version_info[:3] == (3,15,2); print(innovate.capabilities())'

FROM python:3.15.2-slim
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH
CMD ["/bin/bash"]
