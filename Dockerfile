FROM rust:1.99.1-slim@sha256:f6c4a35d21dbb3c89d7a8d056c193e94fc09037f2b58dd858b09c39a47dd02d7 AS rust-toolchain
FROM python:3.15.2-slim@sha256:65a65fc7b30b06cb6df4930fd4650697cb23f4c2698ab81fc87e4b031bc36928 AS build-source
COPY --from=rust-toolchain /usr/local/cargo /usr/local/cargo
COPY --from=rust-toolchain /usr/local/rustup /usr/local/rustup
ENV CARGO_HOME=/usr/local/cargo RUSTUP_HOME=/usr/local/rustup
ENV PATH=/usr/local/cargo/bin:$PATH
COPY --from=ghcr.io/astral-sh/uv:latest@sha256:5cc1805e2426484abca4ad46ea783bab281fdb225fd82d8ce6d66a81254bac85 /uv /bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends build-essential pkg-config && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml uv.lock Cargo.toml Cargo.lock README.md LICENSE ./
COPY rust/ rust/
COPY bindings/python/ bindings/python/
ENV CARGO_BUILD_JOBS=2
FROM rust:1.99.1-slim-trixie@sha256:f6c4a35d21dbb3c89d7a8d056c193e94fc09037f2b58dd858b09c39a47dd02d7 AS remote-worker-build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential pkg-config && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY Cargo.toml Cargo.lock LICENSE ./
COPY rust/ rust/
ENV CARGO_BUILD_JOBS=2
RUN cargo build --locked --release -p innovate-core --bin innovate-remote-worker

FROM debian:trixie-slim@sha256:a72b4433de1cf7d6bae0c4cee74d0547f58601a643b3375f070c51600d63b6b2 AS remote-worker
COPY --from=remote-worker-build /app/LICENSE /usr/local/share/doc/innovate/LICENSE
COPY --from=remote-worker-build /app/target/release/innovate-remote-worker /usr/local/bin/
USER 65532:65532
ENTRYPOINT ["/usr/local/bin/innovate-remote-worker"]

FROM build-source AS build
RUN uv venv --python /usr/local/bin/python .venv && uv sync --locked --no-dev --no-editable
RUN .venv/bin/python -c 'import sys, innovate; assert sys.version_info[:3] == (3,15,2); print(innovate.capabilities())'

FROM python:3.15.2-slim@sha256:65a65fc7b30b06cb6df4930fd4650697cb23f4c2698ab81fc87e4b031bc36928
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH
CMD ["/bin/bash"]
