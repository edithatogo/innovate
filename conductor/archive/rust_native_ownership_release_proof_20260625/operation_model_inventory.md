# Operation & Model-Family Matrix Inventory

> Generated for **Phase 1, Task 1** of the Rust-Native Ownership Release Proof track.
> Sources: Python `innovate/src/innovate/`, Rust `bindings/rust/src/lib.rs`,
> Rust `bindings/rust/inst/discovery_manifest.json`, Rust `bindings/rust/tests/operations.rs`.

---

## 1. Canonical Kernel Operations

The kernel defines exactly **6 canonical operations**:

| # | Operation | Python | Rust | Bridge fallback? |
|---|-----------|--------|------|------------------|
| 1 | `discover_models` | ✅ | ✅ | No (native only) |
| 2 | `fit_model` | ✅ | ✅ | Yes, for non-native models |
| 3 | `predict_model` | ✅ | ✅ | Yes, for non-native models |
| 4 | `simulate_model` | ✅ | ✅ | Yes, for non-native models |
| 5 | `summarize_model` | ✅ | ✅ | Yes, for non-native models |
| 6 | `diagnose_model` | ✅ | ✅ | Yes, for non-native models |

Schema version: **1.0** (consistent across Python and Rust).

---

## 2. Model Families & Inventory

**7 model families** covering **15 model keys** (plus 1 gap).

### Legend
- **Native** = Rust-native implementation in `lib.rs`
- **Bridge** = Falls back to Python bridge (`kernel_bridge.py`)
- **N/A** = Not applicable

### Stable Models (stability: `stable`)

| Model Key | Family | discover | fit | predict | simulate | summarize | diagnose |
|-----------|--------|----------|-----|---------|----------|-----------|----------|
| `bass` | `diffusion` | Native | Native | Native | Native | Native | Native |
| `logistic` | `diffusion` | Native | Native | Native | Native | Native | Native |
| `gompertz` | `diffusion` | Native | Native | Native | Native | Native | Native |
| `fisher_pry` | `substitution` | Native | Native | Native | Native | Native | Native |
| `norton_bass` | `substitution` | Native | Native¹ | Bridge | Bridge | Native | Native |
| `composite` | `substitution` | Native | Bridge | Bridge | Bridge | Bridge | Bridge |
| `multi_product` | `competition` | Native | Bridge | Bridge | Bridge | Bridge | Bridge |
| `lotka_volterra` | `competition` | Native | Bridge | Bridge | Bridge | Bridge | Bridge |
| `complementary_goods` | `ecosystem` | Native | Bridge | Bridge | Bridge | Bridge | Bridge |

¹ Norton-Bass native fit limited to **single-generation** payloads; multi-generation falls back to bridge.

### Provisional / Experimental Models

| Model Key | Family | discover | fit | predict | simulate | summarize | diagnose |
|-----------|--------|----------|-----|---------|----------|-----------|----------|
| `hierarchical` | `advanced_diffusion` | Native | Bridge | Bridge | Bridge | Bridge | Bridge |
| `mixture` | `advanced_diffusion` | Native | Bridge | Bridge | Bridge | Bridge | Bridge |
| `latent_process` | `advanced_diffusion` | Native | Bridge | Bridge | Bridge | Bridge | Bridge |
| `regime_switching` | `advanced_diffusion` | Native | Bridge | Bridge | Bridge | Bridge | Bridge |
| `network_diffusion` | `network_diffusion` | Native | Bridge | Bridge | Bridge | Bridge | Bridge |
| `policy_hazard` | `policy_diffusion` | Native | Bridge | Bridge | Bridge | Bridge | Bridge |
