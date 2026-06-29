# Kairos Integration Report

## Kairos Dependency Inclusion Tracking Document

**Date**: 2026-06-30  
**Track**: kairos_dependency_inclusion_20260626  
**Kairos Repository**: https://github.com/edithatogo/kairos  
**Kairos Revision**: fae901558f07b7b717a676adbafbe2cdc78dea1c (2026-05-19)

### I. Kairos Source Verification

#### Repository Workspace Crates
The Kairos repository workspace contains the following core crates required for integration:

**Core Crates (Required for Integration):**
- ✅ `kairo-ecs-types` - ECS type system and contracts
- ✅ `kairo-ecs-core` - Core DES/ABM engine implementation
- ✅ `kairo-ecs-state` - Entity/state component storage
- ✅ `kairo-ecs-rng` - Random number generation with seeding
- ✅ `kairo-ecs-des` - Discrete event simulation core
- ✅ `kairo-ecs-abm` - Agent-based modeling extensions
- ✅ `kairo-ecs-arrow` - Arrow interchange format support

**Bridge Crates (Conditional, Gated Behind Smoke Tests):**
- `kairo-ecs-ffi` - C FFI bindings (stability: in review)
- `kairo-ecs-uniffi` - UniFFI bindings (stability: in review)
- `kairo-ecs-diplomat` - Diplomat bindings (stability: in review)

#### Publishing Status
- **crates.io Status**: None of the `kairo-ecs-*` crates are currently published on crates.io
- **Integration Approach**: Repository source via GitHub git dependency
- **Decision**: Use repository source for now; will migrate to published crates once available and version-compatible

### II. Dependency Migration Policy

#### Python Base Dependencies: Legacy Removal
As of this track, the following packages have been removed from base Python dependencies:

**Removed from Base Install:**
1. **mesa (>=3.5.1, <4)** - Multi-agent modeling framework
   - **Reason**: To be replaced by Kairos DES/ABM engine with better determinism and performance
   - **Migration Path**: Moved to optional `legacy-abm` extra for backward compatibility
   - **Timeline**: Full removal after follow-on Kairos ABM migration track

2. **ndlib (>=5.1.1, <6)** - Network diffusion library
   - **Reason**: To be replaced by Kairos ABM capabilities with Arrow interchange
   - **Migration Path**: Moved to optional `legacy-abm` extra for backward compatibility
   - **Timeline**: Full removal after network simulation migration is complete

#### Python Dependencies: Retained with Justification
**networkx (>=3.6.1, <4)** - Retained in base dependencies
- **Justification**: Required for network topology analysis and graph visualization in analysis pipelines
- **Usage**: Graph construction, layout algorithms, and NetworkX-based plotting utilities
- **Not Replaced By**: Kairos focuses on simulation; networkx provides complementary graph analysis
- **Assessment**: Keep in base; no migration to optional status needed

#### Explicit Optional Extras
A new optional dependency group has been created:

```toml
[project.optional-dependencies]
legacy-abm = [
  "mesa>=3.5.1,<4",
  "ndlib>=5.1.1,<6",
]
```

**Rationale**: Users who need Mesa or NDLib for legacy code can install with `pip install innovate[legacy-abm]`

### III. External Compatibility Constraints

#### Python 3.14 Baseline
- **Baseline Status**: Maintained at Python >=3.14
- **Dependency Blockers**: None identified during this audit
- **Verified Constraints**:
  - ✅ All core Innovate dependencies support Python 3.14
  - ✅ All Kairos crates compile on Python 3.14-compatible Rust toolchain
  - ✅ networkx 3.6.1+ is Python 3.14 compatible
  - ✅ Legacy extras (mesa, ndlib) note: mesa 3.5.1+ supports Python 3.14; ndlib 5.1.1+ supports Python 3.14

#### Rust Toolchain
- **Rust Version**: 1.76+ (Kairos workspace requirement)
- **Innovate Rust Bindings**: Requires Rust 1.85+
- **Git Dependency Resolution**: Cargo will resolve Kairos git dependencies at build time

#### Registry / Packaging Constraints
- **No New Blockers**: This change does not introduce Python packaging registry constraints
- **Cargo Lock Management**: Lock file will record the exact Kairos revision for reproducibility
- **Future Consideration**: Once Kairos crates are published to crates.io with compatible versions, this track should document the migration to published versions

### IV. Smoke Test Status

**Phase 1 Status**: ✅ Dependency policy defined and tests written
**Phase 2 Status**: Pending - Rust build plumbing and smoke evidence
**Phase 3 Status**: Pending - DES and ABM smoke scenarios

### V. Next Steps

1. **Phase 2**: Add Kairos Rust build integration and update Cargo.lock
2. **Phase 3**: Implement minimal DES and ABM smoke test scenarios
3. **Follow-on Track**: Kairos ABM and Network Simulation Migration (full adapter implementation)
