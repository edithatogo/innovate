# Recurring scenario Arrow and minimum-Rust integration candidate

This is the concrete shared-CI handoff for ALIGN-02f/ALIGN-04c and issue #556.
The [complete candidate workflow](scenario-arrow-msrv.yml) is stored here so the
existing integrator in #624 can review and adopt it at
`.github/workflows/scenario-arrow-msrv.yml`. It is **not an active workflow** in
this branch. Existing required checks, scientific gates and other owners' work
retain their requirements.

## Implemented and executed

The [build executor](build_scenario_wheel.py) fills the previous manual step:
it runs the actual locked Maturin build, records its command/toolchain and all
tracked native/build inputs, and binds the output wheel to the retained log.
It rejects dirty native inputs, changed inputs during execution, failed builds,
missing or ambiguous wheels, mixed provider revisions and reused output paths.
It emits a successful build record only after those checks pass. This is an
executor observation, not a signed supply-chain attestation.

The existing `scripts/qualify_scenario_arrow.py` then validates that record,
wheel RECORD and imported native bytes and executes the fixed 42 installed
cases outside the checkout. Python/Polars/PyArrow/pytest pins are unchanged.
The workflow installs the optional PyArrow dependency explicitly; no optional
test skip is accepted by the qualifier.

The [new execution packet](../evidence/evidence-acquisition-20261116/README.md)
records a fresh build from all **178** native inputs of current main
`b29a1d27f54fdd08d77881ee999772fceab7c03a`, 42 installed passes including all four
Arrow cases, and a separate actual Rust 1.99.1 compile check. Twenty-four tooling
tests include real process failure and independent negative evidence cases;
the retained coverage report measures 85/87 statements and 19/20 branches.

## Candidate jobs and bounds

| Job | Executed contract | Deliberate scope |
| --- | --- | --- |
| Installed scenario Arrow | Python 3.15.2, Rust 1.100.0, locked Maturin 1.16.0 build; fresh environment with wheel, Polars 2.1.0, PyArrow 26.0.1 and pytest 9.2.0; exact 42-case qualifier; retain wheel and logs | Linux development artifact; no release build or wider scientific acceptance |
| Rust 1.99.1 workspace compile | `cargo check --locked --workspace --all-targets -j 2`; retain exact source, lock hashes, actual compiler and output | Linux/default features, five workspace crates; no excluded games crate, execution, linking or other platform claim |

The local commands use one build job to limit this workspace's memory use. The
candidate uses two jobs on hosted Linux; parallelism is the only intended
resource-setting difference. All actions reuse immutable revisions already
present in the repository. Permissions are read-only, checkout credentials are
not persisted, and evidence is retained on both success and failure. The YAML
was parsed locally and [independently reviewed](../evidence/evidence-acquisition-20261116/independent-review.md).
Its hosted jobs have not run.

## Integrator adoption and documentation sequence

1. Receive the source/evidence in draft #662 at its published final head and
   verify the exact receiving revision. The earlier d36c33f hosted results are
   historical receipts, not results for a later commit.
2. Adopt the candidate workflow through the existing shared-CI owner, then
   execute both jobs on the actual receiving head. No new required-check rule
   should be inferred merely from copying the file.
3. Reconcile the documentation owner's active draft #671. Closed #659 is
   superseded. #668 is already in main and its Rustdoc repair passes; main's
   remaining browser failure and #671's passing replacement are documented in
   the [hosted intake](../../research_software_registry_readiness_20260721/evidence/evidence-acquisition-20261116/hosted/README.md).
4. Whichever PR is received first, merging #671 changes **two** manuscript
   source fingerprints: the native migration guide and tabular `src/lib.rs`.
   Review their effect on the paper, deliberately update its assessed source,
   rerun the integrity check and render, and rebuild/requalify changed native
   inputs. The retained drift diagnostic demonstrates that the old snapshot
   correctly fails on the incoming guide; automatic rehashing is not acceptance.

The integrator retains promotion and the shared manifests, pins, workflow and
registry. Offline-security runner qualification, the 38 scientific obligations,
provider distribution and exact release artifacts remain separate work.
