# Innovation capability programme — revised 7 November 2026

## Authority and scope correction

This revision incorporates the maintainer's explicit direction: preserve innovation modelling and prediction as the product, require a Rust core and Polars, cease pandas/xarray support, align with current Kairos, and study hype's effects rather than make claim exaggeration the central construct. It supersedes conflicting choices in the initial PR #570 planning packet. This is an approved implementation direction; implementation, model qualification and release acceptance remain open.

Reviewed Innovate planning base: `40c00f98d929c8109dd5a709ca9fb3ef44b443ab`. Existing `scientific_ecosystem_alignment_20261029` keeps its historical evidence and pin ownership; new requirements override future architecture choices, not historical receipts. Reconcile overlap with open plans before assigning work.

## Stable purpose

Policy/technology diffusion, adoption, competition, substitution, complements, expectations/hype, failure and strategic innovation are in scope. Medicines utilisation is the first substantial multi-mechanism application, not a reason to turn Innovate into a clinical library or a misinformation classifier.

## Architecture decision

Chosen: a binding-independent Rust core, Polars data layer, Arrow interchange and optional native adapters. Rejected: a Python-first kernel with selected Rust speedups; a monolith containing a second scheduler, VOI engine and game solver; and a wholesale rewrite without independent reference tests. Incremental native extraction preserves scientific references while retiring unsupported production paths.

Target components: `crates/innovate-core`, `crates/innovate-data`, `crates/innovate-arrow`, optional `crates/innovate-kairos`, optional `crates/innovate-games`, and direct bindings. These paths are proposed until corresponding tasks pass. Rust compiled to native or Wasm counts as Rust execution only for the capabilities actually qualified.

## Directed ownership

| Capability | Owner | Integration |
|---|---|---|
| Innovation dynamics, observation models, fit/predict and innovation-specific game construction | Innovate | Consumes native Kairos and optional generic game runtime; exports uncertainty-preserving forecasts. |
| Simulation clock/events/state/resources/RNG/replay | Kairos | No reverse dependency on Innovate or applications; no host substitute. |
| Generic game definitions, equilibrium/regret/learning algorithms | One extracted game-runtime library | Reconcile `uogto-game-core`, `uogto-game-solve` and Kairos game crates before promotion; static solvers need no simulator. |
| Semantic game vocabulary/conformance | UOGTO | Semantic dependency, not an equilibrium implementation. |
| VOI, perspective/equity/decision evaluation and generic VOP capability | Voiage | Optional Innovate consumer adapter; application supplies utility/economic mapping. |
| Spline surrogate fitting/replay | MARS | Optional, measured decision-preservation gate. |
| Medicines source acquisition, terminology and reviewed funding evidence | Global Medicines Atlas / Reimbursement Atlas | Innovate consumes pinned, qualified projections; no duplicate data lake. |
| NZ proof of concept | vop_poc_nz | Empirical scenarios/tutorials; reusable methods promoted to Voiage. |
| Tobacco, reimbursement and healthcare operational studies | tobac / new-drug-reimbursement-game / careops-sim | Domain assumptions and validation; generic capabilities move upstream. |

The additional library is a consolidation/extraction of an existing game-runtime incubator, not a new economics umbrella or generic orchestration framework. Its name/repository can be settled at extraction; the existing `uogto-game-*` crate names are the starting point. No new repository is created by this planning change.

## Existing six tracks, revised rather than duplicated

| Track | Taskbook IDs | Acceptance emphasis |
|---|---|---|
| Scientific integrity | I01–I04 | No fabricated posteriors or unimplemented causal methods disguised as results. |
| Rust-core convergence | R01–R10 | Native-only execution; Polars/Arrow; pandas/xarray-free supported closure; qualified pins. |
| Empirical validation/publication | M01–M05, V01–V05, Q01–Q05 | Medicines and cross-domain evidence, independent oracles, packages and hardened release. |
| Advanced diffusion frontier | A01–A08, G01–G04 | Actual innovation capabilities, established methods and falsifiable new research. |
| Ecosystem decision integration | E01–E05 | Actual Kairos execution, joint draws, Voiage/VOP/application promotion and no cycles. |
| Healthcare hype evaluation | H01–H06 | General composite hype and uptake effects; healthcare application. Claim auditing is not a prerequisite. |

The [taskbook](innovation-taskbook-20261107.md) is the authoritative granular work definition. Track plans select IDs and record completion receipts; do not maintain competing checklists. Work starts with integrity and native contracts; source auditing and literature/comparator review can proceed in parallel. No estimated calendar completion is substituted for evidence.

## Frontier and beyond-frontier

Frontier delivery means correct, competitive, reproducible model implementations: heterogeneous early-stage diffusion, multi-generation competition, repeat-use switching, calibrated hype effects and native cross-language operation. Beyond-frontier research targets joint expectations/adoption feedback, strategically endogenous innovation, demand–capacity separation, path dependence and transfer, and learning/implementation decisions. Each must beat or explain failure against named alternatives on held-out data. The [research register](research/innovation-frontier-questions-20261107.md) separates known methods, integration gaps and unresolved hypotheses.

## Dependency alignment

Follow [dependency qualification](dependency-candidates-20261107.md) and repository-standards. Use newest released/prerelease/pinned-development candidates first, but only exact-source qualified candidates can ship. Shared semantic/RNG/Arrow contracts matter more than identical unrelated package version numbers. No rolling main dependencies, unrecorded fallback, blind upgrades or automatic publication.

## Release and research outputs

Mandatory gates: scientific validity for the declared scope; native dispatch proof; supported installs without pandas/xarray; clean cross-platform consumers; reproducible benchmarks; supply-chain/security evidence; migration documentation and honest unsupported capabilities. See [release gates](release-gates-20261107.md).

Initial software paper: JOSS after real research use and declared-scope acceptance. Forecast comparison: International Journal of Forecasting. Innovation/competition/hype contribution: Technological Forecasting and Social Change or Technovation; a medicines application may fit a pharmacoepidemiology journal. Decision-method work: Medical Decision Making. Reusable benchmark: Scientific Data. Venue requirements and AI disclosure are checked at submission; publication is not authorised by a completed software task.

No protected-branch merge, external package publication, manuscript submission, paid compute, private-data transfer or undeclared application migration is implied. One accountable maintainer is supported; automated review must not invent a second human reviewer.
