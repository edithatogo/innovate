# Specification

Innovate owns diffusion/adoption/incentive models, inference and diagnostics.
Kairos owns the Rust event/state/RNG/resource engine only where a model needs
simulation. Closed-form forecasts need not depend on a scheduler. Specific games
remain domain-owned; generic Rust game representation/solvers are reused from
the existing uogto-game-core/solve incubator above Kairos. UOGTO supplies semantic
profiles, not numerical solver correctness. The game-to-Kairos adapter is optional.

The current source baseline is b9998addf32717438df92b23a306903e4db51620. Seven Rust
Git dependencies are declared at fae901558f07b7b717a676adbafbe2cdc78dea1c. Current
examples print intended imports and do not execute them; metadata/source checks
and Python reference execution cannot qualify native Kairos behaviour. Fix that
claim boundary before promoting runtime capability. Kairos engine APIs must fail
explicitly without the Rust path; no supported host-core fallback.

Outputs are an actual minimal native consumer, thin adapter, replay/error tests,
optional data feature closure and honest capability reporting. Adoption artifacts
retain paired strategy/draw and unit/time/provenance identity when passed to
HEOR or Voiage. HEOR valuations remain domain/provider assumptions; Voiage owns
VOI. Existing HEOML extension drafts are interchange contracts, not a required
simulation or economic engine.

Owned work: the bounded plan and selected adapter/example/tests. Shared pins,
manifest feature edits and public semantic contracts are specialist integration
leaves. No pin or runtime change is made by this registration. Existing scientific
alignment and PR570 retain ownership of broader Rust-core/scientific changes.

The runtime consumer receipt is independent of the optional game profile and
solver qualification. I10 retains every native, replay, installed-consumer and
paired-outcome prerequisite while I07 remains separately pending. A static
game can be qualified against its solver/profile without dynamic execution;
claims about simulated games additionally require native adapter evidence.
Neither receipt completes the other scope or closes unfinished work in this
track or the [PR570 game programme](https://github.com/edithatogo/innovate/pull/570).

Routine leaf evidence is a diff and focused result, with one review at a coherent
public/numerical/replay milestone. No per-leaf approval board or duplicate dossier.
