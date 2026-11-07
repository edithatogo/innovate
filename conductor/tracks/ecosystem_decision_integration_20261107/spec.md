# Ecosystem decision integration specification

Innovate estimates adoption; Kairos simulates operational delivery; an application economic model translates joint trajectories into costs/outcomes; Voiage evaluates decision uncertainty and research value. Do not duplicate a scheduler, generic economic model or VOI estimator inside Innovate.

The first end-to-end fixture separates latent willingness/eligibility, attempted uptake, delivery capacity, observed utilisation and continuing use/abandonment. Test whether adoption and capacity are identifiable from the available measurements. Return sensitivity bounds when they are not.

Preserve joint draws, scenario/strategy identifiers, units, time basis, population denominators, simulation seeds, fit status, intervention assumptions and provenance through Arrow interchange. Correlated adoption/effectiveness/cost/capacity uncertainty must remain correlated.

Evaluate diffusion-only, capacity-only, loosely coupled and joint alternatives. Primary decision outcomes include expected net benefit and regret, not only forecast error. VOI calculations must correspond to a coherent observation/research design, not generic reductions in parameter variance.

Use `careops-sim` for healthcare operational examples and `tobac`/`vop_poc_nz` where their scientific scope and current contracts match. Qualify exact versions before use. MARS or another maintained surrogate remains optional until it preserves decision rankings, uncertainty and worthwhile computational savings on held-out full-simulator runs.
