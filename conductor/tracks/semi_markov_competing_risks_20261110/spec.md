# Specification — Semi-Markov tunnel states and competing-risks hazard kernels for tobac

## Purpose and boundary

Extend `innovate`'s Rust-native multi-product substitution and switching family (`A04`, `A05` in [advanced_diffusion_frontier_20261107](../advanced_diffusion_frontier_20261107/spec.md)) with duration-dependent (semi-Markov) tunnel states, bi-exponential / frailty relapse decay kernels, and competing-risks transition matrix exponential / Padé generators consumed by `edithatogo/tobac` (`BF3`, `F02`, `F04` in `conductor/tracks/dynamic_frontier_20261108/`).

`innovate` owns generic multi-state adoption, switching, relapse-hazard, and Polars/Arrow trajectory kernels. `tobac` owns tobacco/vaping product definitions (`never`, `combustible`, `vaping`, `dual`, `former`), disease excess-risk curves, and regulatory policy evaluation.

## Scope

1. **Competing-risks generator and matrix-exponential transition step (`SM01`)**:
   - Provide a generic continuous-time competing-risks generator $Q(\theta, t)$ and uniformization / Padé matrix-exponential transition operator $P(\Delta t) = \exp(Q\,\Delta t)$ that guarantees non-negative transition probabilities, exact row-sum conservation ($\sum_j P_{ij}(\Delta t) = 1$), and exact semigroup composition $P(\Delta t_1)\,P(\Delta t_2) = P(\Delta t_1 + \Delta t_2)$.
2. **Semi-Markov duration-since-switching / cessation tunnel states (`SM02`)**:
   - Support duration-indexed tunnel states $\tau \in \{0, 1, \dots, K\}$ and fast/slow frailty mixtures so relapse and abandonment hazards decay non-exponentially with time since switching or cessation, plus excess-risk decay kernels $w(\tau) = \alpha e^{-\lambda_1 \tau} + (1 - \alpha)e^{-\lambda_2 \tau}$.
3. **Zero-copy Polars/Arrow cohort trajectory interchange (`SM03`)**:
   - Export cohort state-occupancy and transition-flow trajectories as typed Polars/Arrow tables preserving draw, scenario, age/cohort, subgroup, and duration-bin keys for downstream consumption by `tobac` (`src/tobac_supply/dynamic.py`, `src/tobac/health.py`) and `dapane` (`tobac_regulatory_fiscal_valuation_20261110`).
