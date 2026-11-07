# Hype effects on innovation, with a medicines-utilisation application

Revision: 7 November 2026, following explicit maintainer clarification. This replaces the earlier claim-exaggeration-centred protocol. Hype is the trajectory and influence of attention, expectations, enthusiasm and disappointment; unsupported claims are neither its definition nor a prerequisite for this study. No empirical analysis or model validation is claimed yet.

## Questions

1. Which composite or dynamic expectation models fit and forecast observed attention/expectation trajectories, including no-cycle and repeated-wave cases?
2. Does hype improve prediction of innovation adoption, use, switching or abandonment beyond a no-hype model and relevant access/price/availability variables?
3. Under what measurements and designs can the effect of hype be distinguished from adoption-driven attention and shared shocks?
4. How does hype interact with competing products, capacity constraints and strategic launch/investment decisions?

The first two are predictive questions. Model-conditional scenarios are not empirical causal effects. A causal claim requires an additional identification design; lagged correlation alone is insufficient.

## Composite baseline and the current shape problem

The current `src/innovate/hype/hype_cycle.py` is `0.5*logistic + positive Gaussian - negative Gaussian`, clipped at zero. It fixes the maturity scale, does not constrain component ordering/widths, and provides assigned-parameter prediction rather than a complete fit. The extrema of the sum do not occur automatically at the Gaussian centres. Hard clipping can flatten the trough and remove useful gradient information. Do not patch a desired appearance with arbitrary clipping.

Target latent mean:

`h(t) = b + L*sigmoid(k*(t-t0)) + A*exp(-(t-mu_p)^2/(2*w_p^2)) - D*exp(-(t-mu_d)^2/(2*w_d^2))`.

Fit components jointly. Use training-only time scaling, positive width/growth transforms and optional `mu_d = mu_p + softplus(delta)` only for the explicitly cycle-ordered family. Free baseline and plateau replace the fixed 0.5. A general wave model may omit ordering, add pulses or shrink amplitudes to zero. This is an additive composite mean, not a finite-mixture likelihood.

Select an observation model: Student-t/Gaussian for a continuous expectation index; count likelihood with a log/softplus link for counts; a justified bounded model for bounded observations. A negative latent deviation is permitted even when observed counts are nonnegative. Do not fit a normalised search index as an absolute count.

A constrained multistart likelihood fit is the baseline. Conditional variable projection of amplitudes is permitted for identity-link Gaussian least squares only; do not apply its linearity argument to a nonlinear count link. Native Bayesian HMC/NUTS with hierarchical partial pooling can quantify uncertainty across technologies and regularise sparse data. It does not resolve structural non-identifiability or prove a cycle. Require priors expressed on scaled time/amplitude, prior-predictive checks, independent log-density/gradient oracles, convergence and posterior-predictive diagnostics. Failed sampling produces no fabricated posterior.

## From a hype curve to its impact

Replace the existing common multiplier of Bass p and q with separately estimable effects where data support them:

`log p_j(t) = alpha_pj + beta_pj*h_j(t-lag_p) + x_j(t)'gamma_p`;
`log q_j(t) = alpha_qj + beta_qj*h_j(t-lag_q) + x_j(t)'gamma_q`.

Coefficients can be positive or negative. Cumulative first-adopter counts remain nondecreasing; cessation belongs in an active-use/abandonment state, not a negative cumulative-adoption derivative. Competing-product attention can enter via prespecified cross-effects. Begin with exogenous observed hype; introduce joint hype/adoption feedback only after known-truth identifiability tests and appropriate additional measurements.

Verify whether the existing delayed model is merely external forcing `h(t-delay)` or a genuine delayed-state equation before changing its solver. A delayed input does not by itself require a delay differential equation solver.

## Data and study design

Use medicines as an application of the generic method. The [medicines plan](medicines-utilisation-20261107.md) records exact existing PBS archive objects and qualification gaps. Candidate attention measures include pageviews, news and appropriately sourced search/expectation surveys. Attention is a proxy, not a direct measurement of expectations; use multiple indicators or sensitivity analysis and label the construct honestly.

Select technologies from source coverage and a prespecified sampling frame, not observed peaks. Include quiet trajectories and repeated waves. Match geography, observation dates and temporal resolution; preserve revisions and ascertainment changes. The core study does not require manually grading whether claims are exaggerated. Evidence, regulatory, reimbursement, supply and price events may be covariates when appropriately dated and sourced.

Pilot models on known synthetic cases first: monotone maturity only, one peak/trough, repeated peaks, inhibitory hype, lagged response, no effect, reverse feedback, shared shocks, product substitution and capacity-limited utilisation. Then compare frozen temporal and held-out-technology forecasts against no-hype diffusion, naive/autoregressive, spline/GAM or state-space alternatives and existing published hype models. Estimate incremental proper-score improvement, calibration, peak/trough timing uncertainty, false-cycle detection and failure rate. Avoid selecting the final model on the held-out cohort.

## Primary prior art and proposed outputs

Shi and Herniman, The role of expectation in innovation evolution: Exploring hype cycles, DOI 10.1016/j.technovation.2022.102459, connects expectations and diffusion. Silvestrini et al., Rate equation leading to hype-type evolution curves, DOI 10.1016/j.techfore.2016.11.013, reports quantitative fitting and uncertainty code. Therefore neither quantitative hype modelling nor parameter fitting is claimed as new. Check accessible implementations/licences before reusing or describing an implementation gap.

Initial contribution: a verified composite/dynamic fitting toolkit and an out-of-sample study of hype's incremental impact on innovation utilisation. Primary venue candidates are Technological Forecasting and Social Change, Technovation, or International Journal of Forecasting for a substantial methodological comparison. A medicines-specific result can be submitted separately to an appropriate pharmacoepidemiology venue. Claims about novelty, superiority or causal impact require their own evidence.

Stop or simplify when the data do not distinguish mechanisms. Report null predictive results, poor identifiability and cases where a simpler model is better. Native migration and empirical validity remain separate gates.
