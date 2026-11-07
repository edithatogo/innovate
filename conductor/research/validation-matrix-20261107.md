# Model and feature validation matrix

Status: candidate acquisition/evaluation plan, not an acquired or qualified corpus. Exact source versions, licensing, available fields, temporal coverage and preprocessing must be audited before inclusion. Use independent known-truth simulations alongside every empirical comparison.

## Model families

| Model/feature | Candidate empirical sources | Verification and external-evaluation requirements |
|---|---|---|
| Bass, logistic, Gompertz; symmetric/skewed growth | NBER CHAT; Australian Clean Energy Regulator installation data | Analytic/independent reference parity; early-stage rolling forecasts; uncertain saturation; distinguish usage from adoption. |
| Hierarchical, mixture and heterogeneous adoption | CHAT country–technology panels; CER postcode panels; English prescribing by practice | Known-group/mixture recovery; partial pooling; held-out groups; latent-class uncertainty rather than asserted true classes. |
| Fisher–Pry substitution | Historical merchant-marine wood/metal data in R diffusion; IEA EV sales/stocks | Correct denominator and limiting cases; sales shares are not installed-stock shares. |
| Multi-generation Norton–Bass | Four IBM computer generations in R diffusion; qualified generation-resolved sources | Multi-generation parameter/state recovery and reference reproduction; small historical fixtures alone do not establish generality. |
| Multi-product, Lotka–Volterra, composite and market-share models | DIMORA energy competition data; therapeutic-class prescribing shares | Joint constraints and interaction recovery in simulation; held-out joint prediction; do not infer causal competition from co-trending shares. |
| Complementary goods, platforms and co-evolution | IEA EV/charging series; compatible solar/battery series | Coupled-state verification, lagged prediction and confounding sensitivity; separate complementary association from causal effect. |
| Latent-process, regime-switching, time-varying and streaming models | CER revisions/vintages; DIMORA shock examples; dated prescribing panels | Known change points, drift and missingness; real-time vintages; batch/incremental parity; no future revision leakage. |
| Network diffusion and ABM | Diffusion of Microfinance network/participation data; SNAP MemeTracker; suitable temporal contact networks | Known-network recovery, independent simulation and temporal holdouts; network/contact observations are not complete transmission truth. |
| Policy hazard/event history | State Policy Innovation and Diffusion; Correlates of State Policy Project | Censoring, staggered timing and risk-set tests; adoption hazard is not policy effectiveness. |
| Causal policy effects | ACIC known-potential-outcome tasks plus independently generated longitudinal/interference designs | Bias, coverage, overlap, confounding and heterogeneous-effect recovery for the exact estimand; ACIC alone does not cover all designs. |
| Hype, attention, failure and lock-in | Claim/evidence corpora from Sumner et al.; Wikimedia/GDELT/qualified search series paired with utilisation | Follow the separate hype protocol; attention is not unsupported optimism; failure/lock-in definitions require independent justification. |
| SIR/SIS/SEIR and contagion | Applicable sbibm tasks; SocioPatterns with simulated outcomes | Analytic/known-truth inference and conservation; contact data alone do not validate infection/adoption effects. |
| Adopter categorisation | Individually timed adoption where rights permit, plus synthetic cohorts | Verify declared classification thresholds; do not treat conventional categories as discovered behavioural mechanisms. |

## Cross-cutting features

| Feature | Test material and acceptance |
|---|---|
| Optimisation, ODEs and gradients | Analytic/high-precision references, independent generators, parameter constraints, extreme scales and finite-difference/autodiff agreement. |
| Posterior inference | Simulation-based calibration, known/reference posteriors, multimodality, invalid fits and misspecification; distinguish uncertainty types and expose non-convergence. |
| Bootstrap/prediction intervals | Repeated simulations with serial dependence, heteroscedasticity and sparse observations; coverage plus width, not coverage alone. |
| Kairos simulation | Analytic queue/birth–death fixtures, resource conservation, independent engines, reproducible random streams and event-order/replay tests. |
| Ensembles and streaming | Frozen data vintages, controlled drift and revisions, out-of-sample calibration and incremental/batch agreement. |
| MARS/neural/other surrogates | Held-out full-simulator runs, difficult regimes, extrapolation checks, decision rankings and accuracy–cost trade-offs. |
| Arrow, native kernels and language bindings | Golden valid/invalid schemas, units, nulls, categories, draw IDs, ownership/lifetimes and errors; installed native consumers without Python. |
| Reports, provenance and release claims | Known results plus missing/invalid evidence; displayed values, traceability and limitations must survive export; capability claims need executable receipts. |

## Existing estate to audit and reuse

Candidate Hugging Face sources: `edithatogo/australian-pbs-utilisation-archive`, `edithatogo/australian-pbs-source-archive`, `edithatogo/australian-mbs-source-archive`, and `edithatogo/reimbursement-atlas`. Verify live repository, revision, rights and actual coverage before use. A funding schedule is not utilisation, regulatory approval or clinical effectiveness. Preserve upstream acquisition records rather than building competing source archives here.

Code/protocols remain in GitHub; distribute only rights-permitted derived benchmark data, manifests and splits through the relevant Hugging Face repositories. Use immutable paper-specific snapshots and persistent identifiers. Restricted text/data stays with its source; share identifiers, permitted annotations and reproducible acquisition instructions instead.

## Initial comparison protocol

Start with technology (CHAT), energy (CER) and healthcare (PBS plus appropriately harmonised English prescribing). Retain small IBM/DIMORA reference reproductions. Register source vintages, units, inclusion rules, rolling forecast origins, horizons, tuning budgets and baselines before confirmation. Include pre-inflection and held-out technology/region tasks, count-based and cumulative evaluation where appropriate, probabilistic scores, calibration, failure rates and computational cost. Do not estimate saturation or select model families using future observations.

External source starting points: https://www.nber.org/research/data/cross-country-historical-adoption-technology ; https://cer.gov.au/markets/reports-and-data/small-scale-installation-postcode-data ; https://www.iea.org/data-and-statistics/data-tools/global-ev-data-explorer ; https://cran.r-universe.dev/diffusion/doc/manual.html ; https://cran.r-universe.dev/DIMORA/doc/manual.html ; https://openprescribing.net/ ; https://web.stanford.edu/~jacksonm/Data ; https://snap.stanford.edu/data/ ; https://sites.psu.edu/spap/spid-data/ ; https://github.com/sbi-benchmark/sbibm . These links identify candidates, not acceptance receipts.
