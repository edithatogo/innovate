# Innovation and competition research register

Classify proposed work as an established method to reproduce, an integration/implementation gap to verify, or an unresolved empirical/methodological hypothesis. Search current software, inspect supported scope/licence, and record the result before claiming absence or novelty.

## Established methods and comparators

| Topic | Comparator | Target |
|---|---|---|
| Generalised diffusion, shocks and competition | CRAN DIMORA: https://cran.r-project.org/web/packages/DIMORA/index.html | Qualified native models and observation/uncertainty contracts; the underlying algorithms are not claimed new. |
| Differentiated-product demand/substitution/supply | PyBLP: https://pyblp.readthedocs.io/en/stable/introduction.html | Start with explicit logit/nested-logit cases; BLP/random-coefficients is a separate identified extension. |
| Strategic adoption/launch timing | When to adopt a service innovation: Nash equilibria in a competitive diffusion framework; publisher record https://www.sciencedirect.com/science/article/pii/S0377221718304454 | Reproduce assumptions/equilibrium before constructing innovation-specific extensions. |
| Hype/expectation dynamics | DOI 10.1016/j.techfore.2016.11.013; DOI 10.1016/j.technovation.2022.102459 | Joint fitting and uptake effects; neither quantitative hype models nor fitting are new by themselves. |
| General game algorithms | OpenSpiel/Gambit references; https://github.com/google-deepmind/open_spiel/blob/master/docs/algorithms.md | One provider with supported scope, residual/exploitability checks and actual native consumers. |
| Micro-to-macro diffusion | The statistical mechanics of innovation diffusion, DOI 10.1007/s10479-025-06894-0 | Reproduce a selected result and audit available code before assessing an implementation gap. |

These sources identify research/validation candidates, not permission to add domain frameworks to production. Independent Python/R/C++ oracles use isolated environments; supported Innovate scientific execution remains Rust.

## Research hypotheses

| Question | Capability | Falsification/evaluation |
|---|---|---|
| When does hype accelerate, inhibit or redirect adoption, versus respond to it? | Separate hype measurement, exogenous effects and optional joint feedback | No-effect/reverse/shared-shock simulations; temporal/technology holdouts; causal design required for causal claims. |
| Can latent demand, capacity and repeat use be separated from aggregate transactions? | Typed stock/flow observations plus Kairos constraints | Known-truth recovery and observational-equivalence counterexamples; bounds for unidentified quantities. |
| How do price, entry, switching and cannibalisation jointly shape innovation? | Competing diffusion and innovation-specific games | Independent static/dynamic oracles, held-out markets and explicit equilibrium selection. |
| When should firms launch, invest, wait or abandon? | Timing/real-option scenarios above the game provider | Analytic special cases, regret/exploitability, information/commitment sensitivity. |
| When does early-stage transfer across technologies help or harm? | Hierarchical mechanistic transfer and failure diagnostics | Leave-domain-out sparse/pre-inflection forecasts and calibrated out-of-domain uncertainty. |
| Which mechanisms create lock-in, tipping and hysteresis? | Multi-stable networks/complements/path dependence | Independent dynamics, bidirectional intervention sweeps, topology/measurement robustness. |
| Should a policymaker implement, intervene on diffusion or learn first? | Innovate/Kairos scenarios with Voiage study/decision analysis | Net benefit/regret against fixed/simple adaptive policies; coherent observation/research designs. |
| Can strategic-agent models retain useful calibrated macro forecasts? | Selected micro-to-macro family and optional simulation-based inference | Published-result reproduction first; held-out comparison with simpler macro models and simulation-cost accounting. |

These are hypotheses, not assertions that nobody has studied the questions. The possible contribution is reliable joint inference, interoperable native software and evidence in a defined setting. Rust, GPUs, neural inference or a large catalogue alone is not scientific novelty. Record null and negative results.
