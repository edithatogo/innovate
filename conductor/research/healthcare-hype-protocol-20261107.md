# Healthcare attention, evidence and adoption — study concept

Date: 7 November 2026. Status: proposed protocol; no data collection, statistical validation, ethics determination or preregistration is claimed.

## Scientific question

When do healthcare claims exceed the evidence available at the time, and does that mismatch help predict subsequent utilisation beyond evidence, reimbursement, supply and existing adoption trends? Test the prevalence and predictive usefulness of cycle-shaped dynamics rather than fitting every technology into a predetermined cycle.

Keep four constructs separate: attention volume; strength/scope/certainty of claims; quality and maturity of supporting evidence; and observed utilisation. High attention can reflect real benefit, safety concerns or public interest. Low uptake can reflect price, access or capacity. Neither is sufficient evidence of hype.

## Prior art and potential contribution

Sumner et al. (2014, BMJ, DOI 10.1136/bmj.g7015) linked exaggeration in academic press releases and news using paired research sources. Sumner et al. (2016, PLOS ONE, DOI 10.1371/journal.pone.0168217) extended this to journal press releases and caveats. Bratton et al. (2019, Wellcome Open Research, DOI 10.12688/wellcomeopenres.15486.2) replicated some but not all associations. Reuse their construct distinctions and check availability/licensing of supplementary corpora. Do not claim the first healthcare-hype study.

The proposed contribution is a dated, uncertainty-aware linkage between claim/evidence alignment and later utilisation, evaluated across technologies with temporal holdouts and alternatives to a fixed cycle. Failure to find predictive value is an informative result.

## Feasible first cohort

Begin with a source-availability pilot in medicines, where repeated utilisation is measurable. Candidate classes include GLP-1 receptor agonists, SGLT2 inhibitors, PCSK9 inhibitors and CGRP-targeted treatments. These are candidate comparators, not a declaration that any class is overhyped. Select the final intervention/indication/geography units using preregistered coverage rules, not visible peak shapes or fame. Include less-prominent and discontinued/unsuccessful candidates where ascertainable to reduce selection and survival bias.

Claim/evidence coding should be indication-specific. Aggregate dispensing may not reveal indication, new starters or unique patients; do not manufacture those fields. Use a coarser utilisation estimand, obtain suitable data, or exclude that comparison. Clinical AI, genomics, psychedelics and devices are later cohorts when comparable uptake data exist, not prerequisites for the pilot.

## Data layers and qualification

| Layer | Candidate sources | Required caveats |
|---|---|---|
| Attention | Wikimedia pageviews; GDELT; appropriately licensed/search-accessible Google Trends | Wikipedia language is not country of exposure; searches are normalised/sampled, not absolute counts; GDELT historical availability varies by product/mode and must be probed and frozen. |
| Claims | Dated research articles, press releases and news with source linkage | Retain document version, claim span, stated population/intervention/outcome, effect magnitude, uncertainty and availability claims. Respect text copyright and syndication. |
| Evidence | PubMed/Europe PMC, registered-trial histories, systematic reviews and contemporaneous assessments | Counts and trial phase are not proof of benefit. Code design, bias, precision, population relevance, replication and conflicts. Do not use later evidence to label earlier claims without a separate retrospective estimand. |
| Regulation/funding/access | Dated TGA/FDA, PBS/PBAC and relevant reimbursement records | Approval, reimbursement, supply and appropriateness differ. Historical status needs dated evidence, not today's record substituted into past months. |
| Utilisation | Qualified PBS utilisation archives, AIHW/Services Australia series, English prescribing where harmonisable | Repeated scripts are not new adopters; public counts may omit private/off-label use. Revisions, processing-versus-supply date and indication ambiguity must be retained. |

Use public aggregate data first; seek the appropriate institutional determination before research involving restricted records or identifiable people. Do not make allegations about individuals or companies from model scores. Publish evidence-based estimates with uncertainty and accessible correction mechanisms.

## Annotation and measurement

Pilot a stratified sample of paired claims and contemporaneous evidence. Use independent reviewers, an explicit codebook, blinded adjudication and agreement/uncertainty reporting. Distinguish supported, overstated, contradicted and indeterminate claims; retain separate dimensions for causality, magnitude, population generalisation, certainty and time-to-availability. A source paper itself can contain overstatement and is not an infallible reference.

A proposed initial feasibility sample is 400 paired claims, split into development and locked temporal/technology holdouts; this is a workload proposal, not a powered sample-size calculation. Final size, strata and precision targets follow the pilot and are frozen before confirmation. Estimate prevalence using sampling weights where sampling is unequal. Any automated text classifier must be evaluated against held-out human labels with leakage controls, calibration, abstention and error analysis. Carry label uncertainty into downstream models; do not replace expert evidence assessment with an unvalidated language-model score.

## Analysis stages

A. Measurement/descriptive study: source coverage, reliability of claim/evidence alignment, attention profiles and dated evidence changes. This stage does not require the whole Rust migration.

B. Predictive study: compare naive/autoregressive baselines, standard diffusion without attention, flexible distributed-lag/GAM or state-space models, the existing restricted cycle model and any new mechanistic model. Use blocked temporal validation and held-out technologies. Test whether added variables improve proper predictive scores and calibration, not just in-sample R-squared.

C. Mechanistic extension: allow no peak, repeated peaks and inhibition; separate identifiable effects on innovation and imitation; represent abandonment separately from cumulative first adoption. Include an observation model and explicit access/capacity constraints where supported. Verify synthetic known-truth and no-effect/common-cause cases before empirical interpretation.

D. Causal/decision extension: only with a defensible identified design and a clearly stated estimand. Regulatory changes and press releases are not automatically exogenous shocks. Lagged association or Granger predictability is not evidence that hype caused prescribing. Kairos and Voiage become useful for downstream capacity/research decisions only after the upstream constructs and uncertainty are qualified.

## Prespecified safeguards and stopping rules

Freeze cohort, synonyms, source queries, vintages, event dates, outcomes, comparison methods and analysis windows. Address common causes, reverse causation, multiple testing, changing media coverage, duplicate stories and outcome-dependent selection. Do not combine arbitrary standardised scores into a purported universal hype index without measurement validation. Do not infer cost/waste from attention alone.

If claim/evidence coding is unreliable, report a measurement/feasibility result and stop the latent-hype analysis. If uptake cannot be harmonised, report attention/evidence results only. If parameters cannot be recovered in synthetic tests, use a simpler model or bounds. If attention adds no held-out value, report that result rather than searching for a favourable cycle.

## Outputs and venues

First output: measurement/feasibility and predictive protocol, then a confirmatory study. Prefer JMIR Infodemiology for the information-quality/attention study; consider JMIR for a broader externally validated digital-health information contribution. Use International Journal of Forecasting for a substantial forecasting-method result and Medical Decision Making for a later implementation/research-decision contribution. Keep the JOSS software paper separate. Assess current journal requirements and disclose AI assistance; no submission is authorised here.

## Sources checked for this planning update

- https://pubmed.ncbi.nlm.nih.gov/25498121/ — primary paired-source study.
- https://pubmed.ncbi.nlm.nih.gov/27978540/ — journal press releases/caveats study.
- https://pubmed.ncbi.nlm.nih.gov/31728413/ — replication with mixed findings.
- https://doc.wikimedia.org/generated-data-platform/aqs/analytics-api/reference/page-views.html — pageview endpoints start July 2015.
- https://support.google.com/trends/answer/4365533?hl=en — sampling and 0–100 normalisation; low-volume censoring.
- https://gdeltproject.org/data.html and https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/ — product documentation; exact historical retrieval remains an acquisition test.
- https://www.aihw.gov.au/reports/medicines/pbs-monthly-data/contents/technical-notes — date-of-supply series and revision caveats.
- https://infodemiology.jmir.org/about-journal/focus-and-scope and https://www.jmir.org/about-journal/focus-and-scope — venue scope.
- https://joss.readthedocs.io/en/latest/submitting.html — software submission and AI-use requirements.
