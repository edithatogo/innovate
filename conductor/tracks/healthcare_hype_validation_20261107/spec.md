# Healthcare hype evaluation specification

## Question

Do attention and claims that exceed contemporaneously available evidence improve forecasts of healthcare utilisation, and under what conditions? A secondary question is whether flexible models outperform an imposed hype-cycle shape. Do not assume popularity is hype or that all innovations have a peak, trough and plateau.

## Current adequacy audit

At `b9998ad`, `hype/hype_cycle.py` adds a fixed-scale logistic term to positive/negative Gaussian pulses and clips the result at zero; it supplies prediction from assigned parameters, not a complete empirical measurement/inference workflow. `hype/hype_modified_bass.py` multiplies both Bass p and q by the same nonnegative `1 + visibility`, allowing only acceleration relative to its baseline. This is a time-rescaling restriction, not separate identified mechanisms for innovation and imitation.

`hype/delayed_hype_bass.py` time-shifts an external visibility function; its displayed equation contains current adoption state, not delayed adoption state. It also does not apply the non-delayed model's clipping rule. Verify equivalence and boundary behaviour before deciding whether a DDE solver is necessary. `abm/sentiment_hype_cycle.py` uses illustrative sentiment/adoption thresholds, fixed initial sentiment and irreversible adoption; it is not an estimated model of expectations and evidence.

These modules are candidates for illustrative simulation or restricted baselines, not currently adequate for a confirmatory clinical-hype or causal study. This is a source audit, not a newly executed full-package test.

## Required separation

Measure attention, claim strength/evidence alignment, evidence maturity/quality and observed utilisation separately. Pair claims with source evidence available at that date. Trial/publication counts, sentiment, regulatory approval and dispensing volume are not interchangeable proxies for clinical benefit or unsupported claims.

An initial descriptive/measurement study can proceed after source and annotation qualification. Prediction requires leakage-safe evaluation and calibrated inference. Causal claims require a separately defensible design. The [protocol](../../research/healthcare-hype-protocol-20261107.md) defines the staged study and stop rules.
