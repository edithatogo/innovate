# Kairos integration review

The same isolated lane implements bounded Bass finite-population DES at remote
Kairos revision d56863505d938feb069af88a737eb7668113d14b. Kairos owns scheduling,
random streams and distribution sampling. Innovate supplies the Bass birth
hazard and scales counts to market levels. No scheduler or RNG was copied.

Tests cover deterministic replay, changed seeds, monotonic/count bounds,
sampling-grid consistency, the independent innovation-only binomial ensemble
mean, and malformed options/domain errors. The explicit simulation engine is
advertised in discovery. Unknown engines fail rather than silently computing a
curve. Existing deterministic calls retain their semantics.

Review retains important scientific limits: zero initial adoption, finite
population, upward tick rounding, no agent/network/resource/intervention state,
and no simulation resume. RNG checkpoint metadata is provenance only. The
finite-population mean is not asserted equal to deterministic Bass with imitation.
The temporal resolution and population cap are documented. R numeric seed range
is limited by its exact integer representation. Full ABM/scenario parity stays
required in the ledger, together with all original/planned features.

The first package and broader Cargo checks failed because both volumes filled.
Only this lane's regenerable build caches were removed after their processes
ended; retained sources, artifacts and evidence were preserved. Retried native
checks, installed Python and R/Julia consumers, lint/types, Clippy, docs, package
metadata and the audit pass within their recorded scope. Core/C ABI coverage is
79.66%, below the release threshold; simulation module coverage is 90.08%.
Cargo audit retains the existing unsuppressed bincode maintenance warning.
Clean-source rebuild and platform distribution qualification remain open for
this dependency change. No hosted run, main integration or publication occurred.
