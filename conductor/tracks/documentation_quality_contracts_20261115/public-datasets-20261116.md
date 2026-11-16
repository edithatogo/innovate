# Public datasets for the documentation curriculum

## Status and selection

This is the DOC-04/06 shortlist for the
[learning programme](learning-plan-20261116.md) and
[curriculum](tutorial-coverage-20261116.md), researched on 16 November 2026.
Publisher pages and terms were inspected; no dataset has been downloaded,
normalized, fitted or admitted by this planning change. Pin the actual artifact
and inspect its embedded notes during admission. A current landing page is not a
frozen data edition.

| ID | Candidate | Proposed teaching use | Publisher terms observed; admission still pending |
| --- | --- | --- | --- |
| D01 | ITU internet-use indicator distributed through World Bank WDI | B03–B05, I04, I07: scalar fitting, comparison, grouped trajectories and batch work | Indicator metadata specifies CC BY 4.0 and ITU attribution |
| D02 | UK DESNZ solar deployment, Energy Trends ET 6.3 | B02, I06: monthly calendars, capacity, revisions, forecasting and parameter scenarios | GOV.UK states Open Government Licence v3.0 except where otherwise stated |
| D03 | UK DfT/DVLA vehicle licensing and first registrations | B02, I01: multiseries analysis and stock/flow distinctions | GOV.UK states Open Government Licence v3.0 except where otherwise stated |
| D04 | SocioPatterns workplace contacts | A01: empirical graph inputs with explicitly synthetic adoption outcomes | Publisher links CC0 1.0 and requests citation of the associated paper |
| D05 | Criteo corrected uplift dataset, optional | A05: treatment/overlap teaching if matched to an accepted native estimand | CC BY-NC-SA 4.0; exclude from the permissive default fixture bundle pending separate review |

D01 is the first candidate because it is small enough to excerpt transparently
and supports several beginner tasks. D02 and D03 add distinct observation
semantics; D04 adds measured network structure. D05 is optional and must not delay
the main curriculum. Synthetic data remains appropriate when the lesson requires
a known mechanism, ground truth or controlled failure.

## D01 — Internet use

[World Bank indicator metadata](https://databank.worldbank.org/metadataglossary/world-development-indicators/series/IT.NET.USER.ZS)
identifies `IT.NET.USER.ZS` as annual percentage of population using the internet,
with ITU as the originating source. Its definition concerns use in the preceding
three months. It is therefore a current-use measure, not cumulative first adoption.

Use a documented country/year subset and retain missing observations and country
notes. Explain percentage-to-fraction conversion, time origin and chronology.
Cross-country aggregation needs appropriate population weighting; summing national
percentages has no useful population interpretation. Country comparability and
revisions constrain any mechanistic interpretation of fitted curves.

Use the publisher's [API structures](https://datahelpdesk.worldbank.org/knowledgebase/articles/898581-api-basic-call-structures)
for a reproducible query. Save the complete raw response, pagination/query details,
metadata and normalization recipe before selecting a small offline teaching fixture.

## D02 — Solar deployment

The [DESNZ Energy Trends page](https://www.gov.uk/government/statistics/energy-trends-section-6-renewables)
currently distributes monthly solar photovoltaic deployment as **ET 6.3**, in
Excel and ODS formats. Record both the table identifier and edition because titles,
links and numbering can change.

Teach dates, scale, aggregation and forecast assessment using an explicitly chosen
capacity series. Installed capacity is not a count of adopters; changes can reflect
installation sizes, removals and revisions. A visible break near a policy date
does not identify that policy's effect. Show a parameter scenario as an assumption
and reserve causal claims for appropriate designs and supported operations.

Choose end-of-period or explicitly averaged capacity for calendar summaries;
summing monthly installed-capacity stocks would change the quantity. Preserve
missing observations instead of interpreting zero-filled calendar buckets as
observed zero capacity.

## D03 — Vehicle stocks and registrations

The [DfT/DVLA tables](https://www.gov.uk/government/statistical-data-sets/vehicle-licensing-statistics-data-tables)
include `VEH1103` for quarterly licensed vehicles by body/fuel type and `VEH1153`
for first registrations. Select matching geography, body type, fuel definitions
and periods and preserve the table notes.

Use this case to separate stock, entry flow, shares and absolute counts. Registered
vehicles do not count unique human adopters, and BEV/PHEV/ICE series are not
automatically successive Norton–Bass generations or a closed substituting market.
Start with transparent per-series analysis; only add structural substitution after
specifying an appropriate observation model and demonstrating its assumptions.
Aggregate compatible registration flows by summation; define stock summaries
separately and keep absent periods distinct from measured zeros.

## D04 — Workplace contact network

[SocioPatterns workplace contacts](https://sociopatterns.org/datasets/contacts-in-a-workplace/)
provide anonymized temporal contacts measured in France during June–July 2013,
with twenty-second intervals and department labels. The publisher requests citation
of Génois et al., *Network Science* 3, 326 (2015).

Use measured topology to teach graph preparation and distinguish the native
deterministic spillover family from the Bass/Kairos network engine. Adoption
outcomes remain simulated and must be labelled as such. Record edge weighting,
time-window selection, isolated-node handling and whether temporal contacts are
aggregated to a static graph; that aggregation changes the scientific input.
Retain anonymization and avoid individual identity inference.

## D05 — Optional uplift benchmark

The [Criteo publisher page](https://ailab.criteo.com/criteo-uplift-prediction-dataset/)
describes a corrected edition addressing an advertiser-related leak in the first
release. Select and hash that edition explicitly; inspect the actual schema rather
than relying on inconsistent prose feature counts. The source also describes
nonuniform subsampling, so population uplift interpretations require care.

This is a treatment-response example, not longitudinal diffusion data. Confirm
that the intended lesson matches an accepted native estimand and input contract.
Its noncommercial/share-alike terms and large source artifact require a separate
distribution decision; keep it outside default packaged fixtures and static
notebook exports until that is resolved. Existing controlled overlap teaching can
proceed independently.

## Admission contract

For each accepted artifact, extend the existing content/provenance contracts with
the following fields. Astro should derive its dataset catalogue and notebook
provenance panel from that record rather than maintain a second hand-written list.

| Area | Required record |
| --- | --- |
| Identity | Dataset ID, originating organization, distributor, title, indicator/table, edition/vintage, source URL and retrieval timestamp |
| Rights and attribution | Artifact-specific terms, required attribution, third-party exceptions, redistribution decision and accompanying licence record |
| Raw evidence | Original file/response, byte count, SHA-256, metadata/notes and query including pagination; retained archive location if it cannot be bundled |
| Normalized fixture | Schema, row count, hash, units, time origin/frequency, geography, denominator, variable meaning and data dictionary |
| Preparation | Deterministic selection/conversion script, excluded rows/reasons, missingness, duplicates, revisions, transformations and observation-unit decisions |
| Scientific task | Question, model/variant/operation/quantity, parameters/controls, assumptions, temporal split, seeds where relevant, expected failures and limitations |
| Reproduction | Code commit, native artifact and environment identities, fixture/output hashes, bounded command, required resources and explicit tolerance |
| Interpretation | Checked figures/tables, baseline comparison, diagnostics, relevant primary references, reviewer and unresolved findings |

Admission requires a concrete replay from raw or retained normalized evidence,
consistent schema/units, and an appropriate distribution decision. If the source
is revised, retain the old teaching vintage until a reviewable update explains
the change. Attribution accompanies both downloadable data and rendered exports.

## Acquisition, offline execution and updates

1. Acquire data in a separately invoked, bounded step from the publisher. Record
   the actual bytes and terms; do not fetch mutable live data during ordinary CI.
2. Convert the chosen table deterministically to a small supported tabular fixture.
   If XLSX/ODS parsing needs additional software, qualify it as acquisition tooling
   rather than introducing a pandas production path or a default runtime dependency.
3. Validate the teaching fixture and preserve its provenance. Use Polars and the
   native public interface for the maintained workflow.
4. Run normal scripts/notebooks offline. Assert input identity, expected structure,
   relevant numerical invariants and known failure cases within explicit budgets.
5. An explicit update compares source vintage, schema, row selection and important
   numerical/interpretive changes. It opens a reviewable change; it cannot silently
   replace golden outputs or mark a new source as accepted.

The empirical examples complement independent correctness fixtures. A visually
plausible fit to public data does not prove parameter identification, forecast
reliability, causal effect or external validity. Each lesson must explain the
specific evidence its workflow actually supplies.
