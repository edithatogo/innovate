# Medicines-utilisation capability and dataset validation

## Existing assets inspected

GitHub owner: `edithatogo/global-medicines-atlas`. Relevant source: `docs/qualification/m112-pbs-utilisation-role-readiness.md`, `scripts/m112_pbs_utilisation_replay.py`, and `quality/qualifications/australian-pbs-utilisation-landed-lineage-20261019.json`.

Hugging Face dataset: `edithatogo/australian-pbs-utilisation-archive`, exact reviewed revision `9f1d53cd50eea251b326da42a96f3ac72ebce97a`.

| Object path | SHA-256 |
|---|---|
| `raw/pbs/expenditure/Expenditure-prescriptions-report-tables-2023-24.XLSX` | `32ea64558d43531fed4bb395787bbc9279fdf45a2ddc6481ce8234aac676306b` |
| `raw/pbs/expenditure/Expenditure-prescriptions-report-tables-2024-25.XLSX` | `147f4c5392e256d85beb3e47dfd8445085e5a8f0ecbbd330b535774d81aadf99` |
| `raw/pbs/utilisation/dos/dos-jul-2024-to-jun-2025-phrmcy-type.csv` | `4a961d1d128e3aa9449a1d3a38c27f1b742d1c13dc07d12553134e04e701a6a4` |
| `raw/pbs/utilisation/dos/dos-jul-2025-to-jun-2026-phrmcy-type.csv` | `f502a61a472ef6db2986c5ca36a6f73b264cac1156830c38a5cf02f17bf59be8` |
| `raw/pbs/utilisation/reference/pbs-item-drug-map.csv` | `51adf3c80df2c2d526bfe88d2d96199e545579ced1b32e7ba3550dfc15f358a8` |
| `raw/pbs/utilisation/summary/dos-jul-2021-to-jun-2026.xlsx` | `48678536ddb71f9e7e4e6d654a0a90e6b242c22105e6864af1c6ef47318980ca` |

The newer role-readiness record reports six successful hosted replays and exact-scope rights/sensitivity assessment; it does not report v4 admission or a modelling-consumer canary. Older lineage entries retain historical unknown/unreviewed labels and must not override the later scoped assessment. Raw values were not inspected in this planning update. Filenames do not prove complete time coverage, usable temporal grain, ingredient mapping or analytical validity.

Reuse PBS schedule/source archives and Reimbursement/Global Medicines Atlas for funding, identity and reviewed links. Do not build parallel harvesters inside Innovate. Broader/private NICE or other sources discovered in the estate are not assumed available for this study. English prescribing is an external comparison candidate only after separate rights/schema/coverage qualification.

## Scientific observation contract

The first supported empirical outcome is aggregate supplied prescriptions/utilisation, not unique patients, first users or indication-specific use. Preserve transaction units, pack/strength, source date basis, geography, category, coverage and missingness. Distinguish annual reports, detailed monthly data and multi-year summaries; overlapping reports cannot be stacked as independent observations.

Canonical modelling panel fields: `source_id`, `source_revision`, `object_sha256`, `as_of`, `period_start`, `period_end`, `date_basis`, `geography`, `product_id`, `ingredient_id`, `strength`, `pack`, `category`, `measure`, `unit`, `value`, `denominator`, `mapping_status`, `provenance_id`. Nullable fields remain unknown, not invented. Field binding and duplicate keys are explicit. Ingredient/ATC or therapeutic-class mappings must be source-qualified; names alone do not establish therapeutic substitution.

The reusable model distinguishes starts, continuation, switching, cessation and supply intensity. For active stock, `N_next = N + starts + switches_in - stops - switches_out`. Dispensing is an observation process conditional on active stock, pack/supply intervals and coverage. Aggregate prescriptions alone generally do not identify all patient-level transition rates; impose labelled external assumptions, use additional measurements or report bounds.

## Validation programme

M01–M05 in the taskbook implement exact-byte manifest checks, source-faithful parsing, panel qualification, observation-model recovery and a frozen forecasting cohort. Do not make governed byte delivery dependent on already solving every domain question; equally, byte delivery does not certify model-ready data.

For the pilot, inventory observed months and revision dates before selecting origins. Where a sufficiently complete monthly panel exists, reserve the final 12 observed months as confirmation, use rolling origins with at least 24 earlier months, and evaluate 1/3/6/12-month horizons. If insufficient history exists, shorten the scope through a documented protocol amendment before fitting; do not fabricate a panel from workbook names.

Compare naive/seasonal, statistical count/state-space, classical diffusion, repeat-use competition/substitution and hype-augmented models. Evaluate volume, class shares and expenditure separately; price inflation or reimbursement changes are not automatically demand changes. Include zero/low-volume series, new entries, changed item codes, supply disruptions and generic/biosimilar competition when source-supported.

Report weighted interval/proper predictive scores, scale-appropriate point error, calibration, bias by series type, failures and runtime. Use paired block/series resampling appropriate to dependence. A 5% score-improvement target may be preregistered for practical value but is not a universal correctness threshold. Preserve results where simpler models win.

## Application-to-library promotion

The generic transaction/active-use/switching observation capability belongs in Innovate and should also work for non-medical repeat-use products. Medicines mappings, reimbursement facts, class-specific clinical interpretation and policy events stay in the data/application layer. Voiage receives properly mapped joint cost/outcome draws, not raw prescription counts treated as utilities.
