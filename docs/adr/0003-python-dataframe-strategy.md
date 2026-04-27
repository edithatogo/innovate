# ADR 0003: Keep pandas as the Primary Python DataFrame API, Use PyArrow as Infrastructure, and Adopt Polars Selectively

- Status: Accepted
- Date: 2026-04-16

## Context

The project already has substantial pandas usage across source, tests, docs, and examples, while direct Polars usage is absent in the current codebase. At the same time:

- pandas documents deep PyArrow integration, Arrow-backed dtypes, Arrow-backed compute acceleration, and interoperability with other Arrow-based dataframe libraries.
- Polars offers strong performance characteristics, especially through its lazy API and optimizer.
- Polars also documents an active breaking-change policy, explicit unstable functionality, and a release cadence that still tolerates fairly frequent breaking releases.

The project needs a DataFrame strategy that balances performance opportunities with stability, documentation burden, and migration cost.

## Decision

`innovate` will adopt the following Python DataFrame strategy:

1. pandas remains the primary user-facing DataFrame API.
2. PyArrow is treated as foundational infrastructure for columnar dtypes, interchange, and pandas Arrow-backed operation.
3. Polars may be introduced selectively for ETL-heavy, benchmark-corpus, or ingestion workflows where its lazy query engine provides a material win.
4. Polars-specific semantics will not become part of the stable public API without a later decision record.

## Consequences

### Positive

- Existing users and examples remain stable.
- The project gets a standards-aligned path to columnar interoperability through pandas plus PyArrow.
- Polars can still be used where it is strongest without forcing a whole-library rewrite.

### Negative

- The codebase may need adapters where selective Polars workflows feed pandas-facing APIs.
- Performance expectations will differ by workflow type rather than by one universal DataFrame engine.
- Documentation must distinguish public API choices from internal optimization choices.

## Alternatives Considered

### Rewrite the project around Polars

Rejected because it would create a large migration with limited benefit for the stable public contract and would replace one mature user-facing surface with a faster but more actively changing one.

### Ignore Polars entirely

Rejected because Polars remains useful for specific ETL and benchmark workflows that benefit from lazy execution and query optimization.

## References

- pandas PyArrow functionality: https://pandas.pydata.org/pandas-docs/stable/user_guide/pyarrow.html
- Polars versioning policy: https://docs.pola.rs/development/versioning/
- Polars lazy API usage: https://docs.pola.rs/user-guide/lazy/using/
