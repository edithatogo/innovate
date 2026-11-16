# Current architecture principles

[ADR 0006](adr/0006-rust-polars-native-only.md) defines the maintained direction:
Rust owns numerical execution, Polars owns the table interface, and Python,
Julia and R are the supported host-language bindings. Kairos owns the selected
simulation mechanisms and Voiage owns downstream economic analysis.

The earlier Python-first, optional-backend principles were superseded by that
decision. Their complete text remains in Git history; historical acceptance
dates do not override the native contract. See the
[modernization roadmap](architecture_modernization_roadmap.md) for remaining
scientific obligations and the [decision index](adr/index.md) for provenance.
