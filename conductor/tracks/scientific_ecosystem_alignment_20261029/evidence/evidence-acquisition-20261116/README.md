# Current-main installed Arrow and minimum-Rust evidence

Executed on 16 November 2026 from isolated merge
`2748084292e5e57c181c1665a686ad3a291145bf`, tree
`787eb4cd175b7343026cdad88143e8879e0b3062`, incorporating PR #662's prior head
`d36c33f6b4fb480828ddb9aae0121f3379667015` and main
`b29a1d27f54fdd08d77881ee999772fceab7c03a`.

This packet follows the 122-input and 175-input historical builds without
overwriting them. Main's intervening documentation additions, new native tests
and excluded games manifest change produce **178** tracked build inputs. The
[source verification](current-main-source-verification.json) checks each one
against its actual Git blob at b29a1d27. The complete repository trees are not
claimed to be equal; this branch also contains preparation tooling and evidence.
The build commit is local. Its [complete Git commit object](source-commit-object.txt)
hashes to the stated commit ID, and every recorded native input is separately
verified against published main; a later API publication commit has its own ID.

## Fresh observed build and installed result

- [Build provenance](build-provenance.json) and [unaltered build output](build.log):
  the new track-local executor ran Maturin 1.16.0 with `--locked`, Python 3.15.2,
  Rust/Cargo 1.100.0, dev profile and one build job. Exit 0 in 29.101 seconds;
  all native input hashes were unchanged after the build.
- Wheel `innovate-0.7.0a1-cp315-cp315-manylinux_2_35_x86_64.whl` SHA-256:
  `1c2ea5ef7b6408ba750600c3abbbb4a405b39ad42367abab5dd71d0e3842e93b`.
  This is a newly built local development artifact, not an sdist-derived release.
- [Installed qualification](qualification.json), [JUnit](qualification.junit.xml)
  and [pytest output](qualification.pytest.log): **42 passed**, zero skips,
  failures or errors. The 38 scenario-export cases include all four optional
  Arrow cases; four existing native producer/replay cases also passed.
- [Installation output](installation.log) and [commands/exits](commands.json):
  a fresh environment outside the checkout installed the identified wheel,
  Polars 2.1.0, PyArrow 26.0.1 and pytest 9.2.0. `uv pip check` passed. The
  isolated qualifier verifies actual native loading, package members, source,
  test and build-record identities. The JSON receipt was copied byte-for-byte
  from the original output named `qualification` to `qualification.json` for
  navigation; its referenced JUnit and pytest logs are also unaltered.

The qualifier and its 45 guard tests are unchanged by this increment. Its
SHA-256 is `fe894db45329c4fd9906aec6a77f8bd87653b0a7d927ac4aeaed39e6e099b3db`.
The native provider remains Kairos
`a1f6283575ad26918230adc122a99b5c08fc4da3`. This execution does not accept the
separate Paignia pin in the excluded games crate.

## Actual minimum-compiler check

[MSRV receipt](msrv-receipt.json) and [output](msrv-check.log) record
`cargo check --locked --workspace --all-targets -j 1` on **Rust 1.99.1**
(`58cf8f9c3`) and Cargo 1.99.1. Exit 0 in 26.264 seconds; all five workspace
crates compile on Linux with default features. This is compile evidence only:
tests, linking, distributable artifacts, the excluded games crate and other
platforms retain separate qualification requirements.

## Integration

The [build helper and complete recurring workflow candidate](../../integration/README.md)
make adoption concrete. The candidate is inert until the existing integrator
receives it into the root workflow namespace and observes its actual hosted
results. Ordinary CI's optional skips are not substituted for these 42 cases.
The 38 scientific obligations and exact-artifact release/security gates remain
open. This packet provides bounded executor evidence, not scientific sign-off.
