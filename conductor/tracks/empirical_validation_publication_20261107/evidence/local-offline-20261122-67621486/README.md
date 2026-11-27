# Interrupted local offline replay

On 22 November 2026 the local Linux x86-64 guest replayed the selected hosted
workflow steps for Innovate commit `67621486c4d8658610a98d742060bea1b7108885`
and policy source `7d8aa3fc88f86cc0c54b2c4d57ca9a4bfa97bdda`. Its Gitleaks
candidate and scanner image built from the pinned inputs. The image build
verified its vendor assets without network access. The subsequent image controls
passed runtime, network-preflight and manifest-binding checks.

The operator interrupted the replay during asset-preflight when the shared
host drive fell to about 10 GiB free. The qualification receipt records that
stage as failed and the later checks as not run. Cleanup of the replay's own
container passed. This is an incomplete local replay and does not qualify a
local scan or image review.

The [qualification receipt](qualification.json) has SHA-256
`9504bedfba8fddb0f54c4be9e3327420447a3d4334bde3b1acccb111f5aa9c4b`.
The original [execution receipt](hosted-execution.json) and
[step status](check-status.tsv) are retained. The separate
[GitHub-hosted check](https://github.com/edithatogo/innovate/actions/runs/45004435187)
passed on the same Innovate commit and policy source.
