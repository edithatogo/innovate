# PR #611 native documentation hash repair

## Failure and scope

The retained `documentation-execution-evidence` artifact `14133344113` from
Innovate run `43642091421` (head `da852056f2ecd679aa990adba7e70ea2513276bc`)
was downloaded and verified against its published ZIP digest:
`d07e030106e76302467a0b17d25e42add9835cf46278b7af60d688ef9dc565b8`.

Its Julia and R runners reported successful execution, and Python examples and
API doctests passed. The final aggregation failed both native receipts with
`Invalid polyglot source path or hash.` All 23 source identities use the
inventory generator's `sha256:<64 lowercase hexadecimal digits>` form, while
`_verified_native_inventory` required a bare digest. Inventory-file, native
binary, example-code and installed R code hashes use bare digests as intended.

The verifier source is unchanged on PR #611 at
`1ea85c8b3b243edbf5d6d29525cfe496a8c7104b` and was independently checked against
Git blob `110960776df725edc0f1998f0e69678f38da83db`. Its documentation workflow
matched blob `975d4a72d396dc382fd61901a19d69bfdc49886a` before editing.

## Repair and tests

- [x] Reproduce the mismatch using the exact source-verifier functions.
- [x] Normalize only the exact optional `sha256:` label when validating and
  independently rehashing source inventory entries. Preserve the original map
  for exact receipt comparisons and retain compatibility with existing bare
  SHA-256 source inventories. Do not broaden other hash fields.
- [x] Add nine focused regression tests and wire the full module tests into the
  existing documentation-contract job on the required Python 3.15.2.
- [x] Reject three deliberate mutations: removed source rehashing, disabled hash
  format validation, and disabled relative-path validation.
- [ ] Obtain independent current-head review and full hosted documentation,
  security, native/platform and consumer qualification.
- [ ] Verify merge and receiving-main status before closing integration.

Local command: `python test_docs_hashes_locally.py final-green` (the disposable
harness is retained with local evidence, not shipped as production code).
Actual interpreter: CPython 3.13.5. The initial nine-test run produced three
assertion failures and one expected verifier error, all attributable to valid
prefixed identities being rejected. After repair all nine tests pass without
skips. All three safety mutations produce test failures. The eleven existing
release-workflow regression tests also pass unchanged.

Because the full documentation checker contains newer Python exception syntax,
the local harness compiles the eight exact verifier/helper function definitions
from the hash-verified source, without changing their bodies or runtime checks.
This is bounded function qualification, not full-module execution on 3.15.2.
The committed test module imports the complete checker normally in hosted CI.
Its cases cover prefixed and legacy bare identities, CRLF/CR normalization,
changed and missing sources, wrong well-formed digests, 15 malformed/unknown
algorithm identities, path traversal, omitted runners and unchanged strict
native-binary digest validation.

No native execution is fabricated or inferred from the local synthetic fixtures.
The prior hosted receipt is diagnostic evidence, not final-head acceptance.
Source rehashing, exact receipt-source map equality, binary identities, example
counts/assertions and all scientific/full-release restrictions remain intact.
No Pages activation setting, runtime pin, production Rust or offline-security
infrastructure was changed.
