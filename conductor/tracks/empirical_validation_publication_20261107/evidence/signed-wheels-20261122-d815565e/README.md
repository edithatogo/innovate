# Signed installed-wheel evidence for the combined source

[Release run 45003005112](https://github.com/edithatogo/innovate/actions/runs/45003005112)
built and signed wheels on Linux x86-64, macOS ARM64 and Windows x86-64 from
commit `d815565ee740754694ff74468750f27202bafdf2`. Its source inventory
SHA-256 is `188a0e3640509cfc8b61f0bc360ef76418d3be918ecc473e2c06af2e08e5765c`.

Each signed receipt binds the original minimal and Arrow XML reports and the
tested wheel. All platforms passed 164 minimal and 42 Arrow cases without
skips. The live OECD and adopter-classification cases were exercised. The
GitHub CLI verified each provenance bundle against the repository, release
workflow, source commit and signer commit. The archive, wheel and report
hashes were checked after download.

| Target | Signed receipt artifact | Wheel artifact |
| --- | --- | --- |
| Linux x86-64 | 14746707875 | 14746733325 |
| macOS ARM64 | 14746119365 | 14746474203 |
| Windows x86-64 | 14746664495 | 14747160464 |

These are bounded installed-consumer results. The readiness job in the same
run failed because 37 release obligations remain unqualified. Native and
scientific acceptance receipts and the signed release manifest remain open.
