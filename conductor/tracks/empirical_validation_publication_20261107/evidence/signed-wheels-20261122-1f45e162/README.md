# Signed installed-wheel evidence for the combined source

[Release run 45015739813](https://github.com/edithatogo/innovate/actions/runs/45015739813)
built and signed wheels on Linux x86-64, macOS ARM64 and Windows x86-64 from
commit `1f45e1622eddf705596f586856f82cd79d1bc946`. The release source inventory SHA-256 is
`f5341d41dd984b43536daafc3709e5318fca2c711619c60ea46813438694d637`.

Each signed receipt binds the original minimal and Arrow XML reports and the
tested wheel. All platforms passed 164 minimal and 42 Arrow cases without
skips, including live OECD and adopter-classification cases. GitHub CLI
verified each provenance bundle against the repository, release workflow,
source commit and signer commit. Archive, wheel and report hashes were checked
after download. The original wheel archives are retained as workflow artifacts.

| Target | Signed receipt artifact | Wheel artifact |
| --- | --- | --- |
| linux-x86_64 | 14751762205 | 14752406097 |
| macos-arm64 | 14752625298 | 14751923059 |
| windows-x86_64 | 14752047855 | 14751434403 |

These are bounded installed-consumer results. The readiness job in the same
run failed because 37 release obligations remain unqualified. Native and
scientific acceptance receipts and the signed release manifest remain open.
