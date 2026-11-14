---
title: Research software registry readiness
description: Current Software Heritage, RRID and JOSS evidence and remaining gates.
---

The 0.7 development line is preparing research software registry evidence under
[issue #398](https://github.com/edithatogo/innovate/issues/398). Software Heritage,
RRID and JOSS acceptance remain pending. Native scientific qualification and
release approval are separate gates.

| Destination | Current position | Next evidence |
| --- | --- | --- |
| Software Heritage | Two recorded origin lookups returned not found. | Select the release, retain successful ingestion and resolve its SWHID. |
| RRID | Research software is a plausible resource category; duplicate checking is incomplete. | Complete the identity check and retain a curator decision and resolving RRID. |
| JOSS | The legacy manuscripts do not qualify the current native software. | Prepare a current evidence-backed paper after its scientific, research-use and author prerequisites are met. |

Read the [current readiness contract](https://github.com/edithatogo/innovate/blob/main/docs/registry-readiness.md)
for authoritative source links, observed provider responses, the machine-readable
record and outstanding work. It supersedes historical pre-0.7 readiness flags.

From a development checkout, use the selected Python 3.15.2 environment:

```sh
.venv-native/bin/python scripts/check_registry_readiness.py --json
.venv-native/bin/python scripts/check_registry_readiness.py --require-complete --json
```

The default command checks record integrity, including explicit pending work.
The completion check fails while any destination lacks a recorded final
disposition. Neither command performs online verification or a submission.
Tests run in the existing `tests/native` CI suite.
