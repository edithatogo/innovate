# Python/Arrow candidate qualification

Base 83d556ba4995fd5f5238f4953e9a8af34976f464. Python 3.15.1 verified
in this isolated worktree. `uv lock --upgrade --python 3.15.1` and locked dev
installation passed, including statsmodels 0.14.6 source compilation.

- 23 selected Arrow interchange/kernel contract/adapter tests passed.
- 86 migration/release evidence tests passed; one docs-dist check skipped because
  the site is not built. That skip is not documentation acceptance.
- pip-audit of the installed base/dev environment reported no known vulnerabilities.
- Correctness/security agent review found and fixed old release-evidence labels:
  the active contract requires3.15.1; new evidence records the actual interpreter.
- ArviZ1.3.0 changes from_dict and plotting APIs used by optional fitters. Keep
  0.23.4 (<1) until BayesianFitter/BlackJAX adapters and optional tests migrate.
  Owner: Innovate; revisit at that adapter migration or compatible upstream API.
- Ruptures1.1.10 live PyPI metadata requires Python<3.14. Retain1.1.9 for
  change-point detection; revisit on upstream support or qualified exact commit.

A broader unit run was interrupted after review found active migration gaps;
its old-version workflow assertion failure was corrected. It is not a full-suite
pass. Optional JAX/Polars, installed wheel, container/Conda/HPC and Rust/Kairos pin
qualification remain open. Existing accepted scientific fixtures are unchanged.
The updated lock is a candidate, not blanket qualification of every optional extra.
