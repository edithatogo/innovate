---
title: Current Python API
description: Reference generated from the maintained native Python binding source and its checked docstrings.
---

The [generated Python reference](/innovate/api/generated/) is built from
`bindings/python/innovate` in the same checkout as this site. It describes the
current **0.7 development API** and includes signatures, parameters, returns,
errors and scientific limitations.

Start with the [native prediction tutorial](/innovate/user-guide/getting-started/)
for an executable example. Use the [native migration guide](/innovate/native-migration/)
for model-specific conventions and capabilities. The API reference describes
implemented behavior; it does not declare the scientific migration complete.

## How the reference is qualified

The build inventories the maintained source and stubs and rejects missing
Python docstrings or mismatched public parameter documentation. Reference
generation reads syntax without importing the numerical runtime. Generated files
are checked against the source and cannot silently replace missing exports.

Example execution and Rust compiler documentation checks are separate
requirements. Their status is recorded with the
[documentation quality contracts](/innovate/maintainers/documentation-quality/).

## Historical API

The [0.6 Python reference](/innovate/latest/api/python/) preserves the earlier
`fit_model` and backend interface. It is historical material and is not the
current installed development contract.
