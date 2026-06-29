---
title: Arrow Interchange and Schema Layer
description: Versioned contract for moving kernel payloads between Python and bindings.
---

The Arrow interchange layer provides a versioned contract for moving kernel
payloads, diagnostics, and structured metadata between Python, PyArrow, and
future bindings.

## What the layer covers

- Kernel array payloads as typed Arrow tables and pandas DataFrames.
- Kernel table payloads as Arrow tables with schema metadata.
- Kernel discovery responses as compact Arrow tables for downstream tooling.
- Contract metadata that records schema version, payload kind, shapes, dtypes,
  and column names.

## Why it exists

The interchange layer gives binding authors a reference encoding that does not
depend on Python object identity or implementation-specific backend details.
That makes it easier to add non-Python bindings while keeping the payload
contract stable and inspectable.

## Example: inspect the contract summary

```python
from innovate import arrow_interchange

spec = arrow_interchange.describe_arrow_interchange()
print(spec["schema_version"])
print(spec["payloads"]["kernel_array"]["pandas"]["storage"])
```

## Example: round-trip an array payload

```python
import numpy as np
from innovate import arrow_interchange, kernel

payload = kernel.KernelArrayPayload(
    values=np.array([1.0, 2.0, 3.0]),
    metadata={"source": "example"},
)

table = arrow_interchange.kernel_array_payload_to_table(payload)
restored = arrow_interchange.kernel_array_payload_from_table(table)
assert restored.metadata["source"] == "example"
```

## Binding guidance

- Use the contract metadata to validate schema version and payload kind before
  deserializing data.
- Preserve unknown metadata fields when possible.
- Prefer Arrow tables for transport and pandas only for Python-facing
  ergonomics.
- Treat Polars as an optional downstream consumer of the Arrow tables rather
  than a required dependency in the interchange layer itself.
- Treat the contract version as the compatibility boundary for downstream
  bindings.
