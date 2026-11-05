from dataclasses import dataclass
from pydantic import TypeAdapter, ConfigDict

@dataclass(kw_only=True)
class A:
    x: int

# In Pydantic V2, you can pass config to TypeAdapter:
try:
    ta = TypeAdapter(A, config=ConfigDict(extra='forbid', str_strip_whitespace=True))
    ta.validate_python({'x': 1, 'y': 2})
except Exception as e:
    print("Caught:", type(e), str(e))
