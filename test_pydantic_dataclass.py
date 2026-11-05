from dataclasses import dataclass
from pydantic import TypeAdapter, ConfigDict

@dataclass
class MyModel:
    __pydantic_config__ = ConfigDict(extra='forbid')
    name: str

try:
    print(TypeAdapter(MyModel).validate_python({"name": "test", "extra": 1}))
except Exception as e:
    print(type(e), e)
