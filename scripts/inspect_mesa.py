import inspect
import mesa

def inspect_module(module):
    for name, obj in inspect.getmembers(module):
        print(name, obj)
        if inspect.ismodule(obj):
            inspect_module(obj)

inspect_module(mesa)
