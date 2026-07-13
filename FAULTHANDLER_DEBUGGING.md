# Using Faulthandler for Debugging Segmentation Faults

## Overview
Faulthandler has been configured in this project to help diagnose segmentation faults that may occur during testing. This will help differentiate between issues in the Innovate library code versus issues in underlying dependencies.

## Configuration Details

### In tests/conftest.py:
- `faulthandler.enable()` is called to enable fault handling
- A traceback file is registered at `segfault_traceback.txt` in the project root
- The file will receive Python tracebacks when segmentation faults occur

### In enable_faulthandler.py:
- A standalone script to run tests with enhanced fault handling capabilities
- Can be used to run specific tests with fault handling enabled

## How to Use

### During Normal Testing:
When running pytest, faulthandler will automatically be enabled due to the conftest.py configuration.

```bash
python -m pytest tests/
```

### For Specific Debugging:
Use the enable_faulthandler.py script:

```bash
python enable_faulthandler.py
```

## Identifying Issues

### If the segfault traceback shows:
- Your code paths (innovate package): likely an issue in your implementation
- Dependency code (numpy, scipy, etc.): likely an issue in underlying dependencies

## Files Created/Modified

1. `tests/conftest.py` - Automatically enables faulthandler for all tests
2. `enable_faulthandler.py` - Standalone script for enhanced debugging
3. `segfault_traceback.txt` - Output file for segfault tracebacks (created when needed)

## Best Practices

- Always check the segfault_traceback.txt file if a segmentation fault occurs
- Segmentation faults in Python are typically caused by C extensions
- Look at the top of the traceback to see which module is causing the issue
- This helps prioritize fixes on your own code versus reporting issues to dependencies

## Example Output

When a segmentation fault occurs, you'll see a traceback like:
```
Fatal Python error: Segmentation fault
Current thread 0x00007f8b8c0a4740 (most recent call first):
  File "/path/to/your/code.py", line XX in function_name
  File "/path/to/dependency/module.so", line YY in c_function
```
The topmost Python file indicates where the call originated from your code.
