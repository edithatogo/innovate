"""Configuration for pytest with faulthandler enabled to catch segmentation faults."""

import faulthandler

# Enable faulthandler to get Python tracebacks from segmentation faults
faulthandler.enable()

print("Faulthandler enabled for all tests.")
print("This will help identify if segfaults are from your code or dependencies.")
