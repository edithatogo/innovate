with open("tests/unit/test_advanced_runtime_contracts.py", "r") as f:
    content = f.read()

# Remove the import line from the middle
content = content.replace("\nfrom innovate.advanced_runtime import compare_policy_scenarios\n\n", "\n")

# Replace the original import statement at the top to include `compare_policy_scenarios`
original_import = """from innovate.advanced_runtime import (
    AdvancedCapability,
    AdvancedResult,
    AdvancedRuntimePolicy,
    detect_advanced_backends,
    get_advanced_capability,
    list_advanced_capabilities,
    select_advanced_backend,
)"""

new_import = """from innovate.advanced_runtime import (
    AdvancedCapability,
    AdvancedResult,
    AdvancedRuntimePolicy,
    compare_policy_scenarios,
    detect_advanced_backends,
    get_advanced_capability,
    list_advanced_capabilities,
    select_advanced_backend,
)"""

content = content.replace(original_import, new_import)

with open("tests/unit/test_advanced_runtime_contracts.py", "w") as f:
    f.write(content)
