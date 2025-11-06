"""Mutation testing configuration for the innovate library."""

import subprocess
from mutmut import Config, MutationID, Context


def pre_mutation(context):
    """Hook function called before each mutation."""
    # This function can be used to skip mutations for specific files
    pass


def post_mutation(context):
    """Hook function called after each mutation."""
    # This can be used to check results after mutation
    pass


def init():
    """Initialize mutation testing configuration."""
    # Add any initialization code here if needed
    pass


# Configuration for mutmut
def should_skip_mutation(context: Context) -> bool:
    """
    Define conditions to skip specific mutations.
    
    Args:
        context: Mutation context object
        
    Returns:
        True if mutation should be skipped, False otherwise
    """
    # Skip mutations in test files
    if "test" in context.filename:
        return True
    
    # Skip mutations in certain utility files if needed
    # For example, configurations, setup files, etc.
    return False


# Define the command to run tests
def get_test_command():
    """Get the command to run tests for mutation testing."""
    # Run tests with JAX platform set to CPU to avoid CUDA issues
    return "JAX_PLATFORM_NAME=cpu python -m pytest tests/ -x --tb=short"


# Configuration object
config = Config(
    test_command="JAX_PLATFORM_NAME=cpu python -m pytest tests/ -x --tb=short",
    runner="JAX_PLATFORM_NAME=cpu python -m pytest",
    paths_to_mutate=["src/innovate/**/*.py"],
    backup=False,  # Don't create backups
    dict_synonyms=[],
    total_replacement_ids=[],
    no_progress=False,
    ci=False,
    pre_mutation_hook=pre_mutation,
    post_mutation_hook=post_mutation,
    coverage_data=None,
    include_coverage=False,
    paths_to_exclude=[],
    junit_xml=None,
)