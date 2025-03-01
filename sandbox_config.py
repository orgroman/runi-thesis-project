"""
Configure sandbox for Temporal workflows.
This file is imported by the worker to ensure that workflow execution
is properly sandboxed and won't have non-deterministic behavior.
"""

print("Temporal sandbox configuration loaded")
# No special configuration needed - we'll rely on the default sandbox configuration
# and use safeguards in our workflow code instead.
