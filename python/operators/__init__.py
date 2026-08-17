"""
Operators Framework - Universal operator testing framework.

This package provides:
1. Base classes for defining operators
2. Registry for operator discovery
3. Test framework for running tests based on JSON specs
4. Common utilities for quantization

Directory Structure:
    operators/
    ├── __init__.py           # This file
    ├── base.py               # Base classes
    ├── registry.py           # Operator registry
    ├── test_framework.py     # Test runner
    ├── common/               # Shared utilities
    │   ├── types.py          # Type definitions
    │   └── csrc/             # Shared CUDA code
    └── <operator_family>/    # Operator implementations
        ├── manifest.json
        ├── csrc/
        └── variants/

Usage:
    from operators import OperatorRegistry, TestFramework

    # Discover all operators
    registry = OperatorRegistry()
    registry.discover()

    # Run tests
    framework = TestFramework(registry)
    results = framework.run_all()
"""

from .base import (
    TensorSpec,
    ParamSpec,
    OperatorSpec,
    BaseOperator,
    OperatorFamily,
)

from .registry import OperatorRegistry

from .test_framework import (
    TestConfig,
    TestResult,
    TestFramework,
)

__version__ = "0.1.0"

__all__ = [
    # Base classes
    "TensorSpec",
    "ParamSpec",
    "OperatorSpec",
    "BaseOperator",
    "OperatorFamily",
    # Registry
    "OperatorRegistry",
    # Test framework
    "TestConfig",
    "TestResult",
    "TestFramework",
]
