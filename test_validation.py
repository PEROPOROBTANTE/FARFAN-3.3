#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

from cicd.validation_gates import (
    ContractValidator,
    BindingValidator,
    ValidationGatePipeline
)

print("Testing validation gates...")
print()

print("1. Contract Validator:")
cv = ContractValidator()
result = cv.validate()
print(f"   Status: {result.status}")
print(f"   Passed: {result.passed}")
print(f"   Metrics: {result.metrics}")
print()

print("2. Binding Validator:")
bv = BindingValidator()
result = bv.validate()
print(f"   Status: {result.status}")
print(f"   Passed: {result.passed}")
print(f"   Metrics: {result.metrics}")
print()

print("✓ Validation gates initialized successfully")
