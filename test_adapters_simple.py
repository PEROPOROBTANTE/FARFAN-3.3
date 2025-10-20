#!/usr/bin/env python3
"""
Simple test to validate adapter layer modules
"""

def test_imports():
    """Test that all adapter modules can be imported"""
    print("Testing adapter imports...")
    
    try:
        from orchestrator.adapter_policy_processor import PolicyProcessorAdapter
        print("✓ PolicyProcessorAdapter")
    except Exception as e:
        print(f"✗ PolicyProcessorAdapter: {e}")
    
    try:
        from orchestrator.adapter_embedding_policy import EmbeddingPolicyAdapter
        print("✓ EmbeddingPolicyAdapter")
    except Exception as e:
        print(f"✗ EmbeddingPolicyAdapter: {e}")
    
    try:
        from orchestrator.adapter_dereck_beach import DerekBeachAdapter
        print("✓ DerekBeachAdapter")
    except Exception as e:
        print(f"✗ DerekBeachAdapter: {e}")
    
    try:
        from orchestrator.adapter_contradiction_detection import ContradictionDetectionAdapter
        print("✓ ContradictionDetectionAdapter")
    except Exception as e:
        print(f"✗ ContradictionDetectionAdapter: {e}")
    
    try:
        from orchestrator.adapter_causal_processor import CausalProcessorAdapter
        print("✓ CausalProcessorAdapter")
    except Exception as e:
        print(f"✗ CausalProcessorAdapter: {e}")
    
    try:
        from orchestrator.adapter_teoria_cambio import TeoriaCambioAdapter
        print("✓ TeoriaCambioAdapter")
    except Exception as e:
        print(f"✗ TeoriaCambioAdapter: {e}")
    
    try:
        from orchestrator.adapter_financial_viability import FinancialViabilityAdapter
        print("✓ FinancialViabilityAdapter")
    except Exception as e:
        print(f"✗ FinancialViabilityAdapter: {e}")
    
    print("\nAll adapter imports tested!")

if __name__ == "__main__":
    test_imports()
