"""
Test Canary System - Validation Tests
======================================

Unit tests for the canary regression detection system.
"""

import json
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.canary_generator import CanaryGenerator
from tests.canary_runner import CanaryRunner
from tests.canary_fix_generator import CanaryFixGenerator


class TestCanaryGenerator:
    """Test canary generation functionality"""
    
    def test_generator_initialization(self, tmp_path):
        """Test CanaryGenerator initialization"""
        generator = CanaryGenerator(output_dir=tmp_path)
        assert generator.output_dir == tmp_path
        assert len(generator.ADAPTER_METHODS) == 9
        assert sum(generator.ADAPTER_METHODS.values()) == 413
    
    def test_method_definitions_structure(self):
        """Test that method definitions have correct structure"""
        generator = CanaryGenerator()
        
        for adapter_name in generator.ADAPTER_METHODS.keys():
            method_defs = generator._get_method_definitions(adapter_name)
            
            for method_def in method_defs:
                assert "name" in method_def
                assert "inputs" in method_def
                assert "args" in method_def["inputs"]
                assert "kwargs" in method_def["inputs"]
                assert isinstance(method_def["inputs"]["args"], list)
                assert isinstance(method_def["inputs"]["kwargs"], dict)
    
    def test_hash_computation(self):
        """Test hash computation is deterministic"""
        generator = CanaryGenerator()
        
        test_data = {
            "module_name": "test",
            "data": {"value": 42},
            "evidence": [{"type": "test"}]
        }
        
        hash1 = generator._compute_hash(test_data)
        hash2 = generator._compute_hash(test_data)
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex length


class TestCanaryRunner:
    """Test canary runner functionality"""
    
    def test_runner_initialization(self, tmp_path):
        """Test CanaryRunner initialization"""
        runner = CanaryRunner(canary_dir=tmp_path)
        assert runner.canary_dir == tmp_path
        assert runner.report.total_methods == 0
        assert runner.report.passed == 0
        assert runner.report.failed == 0
    
    def test_contract_validation(self):
        """Test contract validation logic"""
        runner = CanaryRunner()
        
        # Valid output
        valid_output = {
            "module_name": "test",
            "class_name": "Test",
            "method_name": "test_method",
            "status": "success",
            "data": {},
            "evidence": [],
            "confidence": 0.9
        }
        
        violations = runner._validate_contract(valid_output)
        assert len(violations) == 0
        
        # Invalid output - missing required key
        invalid_output = {
            "module_name": "test",
            "data": {}
        }
        
        violations = runner._validate_contract(invalid_output)
        assert len(violations) > 0
    
    def test_evidence_validation(self):
        """Test evidence structure validation"""
        runner = CanaryRunner()
        
        # Valid evidence
        valid_output = {
            "evidence": [
                {"type": "test", "confidence": 0.9},
                {"type": "test2", "value": "data"}
            ]
        }
        
        violations = runner._validate_evidence(valid_output)
        assert len(violations) == 0
        
        # Invalid evidence - missing type
        invalid_output = {
            "evidence": [
                {"confidence": 0.9}
            ]
        }
        
        violations = runner._validate_evidence(invalid_output)
        assert len(violations) > 0


class TestCanaryFixGenerator:
    """Test fix generation functionality"""
    
    def test_fix_generator_initialization(self, tmp_path):
        """Test CanaryFixGenerator initialization"""
        report_file = tmp_path / "test_report.json"
        report_file.write_text(json.dumps({
            "violations": [],
            "summary": {"total_methods": 413, "passed": 413, "failed": 0}
        }))
        
        generator = CanaryFixGenerator(report_file=report_file)
        assert generator.report_file == report_file
        assert generator.load_report()
    
    def test_fix_operation_generation(self, tmp_path):
        """Test fix operation generation from violations"""
        report_file = tmp_path / "test_report.json"
        report_data = {
            "violations": [
                {
                    "adapter": "policy_processor",
                    "method": "process",
                    "type": "HASH_DELTA",
                    "details": "Hash mismatch"
                },
                {
                    "adapter": "analyzer_one",
                    "method": "analyze_document",
                    "type": "CONTRACT_TYPE_ERROR",
                    "details": "Missing key 'status'"
                }
            ],
            "summary": {"total_methods": 413, "passed": 411, "failed": 2}
        }
        
        report_file.write_text(json.dumps(report_data))
        
        generator = CanaryFixGenerator(report_file=report_file)
        generator.load_report()
        fix_ops = generator.generate_all_fixes()
        
        assert len(fix_ops) == 2
        assert fix_ops[0].operation_type in ["REBASELINE", "TYPE_FIX", "SCHEMA_FIX"]


class TestIntegration:
    """Integration tests for the complete canary system"""
    
    def test_adapter_count(self):
        """Verify we have all 9 adapters defined"""
        generator = CanaryGenerator()
        assert len(generator.ADAPTER_METHODS) == 9
    
    def test_method_count(self):
        """Verify total method count is 413"""
        generator = CanaryGenerator()
        total = sum(generator.ADAPTER_METHODS.values())
        assert total == 413
    
    def test_adapter_names(self):
        """Verify all expected adapters are present"""
        generator = CanaryGenerator()
        expected_adapters = [
            "policy_processor",
            "policy_segmenter",
            "analyzer_one",
            "embedding_policy",
            "semantic_chunking_policy",
            "financial_viability",
            "dereck_beach",
            "contradiction_detection",
            "teoria_cambio"
        ]
        
        for adapter in expected_adapters:
            assert adapter in generator.ADAPTER_METHODS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
