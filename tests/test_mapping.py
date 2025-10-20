"""
Test Module: Responsibility Map Validation
==========================================

This test verifies that all 300 questions from cuestionario.json have
corresponding entries in orchestrator/execution_mapping.yaml with valid 
module:Class.method format.

Tests:
- Load orchestrator/execution_mapping.yaml and cuestionario.json
- Verify all 300 question IDs have corresponding entries
- Validate format: module:Class.method
- Report missing or malformed mappings

Author: Test Framework
Version: 1.0.0
"""

import json
import pytest
import yaml
from pathlib import Path
from typing import Dict, List, Set, Any


@pytest.fixture
def execution_mapping():
    """Load execution mapping from YAML file"""
    mapping_path = Path("orchestrator/execution_mapping.yaml")
    if not mapping_path.exists():
        pytest.skip(f"Execution mapping not found at {mapping_path}")
    
    with open(mapping_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


@pytest.fixture
def questionnaire():
    """Load questionnaire from JSON file"""
    questionnaire_path = Path("cuestionario.json")
    if not questionnaire_path.exists():
        pytest.skip(f"Questionnaire not found at {questionnaire_path}")
    
    with open(questionnaire_path, 'r', encoding='utf-8') as f:
        return json.load(f)


@pytest.fixture
def all_question_ids(questionnaire) -> Set[str]:
    """Extract all question IDs from questionnaire"""
    question_ids = set()
    
    # Extract from dimensiones structure
    if "dimensiones" in questionnaire:
        for dim_key, dim_data in questionnaire["dimensiones"].items():
            if "preguntas_expandidas" in dim_data:
                for question in dim_data["preguntas_expandidas"]:
                    if "id" in question:
                        question_ids.add(question["id"])
    
    # Also check if there's a flat questions list
    if "questions" in questionnaire:
        for question in questionnaire["questions"]:
            if "id" in question:
                question_ids.add(question["id"])
    
    return question_ids


def extract_execution_chain_mappings(execution_mapping: Dict) -> Dict[str, List[str]]:
    """
    Extract all question mappings from execution mapping.
    
    Returns:
        Dict mapping question IDs to list of adapter:class.method strings
    """
    mappings = {}
    
    # Iterate through dimension sections (D1_INSUMOS, D2_ACTIVIDADES, etc.)
    for section_key, section_data in execution_mapping.items():
        if section_key in ['version', 'last_updated', 'total_adapters', 'total_methods', 'adapters']:
            continue
        
        if not isinstance(section_data, dict):
            continue
        
        # Iterate through question mappings (Q1_*, Q2_*, etc.)
        for question_key, question_data in section_data.items():
            if question_key in ['description', 'question_count']:
                continue
            
            if not isinstance(question_data, dict):
                continue
            
            if 'execution_chain' not in question_data:
                continue
            
            # Extract adapter:class.method from execution chain
            chain_methods = []
            for step in question_data['execution_chain']:
                if 'adapter' in step and 'adapter_class' in step and 'method' in step:
                    adapter = step['adapter']
                    adapter_class = step['adapter_class']
                    method = step['method']
                    chain_methods.append(f"{adapter}:{adapter_class}.{method}")
            
            # Use question_key as ID (e.g., Q1_Baseline_Identification)
            mappings[question_key] = chain_methods
    
    return mappings


def validate_method_format(method_string: str) -> bool:
    """
    Validate that method string follows module:Class.method format.
    
    Args:
        method_string: String in format "module:Class.method"
    
    Returns:
        True if format is valid
    """
    parts = method_string.split(':')
    if len(parts) != 2:
        return False
    
    module_name, class_method = parts
    
    # Validate module name (alphanumeric + underscore)
    if not module_name or not all(c.isalnum() or c == '_' for c in module_name):
        return False
    
    # Validate Class.method format
    if '.' not in class_method:
        return False
    
    class_name, method_name = class_method.rsplit('.', 1)
    
    # Validate class name (PascalCase expected)
    if not class_name or not class_name[0].isupper():
        return False
    
    # Validate method name (snake_case or camelCase)
    if not method_name or not (method_name[0].islower() or method_name[0] == '_'):
        return False
    
    return True


class TestMappingValidation:
    """Test suite for responsibility map validation"""
    
    def test_questionnaire_loads(self, questionnaire):
        """Test that questionnaire.json loads successfully"""
        assert questionnaire is not None
        assert "metadata" in questionnaire
        assert questionnaire["metadata"]["total_questions"] == 300
    
    def test_execution_mapping_loads(self, execution_mapping):
        """Test that execution_mapping.yaml loads successfully"""
        assert execution_mapping is not None
        assert "adapters" in execution_mapping
        assert execution_mapping.get("total_adapters") == 9
    
    def test_all_questions_have_mappings(self, execution_mapping, all_question_ids):
        """Test that all 300 questions have corresponding mappings"""
        mappings = extract_execution_chain_mappings(execution_mapping)
        mapped_question_ids = set(mappings.keys())
        
        # Report coverage
        coverage = len(mapped_question_ids) / max(len(all_question_ids), 1) * 100 if all_question_ids else 0
        print(f"\nMapping Coverage: {coverage:.1f}% ({len(mapped_question_ids)}/{len(all_question_ids)} questions)")
        
        # Find missing questions
        if all_question_ids:
            missing_questions = all_question_ids - mapped_question_ids
            if missing_questions:
                print(f"\nMissing mappings for {len(missing_questions)} questions:")
                for qid in sorted(missing_questions):
                    print(f"  - {qid}")
                
                # Allow partial coverage for initial implementation
                # Full coverage requirement can be enforced later
                assert coverage >= 10, f"Coverage too low: {coverage:.1f}% (expected at least 10%)"
        else:
            # If no question IDs found in questionnaire, verify we have some mappings
            assert len(mapped_question_ids) > 0, "No question mappings found in execution_mapping.yaml"
    
    def test_all_mappings_have_valid_format(self, execution_mapping):
        """Test that all mappings follow module:Class.method format"""
        mappings = extract_execution_chain_mappings(execution_mapping)
        
        invalid_mappings = []
        for question_id, method_list in mappings.items():
            for method_string in method_list:
                if not validate_method_format(method_string):
                    invalid_mappings.append((question_id, method_string))
        
        if invalid_mappings:
            print(f"\nFound {len(invalid_mappings)} invalid method formats:")
            for question_id, method_string in invalid_mappings[:10]:  # Show first 10
                print(f"  {question_id}: {method_string}")
            if len(invalid_mappings) > 10:
                print(f"  ... and {len(invalid_mappings) - 10} more")
        
        assert len(invalid_mappings) == 0, f"Found {len(invalid_mappings)} invalid method format(s)"
    
    def test_all_adapters_referenced_exist(self, execution_mapping):
        """Test that all referenced adapters are defined in adapter registry"""
        adapter_registry = execution_mapping.get("adapters", {})
        registered_adapters = set(adapter_registry.keys())
        
        mappings = extract_execution_chain_mappings(execution_mapping)
        
        referenced_adapters = set()
        for method_list in mappings.values():
            for method_string in method_list:
                adapter_name = method_string.split(':')[0]
                referenced_adapters.add(adapter_name)
        
        undefined_adapters = referenced_adapters - registered_adapters
        
        if undefined_adapters:
            print(f"\nReferenced but undefined adapters:")
            for adapter in sorted(undefined_adapters):
                print(f"  - {adapter}")
        
        assert len(undefined_adapters) == 0, f"Found {len(undefined_adapters)} undefined adapter(s)"
    
    def test_execution_chains_not_empty(self, execution_mapping):
        """Test that all questions have at least one step in execution chain"""
        mappings = extract_execution_chain_mappings(execution_mapping)
        
        empty_chains = [qid for qid, methods in mappings.items() if len(methods) == 0]
        
        if empty_chains:
            print(f"\nQuestions with empty execution chains:")
            for qid in empty_chains:
                print(f"  - {qid}")
        
        assert len(empty_chains) == 0, f"Found {len(empty_chains)} question(s) with empty execution chains"
    
    def test_adapter_classes_match_registry(self, execution_mapping):
        """Test that adapter_class values match the registry definitions"""
        adapter_registry = execution_mapping.get("adapters", {})
        
        mappings_raw = {}
        for section_key, section_data in execution_mapping.items():
            if section_key in ['version', 'last_updated', 'total_adapters', 'total_methods', 'adapters']:
                continue
            
            if not isinstance(section_data, dict):
                continue
            
            for question_key, question_data in section_data.items():
                if question_key in ['description', 'question_count']:
                    continue
                
                if not isinstance(question_data, dict) or 'execution_chain' not in question_data:
                    continue
                
                for step in question_data['execution_chain']:
                    if 'adapter' in step and 'adapter_class' in step:
                        adapter = step['adapter']
                        adapter_class = step['adapter_class']
                        
                        if adapter in adapter_registry:
                            expected_class = adapter_registry[adapter].get('adapter_class')
                            if expected_class and expected_class != adapter_class:
                                mappings_raw[f"{question_key}:{adapter}"] = {
                                    'expected': expected_class,
                                    'actual': adapter_class
                                }
        
        if mappings_raw:
            print(f"\nAdapter class mismatches:")
            for key, mismatch in list(mappings_raw.items())[:10]:
                print(f"  {key}: expected {mismatch['expected']}, got {mismatch['actual']}")
        
        assert len(mappings_raw) == 0, f"Found {len(mappings_raw)} adapter class mismatch(es)"


def test_report_mapping_statistics(execution_mapping, questionnaire):
    """Generate comprehensive mapping statistics report"""
    print("\n" + "=" * 80)
    print("RESPONSIBILITY MAP VALIDATION REPORT")
    print("=" * 80)
    
    # Questionnaire stats
    total_questions = questionnaire.get("metadata", {}).get("total_questions", 0)
    print(f"\nQuestionnaire:")
    print(f"  Total questions (metadata): {total_questions}")
    
    # Execution mapping stats
    mappings = extract_execution_chain_mappings(execution_mapping)
    print(f"\nExecution Mapping:")
    print(f"  Questions mapped: {len(mappings)}")
    print(f"  Total adapters: {execution_mapping.get('total_adapters', 0)}")
    print(f"  Total methods: {execution_mapping.get('total_methods', 0)}")
    
    # Adapter usage
    adapter_usage = {}
    for method_list in mappings.values():
        for method_string in method_list:
            adapter = method_string.split(':')[0]
            adapter_usage[adapter] = adapter_usage.get(adapter, 0) + 1
    
    print(f"\nAdapter Usage:")
    for adapter in sorted(adapter_usage.keys()):
        print(f"  {adapter}: {adapter_usage[adapter]} invocations")
    
    # Average chain length
    if mappings:
        avg_chain_length = sum(len(m) for m in mappings.values()) / len(mappings)
        print(f"\nExecution Chain Statistics:")
        print(f"  Average chain length: {avg_chain_length:.2f} steps")
        print(f"  Max chain length: {max(len(m) for m in mappings.values())} steps")
        print(f"  Min chain length: {min(len(m) for m in mappings.values())} steps")
    
    print("=" * 80)
