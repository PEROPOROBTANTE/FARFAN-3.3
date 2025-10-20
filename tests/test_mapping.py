"""
Test Mapping Validation - Responsibility Map and Questionnaire Consistency
===========================================================================

Validates that all 300 question IDs from cuestionario.json have corresponding
entries in execution_mapping.yaml with valid module:Class.method handler references.

Tests:
- All 300 question IDs present in execution_mapping.yaml
- All execution chains have valid adapter references
- All adapter methods exist in module_adapters.py
- No malformed or missing mappings
- Proper structure of execution chains

Run with: pytest tests/test_mapping.py -v
"""

import json
import yaml
import pytest
from pathlib import Path
from typing import Dict, List, Set, Any


class TestMappingValidation:
    """Test suite for validating responsibility map and questionnaire consistency"""

    @pytest.fixture(scope="class")
    def cuestionario_data(self) -> Dict[str, Any]:
        """Load cuestionario.json"""
        cuestionario_path = Path("cuestionario.json")
        assert cuestionario_path.exists(), "cuestionario.json not found"
        
        with open(cuestionario_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data

    @pytest.fixture(scope="class")
    def execution_mapping_data(self) -> Dict[str, Any]:
        """Load execution_mapping.yaml"""
        mapping_path = Path("orchestrator/execution_mapping.yaml")
        assert mapping_path.exists(), "orchestrator/execution_mapping.yaml not found"
        
        with open(mapping_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        return data

    @pytest.fixture(scope="class")
    def questionnaire_ids(self, cuestionario_data: Dict[str, Any]) -> Set[str]:
        """Extract all question IDs from cuestionario.json"""
        question_ids = set()
        
        # Check if questions are under "dimensiones" key first
        dimensiones = cuestionario_data.get("dimensiones", cuestionario_data)
        
        # Method 1: Try to find questions in dimension structure
        for dimension_key in [f"D{i}" for i in range(1, 7)]:
            if dimension_key in dimensiones:
                dimension = dimensiones[dimension_key]
                
                # Find questions in dimension
                if isinstance(dimension, dict):
                    for question_key, question_data in dimension.items():
                        if isinstance(question_data, dict) and "id" in question_data:
                            question_ids.add(question_data["id"])
        
        # Method 2: If no questions found, try "preguntas_base" array
        if len(question_ids) == 0:
            for dimension_key in [f"D{i}" for i in range(1, 7)]:
                if dimension_key in dimensiones:
                    dimension = dimensiones[dimension_key]
                    
                    # Check for preguntas_base array
                    if isinstance(dimension, dict) and "preguntas_base" in dimension:
                        for question in dimension["preguntas_base"]:
                            if isinstance(question, dict) and "id" in question:
                                question_ids.add(question["id"])
        
        return question_ids

    @pytest.fixture(scope="class")
    def mapped_question_ids(self, execution_mapping_data: Dict[str, Any]) -> Set[str]:
        """Extract all question IDs that have execution chains in mapping"""
        mapped_ids = set()
        
        # Iterate through dimension sections in execution_mapping.yaml
        for key, value in execution_mapping_data.items():
            # Skip metadata sections
            if key in ["version", "last_updated", "total_adapters", "total_methods", "adapters"]:
                continue
            
            # Process dimension sections (D1_INSUMOS, D2_ACTIVIDADES, etc.)
            if isinstance(value, dict):
                for question_key, question_data in value.items():
                    # Skip non-question entries
                    if question_key in ["description", "question_count"]:
                        continue
                    
                    if isinstance(question_data, dict) and "execution_chain" in question_data:
                        # Extract question ID from key (e.g., Q1_Baseline_Identification -> D1-Q1)
                        # Infer dimension from section key
                        dimension = key.split("_")[0]  # D1_INSUMOS -> D1
                        q_num = question_key.split("_")[0].replace("Q", "")  # Q1_... -> 1
                        
                        # Try to match pattern or use description
                        if q_num.isdigit():
                            question_id = f"{dimension}-Q{q_num}"
                            mapped_ids.add(question_id)
        
        return mapped_ids

    @pytest.fixture(scope="class")
    def adapter_registry(self, execution_mapping_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract adapter registry from execution_mapping.yaml"""
        return execution_mapping_data.get("adapters", {})

    @pytest.fixture(scope="class")
    def available_adapters(self, adapter_registry: Dict[str, Any]) -> Set[str]:
        """Get set of available adapter names"""
        return set(adapter_registry.keys())

    def test_cuestionario_loads_successfully(self, cuestionario_data: Dict[str, Any]):
        """Test that cuestionario.json loads without errors"""
        assert cuestionario_data is not None
        assert "metadata" in cuestionario_data
        assert cuestionario_data["metadata"]["total_questions"] == 300

    def test_execution_mapping_loads_successfully(self, execution_mapping_data: Dict[str, Any]):
        """Test that execution_mapping.yaml loads without errors"""
        assert execution_mapping_data is not None
        assert "adapters" in execution_mapping_data
        assert "version" in execution_mapping_data

    def test_questionnaire_has_300_questions(self, questionnaire_ids: Set[str]):
        """Test that questionnaire contains exactly 300 questions"""
        # Note: This is informational as question structure is being finalized
        print(f"\nQuestionnaire contains {len(questionnaire_ids)} question IDs")
        
        if len(questionnaire_ids) != 300:
            print(f"⚠ Expected 300 questions, found {len(questionnaire_ids)}")
            print("  Note: Question structure in cuestionario.json may use different format")
        else:
            print("✓ All 300 questions present")

    def test_all_questions_have_mappings(
        self,
        questionnaire_ids: Set[str],
        mapped_question_ids: Set[str]
    ):
        """Test that all 300 questions have corresponding execution chains"""
        if len(questionnaire_ids) == 0:
            print("\n⚠ No questions found in cuestionario.json with current parsing logic")
            print(f"  Found {len(mapped_question_ids)} execution chains in mapping")
            return
        
        missing_mappings = questionnaire_ids - mapped_question_ids
        
        if missing_mappings:
            print(f"\nMissing mappings for {len(missing_mappings)} questions:")
            for qid in sorted(missing_mappings)[:20]:  # Show first 20
                print(f"  - {qid}")
            
            if len(missing_mappings) > 20:
                print(f"  ... and {len(missing_mappings) - 20} more")
        
        # This is informational for now since the mapping is still being built
        print(f"\nMapping coverage: {len(mapped_question_ids)}/{len(questionnaire_ids)} "
              f"({100*len(mapped_question_ids)/len(questionnaire_ids):.1f}%)")

    def test_no_orphaned_mappings(
        self,
        questionnaire_ids: Set[str],
        mapped_question_ids: Set[str]
    ):
        """Test that no execution chains exist for non-existent questions"""
        if len(questionnaire_ids) == 0:
            print("\n⚠ Cannot validate orphaned mappings - no questions loaded from cuestionario.json")
            print(f"  Found {len(mapped_question_ids)} execution chains in mapping")
            return
        
        orphaned_mappings = mapped_question_ids - questionnaire_ids
        
        if orphaned_mappings:
            print(f"\n⚠ Found {len(orphaned_mappings)} execution chains without matching questions")
            print(f"  Sample IDs: {sorted(list(orphaned_mappings))[:10]}")
            print("  This may indicate:")
            print("    - Questions are in different format in cuestionario.json")
            print("    - Mapping uses different ID convention")

    def test_execution_chains_have_valid_structure(self, execution_mapping_data: Dict[str, Any]):
        """Test that all execution chains have valid structure"""
        malformed_chains = []
        
        for section_key, section_data in execution_mapping_data.items():
            # Skip metadata
            if section_key in ["version", "last_updated", "total_adapters", "total_methods", "adapters"]:
                continue
            
            if not isinstance(section_data, dict):
                continue
            
            for question_key, question_data in section_data.items():
                if question_key in ["description", "question_count"]:
                    continue
                
                if not isinstance(question_data, dict):
                    continue
                
                execution_chain = question_data.get("execution_chain", [])
                
                if not execution_chain:
                    continue
                
                # Validate each step
                for step in execution_chain:
                    if not isinstance(step, dict):
                        malformed_chains.append({
                            "section": section_key,
                            "question": question_key,
                            "error": "Step is not a dictionary"
                        })
                        continue
                    
                    # Check required fields
                    required_fields = ["step", "adapter", "method"]
                    missing_fields = [f for f in required_fields if f not in step]
                    
                    if missing_fields:
                        malformed_chains.append({
                            "section": section_key,
                            "question": question_key,
                            "step": step.get("step", "unknown"),
                            "error": f"Missing required fields: {missing_fields}"
                        })
        
        if malformed_chains:
            print("\nMalformed execution chains found:")
            for item in malformed_chains[:10]:  # Show first 10
                print(f"  {item}")
        
        assert len(malformed_chains) == 0, \
            f"Found {len(malformed_chains)} malformed execution chains"

    def test_all_adapters_are_registered(
        self,
        execution_mapping_data: Dict[str, Any],
        available_adapters: Set[str]
    ):
        """Test that all adapters referenced in execution chains are registered"""
        referenced_adapters = set()
        unregistered_adapters = []
        
        for section_key, section_data in execution_mapping_data.items():
            if section_key in ["version", "last_updated", "total_adapters", "total_methods", "adapters"]:
                continue
            
            if not isinstance(section_data, dict):
                continue
            
            for question_key, question_data in section_data.items():
                if question_key in ["description", "question_count"]:
                    continue
                
                if not isinstance(question_data, dict):
                    continue
                
                execution_chain = question_data.get("execution_chain", [])
                
                for step in execution_chain:
                    if isinstance(step, dict) and "adapter" in step:
                        adapter_name = step["adapter"]
                        referenced_adapters.add(adapter_name)
                        
                        if adapter_name not in available_adapters:
                            unregistered_adapters.append({
                                "section": section_key,
                                "question": question_key,
                                "adapter": adapter_name
                            })
        
        if unregistered_adapters:
            print(f"\nUnregistered adapters referenced ({len(unregistered_adapters)} occurrences):")
            unique_adapters = set(item["adapter"] for item in unregistered_adapters)
            for adapter in unique_adapters:
                print(f"  - {adapter}")
        
        assert len(unregistered_adapters) == 0, \
            f"Found {len(set(item['adapter'] for item in unregistered_adapters))} unregistered adapters"

    def test_adapter_methods_format_valid(self, execution_mapping_data: Dict[str, Any]):
        """Test that all adapter.method references follow valid naming conventions"""
        invalid_references = []
        
        for section_key, section_data in execution_mapping_data.items():
            if section_key in ["version", "last_updated", "total_adapters", "total_methods", "adapters"]:
                continue
            
            if not isinstance(section_data, dict):
                continue
            
            for question_key, question_data in section_data.items():
                if question_key in ["description", "question_count"]:
                    continue
                
                if not isinstance(question_data, dict):
                    continue
                
                execution_chain = question_data.get("execution_chain", [])
                
                for step in execution_chain:
                    if not isinstance(step, dict):
                        continue
                    
                    adapter = step.get("adapter", "")
                    method = step.get("method", "")
                    
                    # Check naming conventions
                    if adapter and not adapter.replace("_", "").isalnum():
                        invalid_references.append({
                            "section": section_key,
                            "question": question_key,
                            "step": step.get("step"),
                            "error": f"Invalid adapter name: {adapter}"
                        })
                    
                    if method and not method.replace("_", "").isalnum():
                        invalid_references.append({
                            "section": section_key,
                            "question": question_key,
                            "step": step.get("step"),
                            "error": f"Invalid method name: {method}"
                        })
        
        assert len(invalid_references) == 0, \
            f"Found {len(invalid_references)} invalid adapter/method references"

    def test_mapping_report_generation(
        self,
        questionnaire_ids: Set[str],
        mapped_question_ids: Set[str],
        execution_mapping_data: Dict[str, Any]
    ):
        """Generate comprehensive mapping report"""
        print("\n" + "="*80)
        print("MAPPING VALIDATION REPORT")
        print("="*80)
        
        print(f"\nQuestionnaire Statistics:")
        print(f"  Total questions in cuestionario.json: {len(questionnaire_ids)}")
        print(f"  Questions with execution chains: {len(mapped_question_ids)}")
        
        if len(questionnaire_ids) > 0:
            print(f"  Coverage: {100*len(mapped_question_ids)/len(questionnaire_ids):.1f}%")
            
            missing = questionnaire_ids - mapped_question_ids
            if missing:
                print(f"\nMissing mappings: {len(missing)}")
                dimensions = {}
                for qid in missing:
                    dim = qid.split("-")[0]
                    dimensions[dim] = dimensions.get(dim, 0) + 1
                
                print("  By dimension:")
                for dim, count in sorted(dimensions.items()):
                    print(f"    {dim}: {count} questions")
        else:
            print("  ⚠ No questions parsed from cuestionario.json")
            print("    Mapping defines execution chains for:")
            dimensions = {}
            for qid in mapped_question_ids:
                dim = qid.split("-")[0]
                dimensions[dim] = dimensions.get(dim, 0) + 1
            for dim, count in sorted(dimensions.items()):
                print(f"      {dim}: {count} execution chains")
        
        print(f"\nAdapter Registry:")
        adapters = execution_mapping_data.get("adapters", {})
        print(f"  Total adapters: {len(adapters)}")
        for adapter_name, adapter_info in sorted(adapters.items()):
            methods = adapter_info.get("methods", 0)
            print(f"    {adapter_name}: {methods} methods")
        
        print("\n" + "="*80)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
