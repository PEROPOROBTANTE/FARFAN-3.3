# coding=utf-8
"""
Question Router - Routes Questions to Module Handlers
======================================================

Loads responsibility_map.json, validates question IDs, routes questions to 
mapped module:Class.method handlers, and raises exceptions for unmapped questions.

Author: FARFAN 3.0 Team
Version: 3.0.0
Python: 3.10+
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RouteInfo:
    """Route information for a question"""
    question_id: str
    module_name: str
    class_name: str
    method_name: str
    dimension: str
    confidence: float = 0.8


class QuestionRoutingError(Exception):
    """Raised when a question cannot be routed"""
    pass


class QuestionRouter:
    """
    Routes questions to module handlers based on responsibility map
    
    Features:
    - Loads and validates responsibility_map.json
    - Maps question IDs to module:Class.method handlers
    - Validates question IDs against mapping
    - Raises exceptions for unmapped questions
    """

    def __init__(self, responsibility_map_path: Optional[Path] = None):
        """
        Initialize question router
        
        Args:
            responsibility_map_path: Path to responsibility map (default: config/responsibility_map.json)
        """
        self.responsibility_map_path = (
            responsibility_map_path or 
            Path(__file__).parent.parent / "config" / "responsibility_map.json"
        )
        self.responsibility_map: Dict[str, Any] = {}
        self._load_responsibility_map()
        
        logger.info(
            f"QuestionRouter initialized with {len(self.responsibility_map)} mappings"
        )

    def _load_responsibility_map(self):
        """Load and validate responsibility_map.json"""
        if not self.responsibility_map_path.exists():
            logger.warning(
                f"responsibility_map.json not found at {self.responsibility_map_path}, "
                f"using fallback mapping"
            )
            self._create_fallback_mapping()
            return
        
        try:
            with open(self.responsibility_map_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict) and "mappings" in data:
                self.responsibility_map = data["mappings"]
            elif isinstance(data, dict):
                self.responsibility_map = data
            else:
                raise ValueError("Invalid responsibility_map.json structure")
            
            logger.info(f"Loaded responsibility map from {self.responsibility_map_path}")
            
        except Exception as e:
            logger.error(f"Failed to load responsibility_map.json: {e}")
            self._create_fallback_mapping()

    def _create_fallback_mapping(self):
        """Create fallback dimension-based mapping"""
        self.responsibility_map = {
            "D1": {
                "module": "policy_processor",
                "class": "IndustrialPolicyProcessor",
                "method": "process"
            },
            "D2": {
                "module": "causal_proccesor",
                "class": "PolicyDocumentAnalyzer",
                "method": "analyze_document"
            },
            "D3": {
                "module": "analyzer_one",
                "class": "MunicipalAnalyzer",
                "method": "analyze"
            },
            "D4": {
                "module": "teoria_cambio",
                "class": "ModulosTeoriaCambio",
                "method": "analizar_teoria_cambio"
            },
            "D5": {
                "module": "dereck_beach",
                "class": "DerekBeachAnalyzer",
                "method": "analyze_causal_chain"
            },
            "D6": {
                "module": "teoria_cambio",
                "class": "ModulosTeoriaCambio",
                "method": "validar_coherencia_causal"
            }
        }
        logger.info("Using fallback dimension-based mapping")

    def route_question(self, question_id: str) -> RouteInfo:
        """
        Route a question to its handler
        
        Args:
            question_id: Question ID (e.g., "P1-D1-Q1" or "D1-Q1")
            
        Returns:
            RouteInfo with module, class, and method
            
        Raises:
            QuestionRoutingError: If question cannot be routed
        """
        if not self.validate_question_id(question_id):
            raise QuestionRoutingError(f"Invalid question ID format: {question_id}")
        
        dimension = self._extract_dimension(question_id)
        
        if question_id in self.responsibility_map:
            mapping = self.responsibility_map[question_id]
        elif dimension in self.responsibility_map:
            mapping = self.responsibility_map[dimension]
        else:
            raise QuestionRoutingError(
                f"No mapping found for question {question_id} (dimension {dimension})"
            )
        
        return RouteInfo(
            question_id=question_id,
            module_name=mapping["module"],
            class_name=mapping["class"],
            method_name=mapping["method"],
            dimension=dimension,
            confidence=mapping.get("confidence", 0.8)
        )

    def validate_question_id(self, question_id: str) -> bool:
        """
        Validate question ID format
        
        Accepts:
        - Full format: P1-D1-Q1
        - Short format: D1-Q1
        - Dimension only: D1
        
        Args:
            question_id: Question ID to validate
            
        Returns:
            True if valid format, False otherwise
        """
        import re
        
        patterns = [
            r'^P\d+-D[1-6]-Q[1-5]$',  # P1-D1-Q1
            r'^D[1-6]-Q[1-5]$',         # D1-Q1
            r'^D[1-6]$'                  # D1
        ]
        
        return any(re.match(pattern, question_id) for pattern in patterns)

    def get_mapped_questions(self) -> List[str]:
        """Get list of all mapped question IDs"""
        return list(self.responsibility_map.keys())

    def is_question_mapped(self, question_id: str) -> bool:
        """Check if a question has a mapping"""
        dimension = self._extract_dimension(question_id)
        return question_id in self.responsibility_map or dimension in self.responsibility_map

    def get_routes_by_module(self, module_name: str) -> List[str]:
        """Get all question IDs routed to a specific module"""
        routes = []
        for qid, mapping in self.responsibility_map.items():
            if mapping.get("module") == module_name:
                routes.append(qid)
        return routes

    def get_routes_by_dimension(self, dimension: str) -> Dict[str, Any]:
        """Get routing information for a dimension"""
        if dimension in self.responsibility_map:
            return {dimension: self.responsibility_map[dimension]}
        
        return {
            qid: mapping
            for qid, mapping in self.responsibility_map.items()
            if self._extract_dimension(qid) == dimension
        }

    def _extract_dimension(self, question_id: str) -> str:
        """Extract dimension from question ID"""
        import re
        match = re.search(r'D[1-6]', question_id)
        return match.group(0) if match else ""


if __name__ == "__main__":
    router = QuestionRouter()
    
    print("=" * 60)
    print("Question Router")
    print("=" * 60)
    print(f"\nMapped questions: {len(router.get_mapped_questions())}")
    print(f"Responsibility map path: {router.responsibility_map_path}")
    
    test_ids = ["P1-D1-Q1", "D2-Q3", "D5"]
    print("\nTest routing:")
    for qid in test_ids:
        try:
            route = router.route_question(qid)
            print(f"  {qid} -> {route.module_name}:{route.class_name}.{route.method_name}")
        except QuestionRoutingError as e:
            print(f"  {qid} -> ERROR: {e}")
