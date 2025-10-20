# mapping_loader.py - Execution Mapping Loader with DAG Validation
# coding=utf-8
"""
Execution Mapping Loader - Parse and Validate execution_mapping.yaml
====================================================================

Implements execution integrity layer between contract definitions and canary-based
regression detection. Ensures 300-question routing is structurally sound before test execution.

Features:
- Parse execution_mapping.yaml (execution chains, adapter registry, bindings)
- Build DAG from execution_chain steps using binding names as edges
- Validate bindings: exactly one producer per source reference
- Validate types: producer output types match consumer input types (via contract registry)
- Detect circular dependencies
- Fail-fast at startup with detailed MAPPING_CONFLICT diagnostics

Author: FARFAN 3.0 Integration Team
Version: 3.0.0
Python: 3.10+
"""

import logging
import yaml
from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import networkx as nx

logger = logging.getLogger(__name__)


# ============================================================================
# MAPPING CONFLICT ERROR TYPES
# ============================================================================

class ConflictType(Enum):
    """Types of mapping conflicts detected during validation"""
    DUPLICATE_PRODUCER = "duplicate_producer"
    MISSING_PRODUCER = "missing_producer"
    TYPE_MISMATCH = "type_mismatch"
    CIRCULAR_DEPENDENCY = "circular_dependency"
    INVALID_BINDING = "invalid_binding"
    UNKNOWN_ADAPTER = "unknown_adapter"
    MALFORMED_CHAIN = "malformed_chain"


@dataclass
class MappingConflict:
    """
    Represents a structural conflict in execution mapping
    
    Contains diagnostics for actionable remediation
    """
    conflict_type: ConflictType
    question_ids: List[str]
    description: str
    affected_bindings: List[str] = field(default_factory=list)
    type_mismatch_details: Optional[Dict[str, Any]] = None
    remediation: str = ""
    
    def __str__(self) -> str:
        """Format conflict for user-friendly display"""
        lines = [
            f"\n{'=' * 80}",
            f"MAPPING_CONFLICT: {self.conflict_type.value.upper()}",
            f"{'=' * 80}",
            f"\nQuestions Affected: {', '.join(self.question_ids)}",
            f"\nDescription: {self.description}"
        ]
        
        if self.affected_bindings:
            lines.append(f"\nAffected Bindings: {', '.join(self.affected_bindings)}")
        
        if self.type_mismatch_details:
            lines.append("\nType Mismatch Details:")
            for key, value in self.type_mismatch_details.items():
                lines.append(f"  {key}: {value}")
        
        if self.remediation:
            lines.append(f"\nRemediation:\n{self.remediation}")
        
        lines.append(f"{'=' * 80}\n")
        return "\n".join(lines)


class MappingValidationError(Exception):
    """Exception raised when mapping validation fails"""
    
    def __init__(self, conflicts: List[MappingConflict]):
        self.conflicts = conflicts
        message = f"Detected {len(conflicts)} mapping conflict(s):\n"
        message += "\n".join(str(c) for c in conflicts)
        super().__init__(message)


# ============================================================================
# CONTRACT REGISTRY INTERFACE (STUB)
# ============================================================================

@dataclass
class TypeContract:
    """Type contract for adapter method"""
    adapter: str
    method: str
    input_types: Dict[str, str]
    output_type: str


class ContractRegistry:
    """
    Contract Registry Interface - Integration point for type checking
    
    NOTE: This is a stub implementation. The actual contract registry should be
    implemented per the contract specification work item. This loader gracefully
    degrades if no registry is available.
    """
    
    def __init__(self):
        self.contracts: Dict[str, TypeContract] = {}
        self._initialized = False
        logger.warning(
            "ContractRegistry using stub implementation - "
            "type validation limited to YAML-specified types"
        )
    
    def register_contract(self, contract: TypeContract):
        """Register a type contract"""
        key = f"{contract.adapter}.{contract.method}"
        self.contracts[key] = contract
        self._initialized = True
    
    def get_contract(self, adapter: str, method: str) -> Optional[TypeContract]:
        """Get contract for adapter.method"""
        key = f"{adapter}.{method}"
        return self.contracts.get(key)
    
    def validate_type_compatibility(
        self,
        producer_type: str,
        consumer_type: str
    ) -> bool:
        """
        Check if producer type is compatible with consumer type
        
        Returns True if compatible, False otherwise
        """
        # Basic type compatibility checks
        if producer_type == consumer_type:
            return True
        
        # Handle generic types
        if self._is_generic_compatible(producer_type, consumer_type):
            return True
        
        return False
    
    def _is_generic_compatible(self, producer: str, consumer: str) -> bool:
        """Check generic type compatibility (e.g., List[Dict] vs List)"""
        # Strip whitespace
        producer = producer.strip()
        consumer = consumer.strip()
        
        # Handle List types
        if consumer.startswith("List") and producer.startswith("List"):
            return True
        
        # Handle Dict types
        if consumer.startswith("Dict") and producer.startswith("Dict"):
            return True
        
        # Handle Any
        if consumer == "Any" or producer == "Any":
            return True
        
        return False


# ============================================================================
# YAML MAPPING LOADER
# ============================================================================

class YAMLMappingLoader:
    """
    Loads and validates execution_mapping.yaml
    
    Parses:
    - Adapter registry (adapters section)
    - Execution chains (dimension questions)
    - Binding types and determinism flags
    - Builds DAG from execution steps
    - Validates structural integrity
    """
    
    def __init__(
        self,
        mapping_path: str = "config/execution_mapping.yaml",
        contract_registry: Optional[ContractRegistry] = None
    ):
        """
        Initialize mapping loader
        
        Args:
            mapping_path: Path to execution_mapping.yaml
            contract_registry: Optional contract registry for type checking
        """
        self.mapping_path = Path(mapping_path)
        self.contract_registry = contract_registry or ContractRegistry()
        
        # Parsed data structures
        self.raw_mapping: Dict[str, Any] = {}
        self.adapter_registry: Dict[str, Dict[str, Any]] = {}
        self.execution_chains: Dict[str, Dict[str, Any]] = {}
        self.binding_producers: Dict[str, List[Tuple[str, int]]] = {}  # binding -> [(question_id, step)]
        self.execution_dags: Dict[str, nx.DiGraph] = {}  # question_id -> DAG
        
        # Validation state
        self.conflicts: List[MappingConflict] = []
        self.validated = False
        
        logger.info(f"YAMLMappingLoader initialized for {mapping_path}")
    
    def load_and_validate(self) -> bool:
        """
        Load mapping and perform full validation
        
        Returns:
            True if validation passes
            
        Raises:
            MappingValidationError: If validation fails
        """
        logger.info("Loading execution mapping...")
        
        # Step 1: Load YAML
        self._load_yaml()
        
        # Step 2: Parse adapter registry
        self._parse_adapter_registry()
        
        # Step 3: Parse execution chains
        self._parse_execution_chains()
        
        # Step 4: Build DAGs
        self._build_dags()
        
        # Step 5: Validate bindings
        self._validate_bindings()
        
        # Step 6: Validate types
        self._validate_types()
        
        # Step 7: Detect circular dependencies
        self._detect_circular_dependencies()
        
        # Step 8: Check for conflicts
        if self.conflicts:
            logger.error(f"Validation failed with {len(self.conflicts)} conflict(s)")
            raise MappingValidationError(self.conflicts)
        
        self.validated = True
        logger.info("✓ Execution mapping validation PASSED")
        return True
    
    def _load_yaml(self):
        """Load YAML file"""
        if not self.mapping_path.exists():
            raise FileNotFoundError(f"Mapping file not found: {self.mapping_path}")
        
        with open(self.mapping_path, 'r', encoding='utf-8') as f:
            self.raw_mapping = yaml.safe_load(f)
        
        logger.info(f"Loaded YAML: {self.raw_mapping.get('version', 'unknown')} "
                   f"({self.raw_mapping.get('total_adapters', 0)} adapters)")
    
    def _parse_adapter_registry(self):
        """Parse adapter registry section"""
        adapters_section = self.raw_mapping.get('adapters', {})
        
        for adapter_name, adapter_info in adapters_section.items():
            self.adapter_registry[adapter_name] = {
                'adapter_class': adapter_info.get('adapter_class'),
                'methods': adapter_info.get('methods', 0),
                'specialization': adapter_info.get('specialization', ''),
                'sub_adapters': adapter_info.get('sub_adapters', [])
            }
        
        logger.info(f"Parsed {len(self.adapter_registry)} adapter registrations")
    
    def _parse_execution_chains(self):
        """Parse execution chains from all dimensions"""
        dimensions = [
            key for key in self.raw_mapping.keys()
            if key.startswith('D') and '_' in key
        ]
        
        for dimension in dimensions:
            dimension_data = self.raw_mapping[dimension]
            
            # Skip metadata
            if not isinstance(dimension_data, dict):
                continue
            
            for question_key, question_data in dimension_data.items():
                # Skip non-question entries
                if not question_key.startswith('Q'):
                    continue
                
                if not isinstance(question_data, dict):
                    continue
                
                execution_chain = question_data.get('execution_chain', [])
                
                if execution_chain:
                    question_id = f"{dimension}.{question_key}"
                    self.execution_chains[question_id] = {
                        'description': question_data.get('description', ''),
                        'execution_chain': execution_chain,
                        'aggregation': question_data.get('aggregation', {})
                    }
        
        logger.info(f"Parsed {len(self.execution_chains)} execution chains")
    
    def _build_dags(self):
        """Build DAG for each execution chain using binding names as edges"""
        for question_id, chain_data in self.execution_chains.items():
            dag = nx.DiGraph()
            execution_chain = chain_data['execution_chain']
            
            # Add nodes for each step
            for step in execution_chain:
                step_num = step.get('step')
                adapter = step.get('adapter')
                method = step.get('method')
                
                if not all([step_num, adapter, method]):
                    self.conflicts.append(MappingConflict(
                        conflict_type=ConflictType.MALFORMED_CHAIN,
                        question_ids=[question_id],
                        description=f"Step missing required fields: {step}",
                        remediation="Ensure each step has 'step', 'adapter', and 'method' fields"
                    ))
                    continue
                
                node_id = f"step_{step_num}_{adapter}.{method}"
                dag.add_node(
                    node_id,
                    step=step_num,
                    adapter=adapter,
                    method=method,
                    args=step.get('args', []),
                    returns=step.get('returns', {})
                )
            
            # Add edges based on binding dependencies
            for step in execution_chain:
                step_num = step.get('step')
                adapter = step.get('adapter')
                method = step.get('method')
                current_node = f"step_{step_num}_{adapter}.{method}"
                
                # Check args for source references
                args = step.get('args', [])
                for arg in args:
                    if isinstance(arg, dict):
                        source = arg.get('source')
                        if source and source != 'plan_text' and source != 'normalized_text':
                            # Find producer of this binding
                            producer_node = self._find_binding_producer(
                                question_id, source, step_num, execution_chain
                            )
                            if producer_node:
                                dag.add_edge(producer_node, current_node, binding=source)
            
            self.execution_dags[question_id] = dag
        
        logger.info(f"Built {len(self.execution_dags)} execution DAGs")
    
    def _find_binding_producer(
        self,
        question_id: str,
        binding_name: str,
        consumer_step: int,
        execution_chain: List[Dict]
    ) -> Optional[str]:
        """Find the step that produces a given binding"""
        for step in execution_chain:
            step_num = step.get('step')
            
            # Only look at steps before the consumer
            if step_num >= consumer_step:
                continue
            
            returns = step.get('returns', {})
            if returns.get('binding') == binding_name:
                adapter = step.get('adapter')
                method = step.get('method')
                
                # Track this producer
                if binding_name not in self.binding_producers:
                    self.binding_producers[binding_name] = []
                self.binding_producers[binding_name].append((question_id, step_num))
                
                return f"step_{step_num}_{adapter}.{method}"
        
        return None
    
    def _validate_bindings(self):
        """Validate that each binding has exactly one producer per question"""
        for question_id, chain_data in self.execution_chains.items():
            execution_chain = chain_data['execution_chain']
            
            # Track bindings within this question's chain
            question_bindings: Dict[str, List[int]] = {}
            
            # Collect all producers in this chain
            for step in execution_chain:
                step_num = step.get('step')
                returns = step.get('returns', {})
                binding = returns.get('binding')
                
                if binding:
                    if binding not in question_bindings:
                        question_bindings[binding] = []
                    question_bindings[binding].append(step_num)
            
            # Check for duplicate producers
            for binding, producer_steps in question_bindings.items():
                if len(producer_steps) > 1:
                    self.conflicts.append(MappingConflict(
                        conflict_type=ConflictType.DUPLICATE_PRODUCER,
                        question_ids=[question_id],
                        description=f"Binding '{binding}' has {len(producer_steps)} producers (steps: {producer_steps})",
                        affected_bindings=[binding],
                        remediation=f"Remove duplicate bindings. Each binding must have exactly one producer.\n"
                                  f"  Affected steps: {producer_steps}\n"
                                  f"  Solution: Use different binding names or merge steps."
                    ))
            
            # Check for missing producers
            for step in execution_chain:
                args = step.get('args', [])
                step_num = step.get('step')
                
                for arg in args:
                    if isinstance(arg, dict):
                        source = arg.get('source')
                        
                        # Skip special sources
                        if source in ['plan_text', 'normalized_text', 'entity_name',
                                     'extracted_tables', 'extracted_indicators',
                                     'institutional_mechanism', 'prior_assessments',
                                     'capacity_evidence', 'temporal_statements',
                                     'process_indicators', 'process_sequence',
                                     'process_mechanism', 'financial_data',
                                     'process_pillar']:
                            continue
                        
                        if source and source not in question_bindings:
                            self.conflicts.append(MappingConflict(
                                conflict_type=ConflictType.MISSING_PRODUCER,
                                question_ids=[question_id],
                                description=f"Step {step_num} references binding '{source}' but no producer exists",
                                affected_bindings=[source],
                                remediation=f"Add a step that produces binding '{source}' before step {step_num},\n"
                                          f"  or change the source reference to an existing binding."
                            ))
        
        if not any(c.conflict_type in [ConflictType.DUPLICATE_PRODUCER, ConflictType.MISSING_PRODUCER]
                   for c in self.conflicts):
            logger.info("✓ Binding validation passed")
    
    def _validate_types(self):
        """Validate producer output types match consumer input types"""
        for question_id, chain_data in self.execution_chains.items():
            execution_chain = chain_data['execution_chain']
            
            # Build type mapping: binding -> type
            binding_types: Dict[str, str] = {}
            
            for step in execution_chain:
                returns = step.get('returns', {})
                binding = returns.get('binding')
                return_type = returns.get('type')
                
                if binding and return_type:
                    binding_types[binding] = return_type
            
            # Check each consumer
            for step in execution_chain:
                args = step.get('args', [])
                step_num = step.get('step')
                adapter = step.get('adapter')
                method = step.get('method')
                
                for arg in args:
                    if isinstance(arg, dict):
                        source = arg.get('source')
                        expected_type = arg.get('type')
                        
                        if source and expected_type and source in binding_types:
                            producer_type = binding_types[source]
                            
                            # Check type compatibility
                            if not self.contract_registry.validate_type_compatibility(
                                producer_type, expected_type
                            ):
                                self.conflicts.append(MappingConflict(
                                    conflict_type=ConflictType.TYPE_MISMATCH,
                                    question_ids=[question_id],
                                    description=f"Type mismatch for binding '{source}' at step {step_num}",
                                    affected_bindings=[source],
                                    type_mismatch_details={
                                        'binding': source,
                                        'producer_type': producer_type,
                                        'consumer_type': expected_type,
                                        'consumer_step': step_num,
                                        'consumer_adapter': adapter,
                                        'consumer_method': method
                                    },
                                    remediation=f"Fix type mismatch:\n"
                                              f"  Producer returns: {producer_type}\n"
                                              f"  Consumer expects: {expected_type}\n"
                                              f"  Solution: Convert type in producer or update consumer signature."
                                ))
        
        if not any(c.conflict_type == ConflictType.TYPE_MISMATCH for c in self.conflicts):
            logger.info("✓ Type validation passed")
    
    def _detect_circular_dependencies(self):
        """Detect circular dependencies in execution DAGs"""
        for question_id, dag in self.execution_dags.items():
            if not nx.is_directed_acyclic_graph(dag):
                # Find cycles
                try:
                    cycles = list(nx.simple_cycles(dag))
                    cycle_descriptions = []
                    
                    for cycle in cycles:
                        cycle_descriptions.append(" -> ".join(cycle))
                    
                    self.conflicts.append(MappingConflict(
                        conflict_type=ConflictType.CIRCULAR_DEPENDENCY,
                        question_ids=[question_id],
                        description=f"Circular dependency detected in execution chain",
                        affected_bindings=[
                            edge_data.get('binding', 'unknown')
                            for _, _, edge_data in dag.edges(data=True)
                            if 'binding' in edge_data
                        ],
                        remediation=f"Break circular dependency:\n"
                                  f"  Detected cycles:\n    " +
                                  "\n    ".join(cycle_descriptions) +
                                  f"\n  Solution: Reorder steps or remove circular binding references."
                    ))
                except Exception as e:
                    logger.error(f"Error detecting cycles in {question_id}: {e}")
        
        if not any(c.conflict_type == ConflictType.CIRCULAR_DEPENDENCY for c in self.conflicts):
            logger.info("✓ Circular dependency check passed")
    
    def get_execution_chain(self, question_id: str) -> Optional[Dict[str, Any]]:
        """Get execution chain for a specific question"""
        return self.execution_chains.get(question_id)
    
    def get_execution_dag(self, question_id: str) -> Optional[nx.DiGraph]:
        """Get execution DAG for a specific question"""
        return self.execution_dags.get(question_id)
    
    def get_adapter_info(self, adapter_name: str) -> Optional[Dict[str, Any]]:
        """Get adapter registry information"""
        return self.adapter_registry.get(adapter_name)
    
    def get_all_bindings(self) -> Dict[str, List[Tuple[str, int]]]:
        """Get all bindings with their producers (question_id, step)"""
        return self.binding_producers.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get mapping statistics"""
        return {
            'total_adapters': len(self.adapter_registry),
            'total_execution_chains': len(self.execution_chains),
            'total_bindings': len(self.binding_producers),
            'total_steps': sum(
                len(chain['execution_chain'])
                for chain in self.execution_chains.values()
            ),
            'validated': self.validated,
            'conflicts_detected': len(self.conflicts)
        }


# ============================================================================
# STARTUP VALIDATOR
# ============================================================================

class MappingStartupValidator:
    """
    Validates execution mapping at application startup
    
    Fails fast with detailed diagnostics if conflicts detected
    """
    
    @staticmethod
    def validate_at_startup(
        mapping_path: str = "orchestrator/execution_mapping.yaml",
        contract_registry: Optional[ContractRegistry] = None
    ) -> YAMLMappingLoader:
        """
        Validate mapping at startup - FAIL FAST on errors
        
        Returns:
            YAMLMappingLoader instance if validation passes
            
        Raises:
            MappingValidationError: If validation fails (application should exit)
        """
        logger.info("=" * 80)
        logger.info("EXECUTION MAPPING VALIDATION - STARTUP CHECK")
        logger.info("=" * 80)
        
        try:
            loader = YAMLMappingLoader(mapping_path, contract_registry)
            loader.load_and_validate()
            
            stats = loader.get_statistics()
            logger.info("\nValidation Summary:")
            logger.info(f"  ✓ Adapters: {stats['total_adapters']}")
            logger.info(f"  ✓ Execution Chains: {stats['total_execution_chains']}")
            logger.info(f"  ✓ Total Steps: {stats['total_steps']}")
            logger.info(f"  ✓ Bindings: {stats['total_bindings']}")
            logger.info("=" * 80)
            
            return loader
            
        except MappingValidationError as e:
            logger.error("\n" + "!" * 80)
            logger.error("VALIDATION FAILED - APPLICATION CANNOT START")
            logger.error("!" * 80)
            logger.error(str(e))
            raise
        
        except Exception as e:
            logger.error(f"\nUnexpected error during validation: {e}", exc_info=True)
            raise


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    'YAMLMappingLoader',
    'MappingStartupValidator',
    'MappingValidationError',
    'MappingConflict',
    'ConflictType',
    'ContractRegistry',
    'TypeContract'
]


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Validate at startup
        loader = MappingStartupValidator.validate_at_startup()
        
        # Example: Get execution chain for a question
        chain = loader.get_execution_chain("D1_INSUMOS.Q1_Baseline_Identification")
        if chain:
            print(f"\nExample Chain: {len(chain['execution_chain'])} steps")
        
        # Example: Get DAG
        dag = loader.get_execution_dag("D1_INSUMOS.Q1_Baseline_Identification")
        if dag:
            print(f"DAG: {dag.number_of_nodes()} nodes, {dag.number_of_edges()} edges")
        
    except MappingValidationError as e:
        print("\nVALIDATION FAILED - See diagnostics above")
        exit(1)
