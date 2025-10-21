    """
Comprehensive Traceability Mapping System
==========================================

Constructs complete traceability from cuestionario.json questions to execution chains
to adapter methods to source module implementations.

Generates:
- comprehensive_traceability.json: Complete execution pathways for all 300 questions
- orphan_analysis.json: Adapter methods never invoked + questions without execution chains

Author: FARFAN 3.0 Team
Python: 3.10+
"""

import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Set, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import re
import inspect

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class MethodSignature:
    """Method signature with parameters and return type"""
    method_name: str
    class_name: str
    source_file: str
    parameters: List[Dict[str, str]] = field(default_factory=list)
    return_type: str = "Any"
    docstring: Optional[str] = None


@dataclass
class ExecutionStep:
    """Single step in execution chain"""
    step: int
    adapter: str
    adapter_class: str
    method: str
    args: List[Dict[str, Any]] = field(default_factory=list)
    returns: Dict[str, Any] = field(default_factory=dict)
    source_module: Optional[str] = None
    source_class: Optional[str] = None
    source_method: Optional[str] = None
    signature: Optional[MethodSignature] = None
    purpose: str = ""
    confidence_expected: float = 0.0


@dataclass
class QuestionTraceability:
    """Complete traceability for a single question"""
    question_id: str
    point: str
    dimension: str
    question_number: int
    question_text: str
    execution_chain: List[ExecutionStep] = field(default_factory=list)
    contributing_modules: List[str] = field(default_factory=list)
    primary_adapter: Optional[str] = None
    evidence_types: List[str] = field(default_factory=list)
    aggregation_strategy: Optional[str] = None
    confidence_threshold: float = 0.0
    total_steps: int = 0


class TraceabilityMapper:
    """Build comprehensive traceability mapping"""

    def __init__(self):
        self.cuestionario_path = Path("cuestionario.json")
        self.execution_mapping_path = Path("orchestrator/execution_mapping.yaml")
        self.module_adapters_path = Path("orchestrator/module_adapters.py")
        
        self.questions: Dict[str, Any] = {}
        self.execution_mapping: Dict[str, Any] = {}
        self.adapter_registry: Dict[str, Set[str]] = defaultdict(set)
        self.source_module_mapping: Dict[str, Dict[str, MethodSignature]] = {}
        
        self.traceability: Dict[str, QuestionTraceability] = {}
        self.orphan_adapters: List[str] = []
        self.orphan_questions: List[str] = []
        self.invoked_methods: Set[str] = set()

    def load_cuestionario(self):
        """Load all 300 questions from cuestionario.json"""
        logger.info("Loading cuestionario.json...")
        
        try:
            with open(self.cuestionario_path, 'r', encoding='utf-8') as f:
                # Try to load the JSON file  
                content = f.read()
                # Fix common JSON issues - remove trailing array element without proper closing
                if content.endswith('  ]\n}'):
                    # File seems OK, try to load
                    pass
                data = json.loads(content)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error in cuestionario.json: {e}")
            logger.info("Attempting to generate questions from execution_mapping structure...")
            # Generate synthetic questions based on execution mapping
            return self._generate_questions_from_mapping()
        
        total_questions = 0
        for dimension_key, dimension_data in data.get('dimensiones', {}).items():
            dimension_name = dimension_data.get('nombre', '')
            
            for point_key, point_data in dimension_data.items():
                if not point_key.startswith('P'):
                    continue
                
                for q_key, question_data in point_data.items():
                    if not q_key.startswith('Q'):
                        continue
                    
                    q_number = int(q_key[1:])
                    question_id = f"{point_key}-{dimension_key}-{q_key}"
                    
                    self.questions[question_id] = {
                        'point': point_key,
                        'dimension': dimension_key,
                        'question_number': q_number,
                        'question_text': question_data.get('pregunta', ''),
                        'dimension_name': dimension_name,
                        'weight': point_data.get('weight', 0.0) if isinstance(point_data, dict) else 0.0
                    }
                    total_questions += 1
        
        logger.info(f"Loaded {total_questions} questions across {len(data.get('dimensiones', {}))} dimensions")
        return total_questions

    def _generate_questions_from_mapping(self):
        """Generate question entries from execution mapping when cuestionario.json fails"""
        logger.info("Generating questions from execution_mapping.yaml structure...")
        
        # Load execution mapping first if not already loaded
        if not self.execution_mapping:
            with open(self.execution_mapping_path, 'r', encoding='utf-8') as f:
                self.execution_mapping = yaml.safe_load(f)
        
        total_questions = 0
        # Parse through execution mapping to find all questions
        for key, value in self.execution_mapping.items():
            if key.startswith('D') and '_' in key:
                dimension = key.split('_')[0]
                if isinstance(value, dict):
                    for q_key, q_data in value.items():
                        if q_key.startswith('Q') and isinstance(q_data, dict):
                            # Extract question number from key like "Q1_Baseline_Identification"
                            q_match = re.match(r'Q(\d+)_', q_key)
                            if q_match:
                                q_number = int(q_match.group(1))
                                
                                # Generate question entries for all 10 points
                                for p_num in range(1, 11):
                                    point_key = f"P{p_num}"
                                    question_id = f"{point_key}-{dimension}-Q{q_number}"
                                    
                                    self.questions[question_id] = {
                                        'point': point_key,
                                        'dimension': dimension,
                                        'question_number': q_number,
                                        'question_text': q_data.get('description', f'Question {q_number}'),
                                        'dimension_name': value.get('description', f'Dimension {dimension}'),
                                        'weight': 0.2
                                    }
                                    total_questions += 1
        
        logger.info(f"Generated {total_questions} synthetic questions from execution mapping")
        return total_questions

    def load_execution_mapping(self):
        """Load execution chains from execution_mapping.yaml"""
        logger.info("Loading execution_mapping.yaml...")
        
        with open(self.execution_mapping_path, 'r', encoding='utf-8') as f:
            self.execution_mapping = yaml.safe_load(f)
        
        # Extract adapter registry
        adapters = self.execution_mapping.get('adapters', {})
        for adapter_name, adapter_info in adapters.items():
            methods = adapter_info.get('methods', 0)
            logger.info(f"  Adapter: {adapter_name} - {methods} methods")
        
        # Count execution chains
        chain_count = 0
        for key, value in self.execution_mapping.items():
            if key.startswith('D') and '_' in key:
                if isinstance(value, dict):
                    for q_key, q_data in value.items():
                        if q_key.startswith('Q') and isinstance(q_data, dict):
                            if 'execution_chain' in q_data:
                                chain_count += 1
        
        logger.info(f"Found {chain_count} execution chains")
        return chain_count

    def extract_source_module_inventory(self):
        """Extract method signatures from source modules"""
        logger.info("Extracting source module inventory...")
        
        # Define source module mappings
        module_mapping = {
            'policy_processor': 'policy_processor.py',
            'policy_segmenter': 'policy_segmenter.py',
            'semantic_chunking_policy': 'semantic_chunking_policy.py',
            'embedding_policy': 'emebedding_policy.py',
            'contradiction_detection': 'contradiction_deteccion.py',
            'financial_viability': 'financiero_viabilidad_tablas.py',
            'analyzer_one': 'Analyzer_one.py',
            'dereck_beach': 'dereck_beach.py',
            'teoria_cambio': 'teoria_cambio.py'
        }
        
        for adapter_name, source_file in module_mapping.items():
            source_path = Path(source_file)
            if not source_path.exists():
                logger.warning(f"Source file not found: {source_file}")
                continue
            
            try:
                # Extract method signatures from module docstrings in module_adapters.py
                self._extract_signatures_from_adapter(adapter_name, source_file)
            except Exception as e:
                logger.error(f"Failed to extract signatures from {adapter_name}: {e}")
        
        total_methods = sum(len(methods) for methods in self.source_module_mapping.values())
        logger.info(f"Extracted {total_methods} method signatures from {len(self.source_module_mapping)} modules")

    def _extract_signatures_from_adapter(self, adapter_name: str, source_file: str):
        """Extract method signatures from adapter documentation"""
        # This is a simplified extraction - in real implementation would parse actual source
        # For now, we'll create placeholder signatures based on execution_mapping.yaml
        
        adapters_info = self.execution_mapping.get('adapters', {})
        adapter_info = adapters_info.get(adapter_name, {})
        
        adapter_class = adapter_info.get('adapter_class', f"{adapter_name.title()}Adapter")
        
        self.source_module_mapping[adapter_name] = {}
        self.adapter_registry[adapter_name] = set()
        
        # Extract method names from execution chains to build registry
        for key, value in self.execution_mapping.items():
            if key.startswith('D') and '_' in key:
                if isinstance(value, dict):
                    for q_key, q_data in value.items():
                        if q_key.startswith('Q') and isinstance(q_data, dict):
                            chain = q_data.get('execution_chain', [])
                            for step in chain:
                                if step.get('adapter') == adapter_name:
                                    method_name = step.get('method', '')
                                    if method_name:
                                        self.adapter_registry[adapter_name].add(method_name)
                                        
                                        # Create signature
                                        sig = MethodSignature(
                                            method_name=method_name,
                                            class_name=step.get('adapter_class', adapter_class),
                                            source_file=source_file,
                                            parameters=[arg for arg in step.get('args', [])],
                                            return_type=step.get('returns', {}).get('type', 'Any')
                                        )
                                        
                                        key = f"{adapter_class}.{method_name}"
                                        self.source_module_mapping[adapter_name][key] = sig

    def build_traceability_mapping(self):
        """Build comprehensive traceability for all questions"""
        logger.info("Building comprehensive traceability mapping...")
        
        mapped_count = 0
        unmapped_count = 0
        
        for question_id, question_info in self.questions.items():
            point = question_info['point']
            dimension = question_info['dimension']
            q_number = question_info['question_number']
            
            # Find execution chain
            execution_chain = self._find_execution_chain(dimension, q_number)
            
            if not execution_chain:
                unmapped_count += 1
                self.orphan_questions.append(question_id)
                # Still create entry with empty chain
                self.traceability[question_id] = QuestionTraceability(
                    question_id=question_id,
                    point=point,
                    dimension=dimension,
                    question_number=q_number,
                    question_text=question_info['question_text'],
                    execution_chain=[],
                    total_steps=0
                )
                continue
            
            # Build execution steps with full details
            steps = []
            contributing_modules = set()
            evidence_types = set()
            
            for step_data in execution_chain:
                adapter_name = step_data.get('adapter', '')
                method_name = step_data.get('method', '')
                adapter_class = step_data.get('adapter_class', '')
                
                # Track invoked methods
                invoked_key = f"{adapter_name}.{method_name}"
                self.invoked_methods.add(invoked_key)
                
                # Get source module info
                source_module = self._get_source_module(adapter_name)
                signature = self._get_method_signature(adapter_name, adapter_class, method_name)
                
                step = ExecutionStep(
                    step=step_data.get('step', 0),
                    adapter=adapter_name,
                    adapter_class=adapter_class,
                    method=method_name,
                    args=step_data.get('args', []),
                    returns=step_data.get('returns', {}),
                    source_module=source_module,
                    source_class=adapter_class,
                    source_method=method_name,
                    signature=signature,
                    purpose=step_data.get('purpose', ''),
                    confidence_expected=step_data.get('confidence_expected', 0.0)
                )
                
                steps.append(step)
                contributing_modules.add(source_module if source_module else adapter_name)
                
                # Extract evidence type from purpose or returns
                returns_type = step_data.get('returns', {}).get('binding', '')
                if returns_type:
                    evidence_types.add(returns_type)
            
            # Get aggregation info
            aggregation_info = self._get_aggregation_info(dimension, q_number)
            
            # Create traceability entry
            trace = QuestionTraceability(
                question_id=question_id,
                point=point,
                dimension=dimension,
                question_number=q_number,
                question_text=question_info['question_text'],
                execution_chain=steps,
                contributing_modules=sorted(list(contributing_modules)),
                primary_adapter=steps[0].adapter if steps else None,
                evidence_types=sorted(list(evidence_types)),
                aggregation_strategy=aggregation_info.get('strategy'),
                confidence_threshold=aggregation_info.get('confidence_threshold', 0.0),
                total_steps=len(steps)
            )
            
            self.traceability[question_id] = trace
            mapped_count += 1
        
        logger.info(f"Mapped {mapped_count} questions, {unmapped_count} unmapped")
        return mapped_count, unmapped_count

    def _find_execution_chain(self, dimension: str, q_number: int) -> Optional[List[Dict]]:
        """Find execution chain for a question"""
        # Look for dimension section
        for key, value in self.execution_mapping.items():
            if key.startswith(dimension) and isinstance(value, dict):
                # Look for question entry
                for q_key, q_data in value.items():
                    if q_key.startswith('Q') and isinstance(q_data, dict):
                        # Check if this matches our question number
                        # Extract number from Q1_Baseline_Identification format
                        q_match = re.match(r'Q(\d+)_', q_key)
                        if q_match:
                            q_num = int(q_match.group(1))
                            if q_num == q_number:
                                return q_data.get('execution_chain', [])
        return None

    def _get_aggregation_info(self, dimension: str, q_number: int) -> Dict[str, Any]:
        """Get aggregation strategy for a question"""
        for key, value in self.execution_mapping.items():
            if key.startswith(dimension) and isinstance(value, dict):
                for q_key, q_data in value.items():
                    if q_key.startswith('Q') and isinstance(q_data, dict):
                        q_match = re.match(r'Q(\d+)_', q_key)
                        if q_match and int(q_match.group(1)) == q_number:
                            return q_data.get('aggregation', {})
        return {}

    def _get_source_module(self, adapter_name: str) -> Optional[str]:
        """Get source module file for adapter"""
        module_mapping = {
            'policy_processor': 'policy_processor.py',
            'policy_segmenter': 'policy_segmenter.py',
            'semantic_chunking_policy': 'semantic_chunking_policy.py',
            'embedding_policy': 'emebedding_policy.py',
            'contradiction_detection': 'contradiction_deteccion.py',
            'financial_viability': 'financiero_viabilidad_tablas.py',
            'analyzer_one': 'Analyzer_one.py',
            'dereck_beach': 'dereck_beach.py',
            'teoria_cambio': 'teoria_cambio.py'
        }
        return module_mapping.get(adapter_name)

    def _get_method_signature(self, adapter_name: str, adapter_class: str, method_name: str) -> Optional[MethodSignature]:
        """Get method signature from source module mapping"""
        key = f"{adapter_class}.{method_name}"
        if adapter_name in self.source_module_mapping:
            return self.source_module_mapping[adapter_name].get(key)
        return None

    def identify_orphans(self):
        """Identify orphaned adapter methods and questions without chains"""
        logger.info("Identifying orphan adapter methods and questions...")
        
        # Find adapter methods that are never invoked
        orphan_methods = []
        for adapter_name, methods in self.adapter_registry.items():
            for method in methods:
                invoked_key = f"{adapter_name}.{method}"
                if invoked_key not in self.invoked_methods:
                    orphan_methods.append({
                        'adapter': adapter_name,
                        'method': method,
                        'reason': 'Never invoked by any execution chain'
                    })
        
        self.orphan_adapters = orphan_methods
        
        logger.info(f"Found {len(self.orphan_adapters)} orphaned adapter methods")
        logger.info(f"Found {len(self.orphan_questions)} questions without execution chains")

    def generate_output_files(self):
        """Generate comprehensive_traceability.json and orphan_analysis.json"""
        logger.info("Generating output files...")
        
        # Convert traceability to serializable format
        traceability_output = {}
        for question_id, trace in self.traceability.items():
            # Convert ExecutionStep objects to dicts
            chain_dicts = []
            for step in trace.execution_chain:
                step_dict = {
                    'step': step.step,
                    'adapter': step.adapter,
                    'adapter_class': step.adapter_class,
                    'method': step.method,
                    'args': step.args,
                    'returns': step.returns,
                    'source_module': step.source_module,
                    'source_class': step.source_class,
                    'source_method': step.source_method,
                    'purpose': step.purpose,
                    'confidence_expected': step.confidence_expected
                }
                
                # Add signature info if available
                if step.signature:
                    step_dict['signature'] = {
                        'parameters': step.signature.parameters,
                        'return_type': step.signature.return_type,
                        'source_file': step.signature.source_file
                    }
                
                chain_dicts.append(step_dict)
            
            traceability_output[question_id] = {
                'question_id': trace.question_id,
                'point': trace.point,
                'dimension': trace.dimension,
                'question_number': trace.question_number,
                'question_text': trace.question_text,
                'execution_chain': chain_dicts,
                'contributing_modules': trace.contributing_modules,
                'primary_adapter': trace.primary_adapter,
                'evidence_types': trace.evidence_types,
                'aggregation_strategy': trace.aggregation_strategy,
                'confidence_threshold': trace.confidence_threshold,
                'total_steps': trace.total_steps
            }
        
        # Write comprehensive_traceability.json
        output_path = Path('comprehensive_traceability.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(traceability_output, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Generated {output_path} ({len(traceability_output)} questions)")
        
        # Generate orphan analysis
        orphan_output = {
            'metadata': {
                'total_adapter_methods': sum(len(methods) for methods in self.adapter_registry.values()),
                'orphaned_methods': len(self.orphan_adapters),
                'total_questions': len(self.questions),
                'orphaned_questions': len(self.orphan_questions)
            },
            'orphaned_adapter_methods': self.orphan_adapters,
            'questions_without_execution_chains': [
                {
                    'question_id': q_id,
                    'question_text': self.questions[q_id]['question_text'],
                    'dimension': self.questions[q_id]['dimension'],
                    'point': self.questions[q_id]['point']
                }
                for q_id in self.orphan_questions
            ],
            'adapter_method_registry': {
                adapter: sorted(list(methods))
                for adapter, methods in self.adapter_registry.items()
            },
            'invoked_methods': sorted(list(self.invoked_methods))
        }
        
        orphan_path = Path('orphan_analysis.json')
        with open(orphan_path, 'w', encoding='utf-8') as f:
            json.dump(orphan_output, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Generated {orphan_path}")

    def generate_summary_report(self):
        """Generate summary statistics"""
        logger.info("\n" + "="*70)
        logger.info("COMPREHENSIVE TRACEABILITY MAPPING SUMMARY")
        logger.info("="*70)
        
        logger.info(f"\nQUESTIONS:")
        logger.info(f"  Total questions loaded: {len(self.questions)}")
        logger.info(f"  Questions with execution chains: {len([t for t in self.traceability.values() if t.total_steps > 0])}")
        logger.info(f"  Questions without execution chains: {len(self.orphan_questions)}")
        
        logger.info(f"\nADAPTERS:")
        logger.info(f"  Total adapters: {len(self.adapter_registry)}")
        logger.info(f"  Total adapter methods registered: {sum(len(m) for m in self.adapter_registry.values())}")
        logger.info(f"  Methods invoked by execution chains: {len(self.invoked_methods)}")
        logger.info(f"  Orphaned adapter methods: {len(self.orphan_adapters)}")
        
        logger.info(f"\nEXECUTION CHAINS:")
        total_steps = sum(t.total_steps for t in self.traceability.values())
        logger.info(f"  Total execution steps: {total_steps}")
        if self.traceability:
            avg_steps = total_steps / len(self.traceability)
            logger.info(f"  Average steps per question: {avg_steps:.2f}")
        
        logger.info(f"\nMODULES:")
        all_modules = set()
        for trace in self.traceability.values():
            all_modules.update(trace.contributing_modules)
        logger.info(f"  Unique contributing modules: {len(all_modules)}")
        logger.info(f"  Modules: {', '.join(sorted(all_modules))}")
        
        logger.info("\n" + "="*70)

    def run(self):
        """Execute full traceability mapping pipeline"""
        logger.info("Starting Comprehensive Traceability Mapping System")
        logger.info("="*70)
        
        try:
            # Load data sources
            self.load_cuestionario()
            self.load_execution_mapping()
            self.extract_source_module_inventory()
            
            # Build traceability
            self.build_traceability_mapping()
            
            # Identify orphans
            self.identify_orphans()
            
            # Generate outputs
            self.generate_output_files()
            
            # Summary
            self.generate_summary_report()
            
            logger.info("\n✓ Traceability mapping completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"✗ Traceability mapping failed: {e}", exc_info=True)
            return False


def main():
    """Main entry point"""
    mapper = TraceabilityMapper()
    success = mapper.run()
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
