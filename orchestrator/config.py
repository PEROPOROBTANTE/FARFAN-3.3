"""
Configuration for FARFAN 3.0 Orchestrator
Immutable, deterministic, production-grade
"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal

@dataclass(frozen=True)
class ModuleConfig:
    """Configuration for individual processing modules"""
    name: str
    file_path: Path
    entry_function: str
    max_retries: int = 3
    timeout_seconds: int = 300
    required_dimensions: List[str] = field(default_factory=list)
    priority: int = 1  # 1=highest, 5=lowest

@dataclass(frozen=True)
class OrchestratorConfig:
    """Master configuration for the orchestrator"""

    # Paths
    base_dir: Path = Path(__file__).parent.parent
    cuestionario_path: Path = base_dir / "cuestionario.json"
    rubric_path: Path = base_dir / "rubric_scoring.json"
    output_dir: Path = base_dir / "output"
    logs_dir: Path = base_dir / "logs"

    # Module Registry (8 core modules)
    modules: Dict[str, ModuleConfig] = field(default_factory=lambda: {
        "analyzer_one": ModuleConfig(
            name="Analyzer One",
            file_path=Path("Analyzer_one.py"),
            entry_function="MunicipalAnalyzer",
            required_dimensions=["D1", "D2", "D3", "D4"],
            priority=1
        ),
        "causal_processor": ModuleConfig(
            name="Causal Processor",
            file_path=Path("causal_proccesor.py"),
            entry_function="PolicyDocumentAnalyzer",
            required_dimensions=["D1", "D2", "D3", "D4", "D5", "D6"],
            priority=1
        ),
        "contradiction_detector": ModuleConfig(
            name="Contradiction Detector",
            file_path=Path("contradiction_deteccion.py"),
            entry_function="PolicyContradictionDetectorV2",
            required_dimensions=["D6"],
            priority=2
        ),
        "dereck_beach": ModuleConfig(
            name="Derek Beach CDAF",
            file_path=Path("dereck_beach"),
            entry_function="CDAFFramework",
            required_dimensions=["D6"],
            priority=2
        ),
        "embedding_policy": ModuleConfig(
            name="Policy Embedder",
            file_path=Path("emebedding_policy.py"),
            entry_function="PolicyAnalysisEmbedder",
            required_dimensions=["D1", "D2", "D3", "D4"],
            priority=1
        ),
        "financial_viability": ModuleConfig(
            name="Financial Auditor",
            file_path=Path("financiero_viabilidad_tablas.py"),
            entry_function="FinancialAnalyzer",
            required_dimensions=["D1", "D3"],
            priority=3
        ),
        "policy_processor": ModuleConfig(
            name="Policy Processor",
            file_path=Path("policy_processor.py"),
            entry_function="IndustrialPolicyProcessor",
            required_dimensions=["D1", "D2", "D3", "D4", "D5"],
            priority=1
        ),
        "policy_segmenter": ModuleConfig(
            name="Policy Segmenter",
            file_path=Path("policy_segmenter.py"),
            entry_function="DocumentSegmenter",
            required_dimensions=["D1", "D2", "D3", "D4", "D5", "D6"],
            priority=1
        )
    })

    # Dimension to Module Mapping
    dimension_module_map: Dict[str, List[str]] = field(default_factory=lambda: {
        "D1": ["policy_processor", "causal_processor", "embedding_policy",
               "analyzer_one", "financial_viability", "policy_segmenter"],
        "D2": ["policy_processor", "causal_processor", "embedding_policy",
               "analyzer_one", "policy_segmenter"],
        "D3": ["policy_processor", "causal_processor", "embedding_policy",
               "analyzer_one", "financial_viability", "policy_segmenter"],
        "D4": ["policy_processor", "causal_processor", "embedding_policy",
               "analyzer_one", "policy_segmenter"],
        "D5": ["policy_processor", "causal_processor", "policy_segmenter"],
        "D6": ["causal_processor", "contradiction_detector", "dereck_beach",
               "policy_segmenter"]
    })

    # Circuit Breaker Configuration
    circuit_breaker_failure_threshold: int = 3
    circuit_breaker_timeout: int = 60
    circuit_breaker_half_open_timeout: int = 30

    # Execution Strategy
    execution_mode: Literal["parallel", "sequential", "hybrid"] = "hybrid"
    max_parallel_workers: int = 4

    # Report Configuration
    report_formats: List[str] = field(default_factory=lambda: ["json", "markdown", "pdf"])
    doctoral_explanation_min_words: int = 150
    doctoral_explanation_max_words: int = 300

    # Clusters Configuration (MESO level)
    clusters: Dict[str, List[str]] = field(default_factory=lambda: {
        "CLUSTER_1": ["P1"],  # Paz, Seguridad y Protección de Defensores
        "CLUSTER_2": ["P2", "P3", "P4", "P5"],  # Grupos poblacionales
        "CLUSTER_3": ["P6", "P7"],  # Tierra, Ambiente y Territorio
        "CLUSTER_4": ["P8", "P9", "P10"]  # DESC
    })

    # Quality Thresholds
    min_evidence_confidence: float = 0.5
    min_coherence_score: float = 0.6
    min_cluster_coverage: float = 0.7

    def __post_init__(self):
        """Create necessary directories"""
        object.__setattr__(self, 'output_dir', self.output_dir)
        object.__setattr__(self, 'logs_dir', self.logs_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.logs_dir.mkdir(exist_ok=True, parents=True)

# Global singleton
CONFIG = OrchestratorConfig()
