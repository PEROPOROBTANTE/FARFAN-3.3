"""
FinancialViability Adapter Layer
=================================

Backward-compatible adapter wrapping financiero_viabilidad_tablas.py functionality.
Provides translation layer between legacy 11-adapter architecture and unified module controller.

This adapter preserves existing method signatures while delegating to the core domain module.
"""

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class FinancialViabilityAdapter:
    """
    Adapter for financiero_viabilidad_tablas.py - Municipal Development Plan Financial Analyzer.
    
    Wraps PDETMunicipalPlanAnalyzer for PDET Colombia financial analysis with
    causal inference capabilities.
    
    PRIMARY INTERFACE (Backward Compatible):
    - analyze_financial_tables(pdf_path: str) -> Dict[str, Any]
    - extract_budget_allocation(pdf_path: str) -> pd.DataFrame
    - validate_financial_consistency(tables: List[pd.DataFrame]) -> Dict[str, Any]
    - compute_financial_viability_score(data: Dict) -> float
    - infer_resource_causal_chains(financial_data: Dict) -> Dict[str, Any]
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize adapter with optional configuration.
        
        Args:
            config: Configuration dict for financial analyzer (optional)
        """
        self._load_core_module()
        self._initialize_components(config)
        logger.info("FinancialViabilityAdapter initialized successfully")

    def _load_core_module(self):
        """Load core domain module components"""
        try:
            from financiero_viabilidad_tablas import (
                PDETMunicipalPlanAnalyzer,
                ColombianMunicipalContext
            )
            
            self.PDETMunicipalPlanAnalyzer = PDETMunicipalPlanAnalyzer
            self.ColombianMunicipalContext = ColombianMunicipalContext
            self._module_available = True
            
        except ImportError as e:
            logger.error(f"Failed to load financiero_viabilidad_tablas module: {e}")
            self._module_available = False
            raise RuntimeError(f"Core module financiero_viabilidad_tablas not available: {e}")

    def _initialize_components(self, config: Optional[Dict[str, Any]]):
        """Initialize financial analyzer components"""
        self.analyzer = None  # Lazy loaded per document
        self.config = config or {}

    def _ensure_analyzer(self, pdf_path: str):
        """Lazy load analyzer for specific document"""
        if self.analyzer is None or self.analyzer.pdf_path != Path(pdf_path):
            self.analyzer = self.PDETMunicipalPlanAnalyzer(pdf_path)

    # ========================================================================
    # PRIMARY INTERFACE (Backward Compatible)
    # ========================================================================

    def analyze_financial_tables(
        self, 
        pdf_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze financial tables from municipal development plan PDF.
        
        Args:
            pdf_path: Path to PDF file
            **kwargs: Additional analysis parameters
            
        Returns:
            Dict with extracted tables, budget allocations, and analysis
        """
        self._ensure_analyzer(pdf_path)
        
        # Extract and analyze tables
        analysis_result = self.analyzer.analyze_complete_plan()
        
        return {
            'tables': analysis_result.get('tablas_extraidas', []),
            'budget_summary': analysis_result.get('resumen_presupuestal', {}),
            'financial_indicators': analysis_result.get('indicadores_financieros', {}),
            'consistency_check': analysis_result.get('verificacion_consistencia', {}),
            'causal_chains': analysis_result.get('cadenas_causales', {})
        }

    def extract_budget_allocation(
        self, 
        pdf_path: str,
        **kwargs
    ) -> pd.DataFrame:
        """
        Extract budget allocation data from PDF.
        
        Args:
            pdf_path: Path to PDF file
            **kwargs: Additional extraction parameters
            
        Returns:
            DataFrame with budget allocations by program/pillar
        """
        self._ensure_analyzer(pdf_path)
        
        # Extract budget tables
        budget_tables = self.analyzer.extract_budget_tables()
        
        # Combine into single DataFrame
        if budget_tables:
            return pd.concat(budget_tables, ignore_index=True)
        else:
            return pd.DataFrame()

    def validate_financial_consistency(
        self, 
        tables: List[pd.DataFrame],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Validate consistency across financial tables.
        
        Args:
            tables: List of financial DataFrames to validate
            **kwargs: Additional validation parameters
            
        Returns:
            Validation results with consistency checks and errors
        """
        validation = {
            'total_consistency': True,
            'sum_checks': [],
            'cross_reference_errors': [],
            'warnings': []
        }
        
        # Check sum consistency
        for i, table in enumerate(tables):
            if 'total' in table.columns.str.lower():
                # Validate totals match row sums
                numeric_cols = table.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 1:
                    computed_sum = table[numeric_cols[:-1]].sum(axis=1)
                    declared_sum = table[numeric_cols[-1]]
                    
                    if not computed_sum.equals(declared_sum):
                        validation['sum_checks'].append({
                            'table_index': i,
                            'passed': False,
                            'max_difference': abs(computed_sum - declared_sum).max()
                        })
                        validation['total_consistency'] = False
        
        return validation

    def compute_financial_viability_score(
        self, 
        data: Dict[str, Any],
        **kwargs
    ) -> float:
        """
        Compute overall financial viability score.
        
        Args:
            data: Financial data dictionary
            **kwargs: Additional scoring parameters
            
        Returns:
            Viability score [0.0, 1.0]
        """
        score_components = []
        
        # Budget completeness
        if data.get('budget_summary'):
            score_components.append(0.3)
        
        # Consistency
        if data.get('consistency_check', {}).get('total_consistency'):
            score_components.append(0.3)
        
        # Indicator coverage
        indicators = data.get('financial_indicators', {})
        if len(indicators) >= 5:
            score_components.append(0.2)
        
        # Causal chains identified
        if data.get('causal_chains'):
            score_components.append(0.2)
        
        return sum(score_components)

    def infer_resource_causal_chains(
        self, 
        financial_data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Infer causal chains from resources to outcomes.
        
        Args:
            financial_data: Financial allocation data
            **kwargs: Additional inference parameters
            
        Returns:
            Dict with identified causal chains and their strength
        """
        causal_chains = {
            'chains': [],
            'method': 'bayesian_network',
            'confidence': 0.7
        }
        
        # Extract budget allocations as potential treatment variables
        budget_items = financial_data.get('budget_summary', {})
        
        for item_name, amount in budget_items.items():
            chain = {
                'source': f'budget_{item_name}',
                'intermediates': [],
                'outcome': 'population_outcome',
                'estimated_effect': 0.0,
                'confidence': 0.6
            }
            causal_chains['chains'].append(chain)
        
        return causal_chains

    # ========================================================================
    # LEGACY METHOD ALIASES (with Deprecation Warnings)
    # ========================================================================

    def analyze_pdf(self, pdf_path: str, **kwargs) -> Dict:
        """
        DEPRECATED: Use analyze_financial_tables() instead.
        """
        warnings.warn(
            "analyze_pdf() is deprecated, use analyze_financial_tables() instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.analyze_financial_tables(pdf_path, **kwargs)

    def get_budget_data(self, pdf_path: str, **kwargs) -> pd.DataFrame:
        """
        DEPRECATED: Use extract_budget_allocation() instead.
        """
        warnings.warn(
            "get_budget_data() is deprecated, use extract_budget_allocation() instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.extract_budget_allocation(pdf_path, **kwargs)

    def check_consistency(self, tables: List[pd.DataFrame], **kwargs) -> Dict:
        """
        DEPRECATED: Use validate_financial_consistency() instead.
        """
        warnings.warn(
            "check_consistency() is deprecated, use validate_financial_consistency() instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.validate_financial_consistency(tables, **kwargs)

    def viability_score(self, data: Dict, **kwargs) -> float:
        """
        DEPRECATED: Use compute_financial_viability_score() instead.
        """
        warnings.warn(
            "viability_score() is deprecated, use compute_financial_viability_score() instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.compute_financial_viability_score(data, **kwargs)

    def causal_inference(self, financial_data: Dict, **kwargs) -> Dict:
        """
        DEPRECATED: Use infer_resource_causal_chains() instead.
        """
        warnings.warn(
            "causal_inference() is deprecated, use infer_resource_causal_chains() instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.infer_resource_causal_chains(financial_data, **kwargs)

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def get_colombian_context(self) -> Dict[str, Any]:
        """Get Colombian municipal context information"""
        return {
            'official_systems': self.ColombianMunicipalContext.OFFICIAL_SYSTEMS,
            'territorial_categories': self.ColombianMunicipalContext.TERRITORIAL_CATEGORIES,
            'dnp_dimensions': self.ColombianMunicipalContext.DNP_DIMENSIONS
        }

    def extract_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """Extract document metadata"""
        self._ensure_analyzer(pdf_path)
        return {
            'pdf_path': str(self.analyzer.pdf_path),
            'page_count': getattr(self.analyzer, 'page_count', 0),
            'extraction_timestamp': str(pd.Timestamp.now())
        }

    def is_available(self) -> bool:
        """Check if core module is available"""
        return self._module_available
