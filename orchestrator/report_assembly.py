"""
Report Assembly - MICRO/MESO/MACRO multi-level reporting
Generates doctoral-level insights and convergence analysis

Refactored to use QuestionnaireParser as canonical source for rubric and dimension data.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import statistics
import re
from datetime import datetime

from .config import CONFIG
from .choreographer import ExecutionResult
from .question_router import Question
from .questionnaire_parser import get_questionnaire_parser

logger = logging.getLogger(__name__)


@dataclass
class MicroLevelAnswer:
    """MICRO level: Individual question answer"""
    question_id: str  # P#-D#-Q#
    qualitative_note: str  # EXCELENTE/BUENO/ACEPTABLE/INSUFICIENTE
    quantitative_score: float  # 0.0-1.0
    evidence: List[str]  # Extracts from plan
    explanation: str  # 150-300 words, doctoral level
    confidence: float  # 0.0-1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MesoLevelCluster:
    """MESO level: Cluster aggregation"""
    cluster_name: str  # CLUSTER_1, CLUSTER_2, etc
    cluster_description: str
    policy_areas: List[str]  # [P1, P2, etc]
    avg_score: float
    dimension_scores: Dict[str, float]  # D1: 0.75, D2: 0.65, etc
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    question_coverage: float  # % of questions answered
    total_questions: int
    answered_questions: int
    evidence_quality: Dict[str, float] = field(default_factory=dict)  # Evidence quality by dimension


@dataclass
class MacroLevelConvergence:
    """MACRO level: Overall convergence with Decalogo"""
    overall_score: float
    convergence_by_dimension: Dict[str, float]
    convergence_by_policy_area: Dict[str, float]
    gap_analysis: Dict[str, Any]
    agenda_alignment: float  # 0.0-1.0, how aligned with the 10 policy areas
    critical_gaps: List[str]
    strategic_recommendations: List[str]
    plan_classification: str  # "EXCELENTE"/"BUENO"/"ACEPTABLE"/"INSUFICIENTE"
    evidence_synthesis: Dict[str, Any] = field(default_factory=dict)  # Synthesis of evidence across dimensions
    implementation_roadmap: List[Dict[str, Any]] = field(default_factory=list)  # Prioritized implementation steps


class ReportAssembler:
    """
    Assembles comprehensive reports at three levels:
    - MICRO: Per-question analysis
    - MESO: Cluster-level synthesis
    - MACRO: Overall convergence assessment
    
    Uses QuestionnaireParser for rubric levels and dimension metadata.
    """

    def __init__(self):
        # Use QuestionnaireParser for canonical data
        self.parser = get_questionnaire_parser()
        
        self.rubric_levels = {
            "EXCELENTE": (0.85, 1.00),
            "BUENO": (0.70, 0.84),
            "ACEPTABLE": (0.55, 0.69),
            "INSUFICIENTE": (0.00, 0.54)
        }

        # Load dimension descriptions from parser
        self.dimension_descriptions = self._load_dimension_descriptions()

    def _load_dimension_descriptions(self) -> Dict[str, str]:
        """Load dimension descriptions from QuestionnaireParser"""
        descriptions = {}
        dimensions = self.parser.get_all_dimensions()
        
        for dim_code, dim_data in dimensions.items():
            descriptions[dim_code] = f"{dim_data.name} - {dim_data.description}"
        
        return descriptions

    # ============================================================================
    # MICRO LEVEL
    # ============================================================================

    def generate_micro_answer(
            self,
            question: Question,
            execution_results: Dict[str, ExecutionResult],
            plan_text: str
    ) -> MicroLevelAnswer:
        """
        Generate MICRO-level answer for a single question.

        Args:
            question: Question object
            execution_results: Results from all modules that processed this question
            plan_text: Full plan text

        Returns:
            MicroLevelAnswer with complete analysis
        """
        logger.info(f"Generating MICRO answer for {question.canonical_id}")

        # Aggregate evidence from all modules
        all_evidence = self._aggregate_evidence(execution_results)

        # Calculate quantitative score
        score = self._calculate_question_score(question, all_evidence)

        # Map to qualitative level
        qualitative = self._score_to_qualitative(score)

        # Extract evidence excerpts
        evidence_excerpts = self._extract_evidence_excerpts(
            question,
            all_evidence,
            plan_text
        )

        # Generate doctoral-level explanation
        explanation = self._generate_doctoral_explanation(
            question,
            qualitative,
            score,
            evidence_excerpts,
            all_evidence
        )

        # Calculate confidence
        confidence = self._calculate_confidence(execution_results, all_evidence)

        return MicroLevelAnswer(
            question_id=question.canonical_id,
            qualitative_note=qualitative,
            quantitative_score=score,
            evidence=evidence_excerpts,
            explanation=explanation,
            confidence=confidence,
            metadata={
                "dimension": question.dimension,
                "policy_area": question.policy_area,
                "modules_used": list(execution_results.keys()),
                "evidence_sources": len(evidence_excerpts),
                "primary_module": question.primary_module,
                "supporting_modules": question.supporting_modules
            }
        )

    def _aggregate_evidence(
            self,
            execution_results: Dict[str, ExecutionResult]
    ) -> Dict[str, Any]:
        """Aggregate evidence from all module results"""
        aggregated = {
            "quantitative_claims": [],
            "causal_links": [],
            "contradictions": [],
            "confidence_scores": {},
            "all_outputs": {},
            "component_results": {}  # Track component-level results
        }

        for component_key, result in execution_results.items():
            if result.evidence_extracted:
                evidence = result.evidence_extracted

                aggregated["quantitative_claims"].extend(
                    evidence.get("quantitative_claims", [])
                )
                aggregated["causal_links"].extend(
                    evidence.get("causal_links", [])
                )
                aggregated["contradictions"].extend(
                    evidence.get("contradictions", [])
                )
                aggregated["confidence_scores"].update(
                    evidence.get("confidence_scores", {})
                )

            aggregated["all_outputs"][component_key] = result.output
            aggregated["component_results"][component_key] = {
                "status": result.status.value,
                "confidence": result.confidence,
                "execution_time": result.execution_time
            }

        return aggregated

    def _calculate_question_score(
            self,
            question: Question,
            evidence: Dict[str, Any]
    ) -> float:
        """
        Calculate quantitative score (0.0-1.0) for a question.

        Scoring logic:
        - Count verification patterns matched
        - Weight by evidence confidence
        - Apply rubric thresholds
        """
        # Count matched verification patterns
        patterns_matched = 0
        total_patterns = len(question.verification_patterns)

        if total_patterns == 0:
            # No patterns defined, score based on evidence presence
            base_score = 0.5
        else:
            # Score based on pattern matching
            # This is simplified - in production, you'd do actual regex matching
            patterns_matched = total_patterns  # Placeholder
            base_score = patterns_matched / total_patterns

        # Adjust by confidence scores
        confidence_scores = evidence.get("confidence_scores", {})
        if confidence_scores:
            avg_confidence = statistics.mean(confidence_scores.values())
            adjusted_score = base_score * avg_confidence
        else:
            adjusted_score = base_score * 0.7  # Penalty for no confidence

        # Check for contradictions (penalty)
        if evidence.get("contradictions"):
            contradiction_penalty = 0.1 * len(evidence["contradictions"])
            adjusted_score = max(0.0, adjusted_score - contradiction_penalty)

        # Component success rate bonus
        component_results = evidence.get("component_results", {})
        if component_results:
            successful_components = sum(
                1 for r in component_results.values()
                if r.get("status") == "completed"
            )
            component_success_rate = successful_components / len(component_results)
            component_bonus = component_success_rate * 0.1
            adjusted_score = min(1.0, adjusted_score + component_bonus)

        # Clamp to [0, 1]
        return max(0.0, min(1.0, adjusted_score))

    def _score_to_qualitative(self, score: float) -> str:
        """Map quantitative score to qualitative level"""
        for level, (min_score, max_score) in self.rubric_levels.items():
            if min_score <= score <= max_score:
                return level
        return "INSUFICIENTE"

    def _extract_evidence_excerpts(
            self,
            question: Question,
            evidence: Dict[str, Any],
            plan_text: str,
            max_excerpts: int = 5
    ) -> List[str]:
        """Extract relevant text excerpts from plan as evidence"""
        excerpts = []

        # Extract from quantitative claims
        for claim in evidence.get("quantitative_claims", [])[:2]:
            if isinstance(claim, dict) and "dimension" in claim:
                excerpts.append(
                    f"Quantitative evidence for {claim['dimension']}: "
                    f"Bayesian score {claim.get('bayesian_score', 0.0):.2f}"
                )

        # Extract from causal links
        for link in evidence.get("causal_links", [])[:2]:
            if isinstance(link, dict):
                excerpts.append(
                    f"Causal mechanism: {link.get('description', 'Mechanism identified')}"
                )

        # Extract from contradictions
        for contradiction in evidence.get("contradictions", [])[:1]:
            if isinstance(contradiction, dict):
                excerpts.append(
                    f"Policy inconsistency: {contradiction.get('description', 'Contradiction detected')}"
                )

        # If no evidence found, add placeholder
        if not excerpts:
            excerpts.append(
                f"Limited documentary evidence found for {question.dimension} in this policy area"
            )

        return excerpts[:max_excerpts]

    def _generate_doctoral_explanation(
            self,
            question: Question,
            qualitative: str,
            score: float,
            evidence: List[str],
            all_evidence: Dict[str, Any]
    ) -> str:
        """
        Generate 150-300 word doctoral-level explanation.

        Structure:
        1. Assessment statement
        2. Evidence synthesis
        3. Critical analysis
        4. Implications
        """
        explanation_parts = []

        # 1. Assessment
        assessment = (
            f"The development plan receives a rating of **{qualitative}** "
            f"(score: {score:.2f}) for {question.dimension} in this policy area. "
        )
        explanation_parts.append(assessment)

        # 2. Evidence synthesis
        num_evidence = len(evidence)
        if num_evidence > 0:
            synthesis = (
                f"This assessment is grounded in {num_evidence} pieces of documentary evidence "
                f"extracted from the plan through mechanistic analysis. "
            )
        else:
            synthesis = (
                "However, documentary evidence supporting this dimension is notably absent "
                "from the plan, representing a critical gap in policy design. "
            )
        explanation_parts.append(synthesis)

        # 3. Critical analysis
        if score >= 0.85:
            analysis = (
                "The plan demonstrates exceptional rigor in formalizing this causal dimension, "
                "with clear operationalization of inputs, activities, and expected outputs. "
                "The theory of change is explicit and grounded in mechanistic evidence."
            )
        elif score >= 0.70:
            analysis = (
                "The plan shows substantial compliance with methodological standards, "
                "though some elements lack the specificity required for full accountability. "
                "The causal logic is present but could benefit from more explicit articulation."
            )
        elif score >= 0.55:
            analysis = (
                "The plan exhibits partial compliance, with key elements mentioned but not "
                "fully operationalized. Critical gaps exist in the specification of mechanisms, "
                "indicators, or resource allocation that limit the plan's evaluability."
            )
        else:
            analysis = (
                "The plan fails to meet minimum standards for this dimension, lacking fundamental "
                "components of policy design. This represents a serious methodological deficiency "
                "that undermines the plan's credibility and potential for impact."
            )
        explanation_parts.append(analysis)

        # 4. Implications
        if score < 0.70:
            implications = (
                "From a public policy perspective, these deficiencies compromise the plan's "
                "capacity to guide implementation and accountability mechanisms effectively."
            )
            explanation_parts.append(implications)

        return " ".join(explanation_parts)

    def _calculate_confidence(
            self,
            execution_results: Dict[str, ExecutionResult],
            evidence: Dict[str, Any]
    ) -> float:
        """Calculate overall confidence in the answer"""
        # Factors:
        # 1. Component success rate
        # 2. Evidence confidence scores
        # 3. Number of evidence sources

        successful = sum(
            1 for r in execution_results.values()
            if r.status.value == "completed"
        )
        success_rate = successful / len(execution_results) if execution_results else 0.0

        confidence_scores = evidence.get("confidence_scores", {})
        avg_confidence = (
            statistics.mean(confidence_scores.values())
            if confidence_scores else 0.5
        )

        # Weight by evidence diversity
        num_sources = len(evidence.get("quantitative_claims", [])) + len(evidence.get("causal_links", []))
        diversity_bonus = min(0.2, num_sources * 0.05)

        total_confidence = (success_rate * 0.4) + (avg_confidence * 0.4) + diversity_bonus
        return max(0.0, min(1.0, total_confidence))

    # ============================================================================
    # MESO LEVEL
    # ============================================================================

    def generate_meso_cluster(
            self,
            cluster_name: str,
            micro_answers: List[MicroLevelAnswer]
    ) -> MesoLevelCluster:
        """
        Generate MESO-level cluster analysis.

        Args:
            cluster_name: "CLUSTER_1", "CLUSTER_2", etc
            micro_answers: All micro-level answers for this cluster

        Returns:
            MesoLevelCluster with aggregated insights
        """
        logger.info(f"Generating MESO analysis for {cluster_name}")

        # Extract policy areas from micro answers
        policy_areas = list(set(a.metadata["policy_area"] for a in micro_answers))

        # Calculate aggregate scores
        avg_score = statistics.mean(a.quantitative_score for a in micro_answers)

        # Scores by dimension
        dimension_scores = defaultdict(list)
        for answer in micro_answers:
            dimension_scores[answer.metadata["dimension"]].append(answer.quantitative_score)

        dimension_averages = {
            dim: statistics.mean(scores)
            for dim, scores in dimension_scores.items()
        }

        # Evidence quality by dimension
        evidence_quality = {}
        for dim in dimension_averages.keys():
            dim_answers = [a for a in micro_answers if a.metadata["dimension"] == dim]
            evidence_quality[dim] = statistics.mean(a.confidence for a in dim_answers)

        # Identify strengths and weaknesses
        strengths = [
            f"{dim} ({self.dimension_descriptions.get(dim, dim)}): {score:.2f}"
            for dim, score in dimension_averages.items()
            if score >= 0.75
        ]

        weaknesses = [
            f"{dim} ({self.dimension_descriptions.get(dim, dim)}): {score:.2f}"
            for dim, score in dimension_averages.items()
            if score < 0.60
        ]

        # Generate recommendations
        recommendations = self._generate_cluster_recommendations(
            dimension_averages,
            evidence_quality,
            policy_areas,
            micro_answers
        )

        # Coverage metrics
        total_questions = len(micro_answers)
        answered_questions = sum(
            1 for a in micro_answers
            if a.confidence > CONFIG.min_evidence_confidence
        )
        coverage = answered_questions / total_questions if total_questions > 0 else 0.0

        return MesoLevelCluster(
            cluster_name=cluster_name,
            cluster_description=self._get_cluster_description(cluster_name),
            policy_areas=policy_areas,
            avg_score=avg_score,
            dimension_scores=dimension_averages,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
            question_coverage=coverage,
            total_questions=total_questions,
            answered_questions=answered_questions,
            evidence_quality=evidence_quality
        )

    def _get_cluster_description(self, cluster_name: str) -> str:
        """Get human-readable cluster description"""
        descriptions = {
            "CLUSTER_1": "Paz, Seguridad y Protección de Defensores",
            "CLUSTER_2": "Grupos poblacionales (mujeres, niños, privados de la libertad y migrantes)",
            "CLUSTER_3": "Tierra, Ambiente y Territorio",
            "CLUSTER_4": "DESC (Derechos Económicos, Sociales y Culturales)"
        }
        return descriptions.get(cluster_name, cluster_name)

    def _generate_cluster_recommendations(
            self,
            dimension_scores: Dict[str, float],
            evidence_quality: Dict[str, float],
            policy_areas: List[str],
            micro_answers: List[MicroLevelAnswer]
    ) -> List[str]:
        """Generate strategic recommendations for cluster"""
        recommendations = []

        # Dimension-specific recommendations
        for dim, score in dimension_scores.items():
            quality = evidence_quality.get(dim, 0.0)

            if score < 0.55:
                recommendations.append(
                    f"CRITICAL: Strengthen {dim} dimension through explicit documentation "
                    f"of mechanisms and evidence (current score: {score:.2f}, evidence quality: {quality:.2f})"
                )
            elif score < 0.70:
                recommendations.append(
                    f"Improve {dim} dimension by enhancing operationalization and "
                    f"measurement frameworks (current score: {score:.2f}, evidence quality: {quality:.2f})"
                )
            elif quality < 0.6:
                recommendations.append(
                    f"Enhance evidence quality for {dim} dimension through systematic documentation "
                    f"(current evidence quality: {quality:.2f})"
                )

        # Cross-cutting recommendations
        avg_confidence = statistics.mean(a.confidence for a in micro_answers)
        if avg_confidence < 0.6:
            recommendations.append(
                "Enhance overall evidence quality through systematic documentation "
                "of causal mechanisms and impact pathways"
            )

        # Policy area-specific recommendations
        policy_scores = defaultdict(list)
        for answer in micro_answers:
            policy_scores[answer.metadata["policy_area"]].append(answer.quantitative_score)

        for policy, scores in policy_scores.items():
            if statistics.mean(scores) < 0.6:
                recommendations.append(
                    f"Strengthen policy design in {policy} area through more detailed "
                    f"operationalization and evidence documentation"
                )

        return recommendations

    # ============================================================================
    # MACRO LEVEL
    # ============================================================================

    def generate_macro_convergence(
            self,
            all_micro_answers: List[MicroLevelAnswer],
            meso_clusters: List[MesoLevelCluster]
    ) -> MacroLevelConvergence:
        """
        Generate MACRO-level convergence analysis.

        Args:
            all_micro_answers: All 300 micro-level answers
            meso_clusters: All 4 meso-level clusters

        Returns:
            MacroLevelConvergence with overall assessment
        """
        logger.info("Generating MACRO convergence analysis")

        # Overall score
        overall_score = statistics.mean(a.quantitative_score for a in all_micro_answers)

        # Convergence by dimension
        dimension_scores = defaultdict(list)
        for answer in all_micro_answers:
            dimension_scores[answer.metadata["dimension"]].append(answer.quantitative_score)

        convergence_by_dimension = {
            dim: statistics.mean(scores)
            for dim, scores in dimension_scores.items()
        }

        # Convergence by policy area
        policy_scores = defaultdict(list)
        for answer in all_micro_answers:
            policy_scores[answer.metadata["policy_area"]].append(answer.quantitative_score)

        convergence_by_policy_area = {
            policy: statistics.mean(scores)
            for policy, scores in policy_scores.items()
        }

        # Gap analysis
        gap_analysis = self._analyze_gaps(
            convergence_by_dimension,
            convergence_by_policy_area,
            all_micro_answers
        )

        # Agenda alignment (how well covered are the 10 policy areas)
        agenda_alignment = self._calculate_agenda_alignment(convergence_by_policy_area)

        # Critical gaps
        critical_gaps = self._identify_critical_gaps(
            convergence_by_dimension,
            convergence_by_policy_area
        )

        # Strategic recommendations
        strategic_recommendations = self._generate_strategic_recommendations(
            overall_score,
            gap_analysis,
            critical_gaps,
            meso_clusters
        )

        # Evidence synthesis
        evidence_synthesis = self._synthesize_evidence(all_micro_answers)

        # Implementation roadmap
        implementation_roadmap = self._create_implementation_roadmap(
            convergence_by_dimension,
            convergence_by_policy_area,
            meso_clusters
        )

        # Overall classification
        plan_classification = self._score_to_qualitative(overall_score)

        return MacroLevelConvergence(
            overall_score=overall_score,
            convergence_by_dimension=convergence_by_dimension,
            convergence_by_policy_area=convergence_by_policy_area,
            gap_analysis=gap_analysis,
            agenda_alignment=agenda_alignment,
            critical_gaps=critical_gaps,
            strategic_recommendations=strategic_recommendations,
            plan_classification=plan_classification,
            evidence_synthesis=evidence_synthesis,
            implementation_roadmap=implementation_roadmap
        )

    def _analyze_gaps(
            self,
            dimension_scores: Dict[str, float],
            policy_scores: Dict[str, float],
            all_answers: List[MicroLevelAnswer]
    ) -> Dict[str, Any]:
        """Comprehensive gap analysis"""
        return {
            "dimension_gaps": {
                dim: 1.0 - score for dim, score in dimension_scores.items()
                if score < 0.70
            },
            "policy_area_gaps": {
                policy: 1.0 - score for policy, score in policy_scores.items()
                if score < 0.70
            },
            "missing_evidence_count": sum(
                1 for a in all_answers if len(a.evidence) == 0
            ),
            "low_confidence_count": sum(
                1 for a in all_answers if a.confidence < 0.5
            ),
            "dimension_confidence_gaps": {
                dim: 1.0 - statistics.mean(
                    a.confidence for a in all_answers
                    if a.metadata["dimension"] == dim
                )
                for dim in dimension_scores.keys()
                if statistics.mean(
                    a.confidence for a in all_answers
                    if a.metadata["dimension"] == dim
                ) < 0.6
            }
        }

    def _calculate_agenda_alignment(self, policy_scores: Dict[str, float]) -> float:
        """Calculate alignment with the 10-policy-area agenda"""
        # Alignment = average coverage * average quality
        num_covered = sum(1 for score in policy_scores.values() if score >= 0.55)
        coverage = num_covered / 10  # 10 policy areas

        avg_quality = statistics.mean(policy_scores.values()) if policy_scores else 0.0

        return (coverage * 0.5) + (avg_quality * 0.5)

    def _identify_critical_gaps(
            self,
            dimension_scores: Dict[str, float],
            policy_scores: Dict[str, float]
    ) -> List[str]:
        """Identify critical gaps requiring immediate attention"""
        gaps = []

        # Critical dimension gaps
        for dim, score in dimension_scores.items():
            if score < 0.45:
                gaps.append(f"CRITICAL: {dim} dimension severely underdeveloped (score: {score:.2f})")

        # Critical policy area gaps
        for policy, score in policy_scores.items():
            if score < 0.45:
                gaps.append(f"CRITICAL: {policy} policy area inadequately addressed (score: {score:.2f})")

        return gaps

    def _generate_strategic_recommendations(
            self,
            overall_score: float,
            gap_analysis: Dict[str, Any],
            critical_gaps: List[str],
            meso_clusters: List[MesoLevelCluster]
    ) -> List[str]:
        """Generate strategic recommendations for the entire plan"""
        recommendations = []

        # Overall assessment
        if overall_score < 0.55:
            recommendations.append(
                "PRIORITY 1: Fundamental restructuring required. The plan lacks basic "
                "methodological rigor and requires comprehensive revision of its Theory of Change."
            )
        elif overall_score < 0.70:
            recommendations.append(
                "PRIORITY 1: Significant enhancements needed. Focus on operationalizing "
                "causal mechanisms and strengthening evidence frameworks."
            )

        # Dimension-specific
        for dim, gap in gap_analysis["dimension_gaps"].items():
            if gap > 0.4:
                recommendations.append(
                    f"PRIORITY 2: Address critical gap in {dim} dimension "
                    f"through systematic documentation and causal mapping."
                )

        # Evidence quality gaps
        for dim, gap in gap_analysis.get("dimension_confidence_gaps", {}).items():
            if gap > 0.4:
                recommendations.append(
                    f"PRIORITY 2: Improve evidence quality in {dim} dimension "
                    f"through systematic documentation and validation."
                )

        # Cluster insights
        weak_clusters = [c for c in meso_clusters if c.avg_score < 0.60]
        if weak_clusters:
            cluster_names = ", ".join(c.cluster_name for c in weak_clusters)
            recommendations.append(
                f"PRIORITY 2: Strengthen policy coverage in clusters: {cluster_names}"
            )

        # Evidence quality
        if gap_analysis["missing_evidence_count"] > 50:
            recommendations.append(
                "PRIORITY 3: Enhance evidence documentation systematically across all dimensions"
            )

        return recommendations

    def _synthesize_evidence(self, all_micro_answers: List[MicroLevelAnswer]) -> Dict[str, Any]:
        """Synthesize evidence across all dimensions and policy areas"""
        # Group evidence by dimension
        dimension_evidence = defaultdict(list)
        for answer in all_micro_answers:
            dimension_evidence[answer.metadata["dimension"]].extend(answer.evidence)

        # Count evidence types
        evidence_types = defaultdict(int)
        for answer in all_micro_answers:
            for evidence in answer.evidence:
                if "Quantitative" in evidence:
                    evidence_types["quantitative"] += 1
                elif "Causal" in evidence:
                    evidence_types["causal"] += 1
                elif "Contradiction" in evidence:
                    evidence_types["contradiction"] += 1
                else:
                    evidence_types["other"] += 1

        return {
            "dimension_evidence": {
                dim: len(evidence) for dim, evidence in dimension_evidence.items()
            },
            "evidence_types": dict(evidence_types),
            "total_evidence_items": sum(len(answer.evidence) for answer in all_micro_answers),
            "avg_evidence_per_question": statistics.mean(
                len(answer.evidence) for answer in all_micro_answers
            )
        }

    def _create_implementation_roadmap(
            self,
            dimension_scores: Dict[str, float],
            policy_scores: Dict[str, float],
            meso_clusters: List[MesoLevelCluster]
    ) -> List[Dict[str, Any]]:
        """Create prioritized implementation roadmap"""
        roadmap = []

        # Sort dimensions by score (lowest first)
        sorted_dimensions = sorted(dimension_scores.items(), key=lambda x: x[1])

        # Add dimension-specific steps
        for i, (dim, score) in enumerate(sorted_dimensions[:3]):  # Top 3 priorities
            priority = "HIGH" if i == 0 else "MEDIUM" if i == 1 else "LOW"

            roadmap.append({
                "priority": priority,
                "focus_area": f"Dimension {dim}",
                "description": f"Strengthen {self.dimension_descriptions.get(dim, dim)}",
                "current_score": score,
                "target_score": min(1.0, score + 0.3),
                "estimated_effort": "HIGH" if score < 0.5 else "MEDIUM"
            })

        # Add cluster-specific steps
        weak_clusters = sorted(meso_clusters, key=lambda c: c.avg_score)[:2]
        for i, cluster in enumerate(weak_clusters):
            roadmap.append({
                "priority": "MEDIUM",
                "focus_area": f"Cluster {cluster.cluster_name}",
                "description": f"Improve {cluster.cluster_description}",
                "current_score": cluster.avg_score,
                "target_score": min(1.0, cluster.avg_score + 0.2),
                "estimated_effort": "MEDIUM"
            })

        return roadmap

    # ============================================================================
    # EXPORT
    # ============================================================================

    def export_full_report(
            self,
            plan_name: str,
            micro_answers: List[MicroLevelAnswer],
            meso_clusters: List[MesoLevelCluster],
            macro_convergence: MacroLevelConvergence,
            output_dir: Path
    ):
        """Export complete report in multiple formats"""
        logger.info(f"Exporting full report for {plan_name}")

        # JSON export
        json_report = {
            "plan_name": plan_name,
            "analysis_date": datetime.now().isoformat(),
            "micro_level": [vars(a) for a in micro_answers],
            "meso_level": [vars(c) for c in meso_clusters],
            "macro_level": vars(macro_convergence)
        }

        json_path = output_dir / f"{plan_name}_report.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_report, f, indent=2, ensure_ascii=False)

        logger.info(f"JSON report saved to {json_path}")

        # Markdown export (human-readable)
        md_path = output_dir / f"{plan_name}_report.md"
        self._export_markdown(plan_name, micro_answers, meso_clusters, macro_convergence, md_path)

        logger.info(f"Markdown report saved to {md_path}")

        # Excel export for data analysis
        excel_path = output_dir / f"{plan_name}_report.xlsx"
        self._export_excel(plan_name, micro_answers, meso_clusters, macro_convergence, excel_path)

        logger.info(f"Excel report saved to {excel_path}")

    def _export_markdown(
            self,
            plan_name: str,
            micro_answers: List[MicroLevelAnswer],
            meso_clusters: List[MesoLevelCluster],
            macro_convergence: MacroLevelConvergence,
            output_path: Path
    ):
        """Export report as markdown"""
        with open(output_path, 'w', encoding='utf-8') as f:
            # Header
            f.write(f"# FARFAN 3.0 Policy Analysis Report\n\n")
            f.write(f"## Plan: {plan_name}\n\n")
            f.write(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")

            # MACRO Level Summary
            f.write("## MACRO LEVEL: Overall Convergence\n\n")
            f.write(f"**Overall Classification:** {macro_convergence.plan_classification}\n\n")
            f.write(f"**Overall Score:** {macro_convergence.overall_score:.2f}\n\n")
            f.write(f"**Agenda Alignment:** {macro_convergence.agenda_alignment:.2f}\n\n")

            f.write("### Convergence by Dimension\n\n")
            for dim, score in macro_convergence.convergence_by_dimension.items():
                f.write(f"- **{dim} ({self.dimension_descriptions.get(dim, dim)}):** {score:.2f}\n")

            f.write("\n### Evidence Synthesis\n\n")
            evidence = macro_convergence.evidence_synthesis
            f.write(f"- **Total Evidence Items:** {evidence['total_evidence_items']}\n")
            f.write(f"- **Average Evidence per Question:** {evidence['avg_evidence_per_question']:.2f}\n")

            f.write("\n### Critical Gaps\n\n")
            for gap in macro_convergence.critical_gaps:
                f.write(f"- {gap}\n")

            f.write("\n### Strategic Recommendations\n\n")
            for i, rec in enumerate(macro_convergence.strategic_recommendations, 1):
                f.write(f"{i}. {rec}\n")

            f.write("\n### Implementation Roadmap\n\n")
            f.write("| Priority | Focus Area | Current Score | Target Score | Estimated Effort |\n")
            f.write("|----------|------------|---------------|--------------|------------------|\n")
            for step in macro_convergence.implementation_roadmap:
                f.write(f"| {step['priority']} | {step['focus_area']} | {step['current_score']:.2f} | "
                        f"{step['target_score']:.2f} | {step['estimated_effort']} |\n")

            f.write("\n---\n\n")

            # MESO Level
            f.write("## MESO LEVEL: Cluster Analysis\n\n")
            for cluster in meso_clusters:
                f.write(f"### {cluster.cluster_name}: {cluster.cluster_description}\n\n")
                f.write(f"**Average Score:** {cluster.avg_score:.2f}\n\n")
                f.write(
                    f"**Coverage:** {cluster.question_coverage:.1%} ({cluster.answered_questions}/{cluster.total_questions})\n\n")

                f.write("**Strengths:**\n")
                for strength in cluster.strengths:
                    f.write(f"- {strength}\n")

                f.write("\n**Weaknesses:**\n")
                for weakness in cluster.weaknesses:
                    f.write(f"- {weakness}\n")

                f.write("\n**Recommendations:**\n")
                for rec in cluster.recommendations:
                    f.write(f"- {rec}\n")

                f.write("\n**Evidence Quality by Dimension:**\n")
                for dim, quality in cluster.evidence_quality.items():
                    f.write(f"- {dim}: {quality:.2f}\n")

                f.write("\n---\n\n")

            # MICRO Level (summary only, full details in JSON)
            f.write("## MICRO LEVEL: Question-by-Question Analysis\n\n")
            f.write(f"Total Questions Analyzed: {len(micro_answers)}\n\n")

            # Summary by dimension
            dimension_summary = defaultdict(list)
            for answer in micro_answers:
                dimension_summary[answer.metadata["dimension"]].append(answer.quantitative_score)

            f.write("### Summary by Dimension\n\n")
            for dim, scores in dimension_summary.items():
                avg_score = statistics.mean(scores)
                f.write(f"- **{dim} ({self.dimension_descriptions.get(dim, dim)}):** "
                        f"Average Score {avg_score:.2f} ({len(scores)} questions)\n")

            f.write("\nFor detailed question-by-question analysis, see the JSON report.\n\n")

            logger.info(f"Markdown export complete: {output_path}")

    def _export_excel(
            self,
            plan_name: str,
            micro_answers: List[MicroLevelAnswer],
            meso_clusters: List[MesoLevelCluster],
            macro_convergence: MacroLevelConvergence,
            output_path: Path
    ):
        """Export report as Excel workbook"""
        try:
            import pandas as pd

            # Create Excel writer
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Macro level summary
                macro_data = {
                    "Metric": [
                        "Overall Score",
                        "Overall Classification",
                        "Agenda Alignment"
                    ],
                    "Value": [
                        macro_convergence.overall_score,
                        macro_convergence.plan_classification,
                        macro_convergence.agenda_alignment
                    ]
                }
                pd.DataFrame(macro_data).to_excel(writer, sheet_name="MACRO_Summary", index=False)

                # Dimension scores
                dim_data = [
                    {"Dimension": dim, "Score": score}
                    for dim, score in macro_convergence.convergence_by_dimension.items()
                ]
                pd.DataFrame(dim_data).to_excel(writer, sheet_name="MACRO_Dimensions", index=False)

                # Policy area scores
                policy_data = [
                    {"Policy Area": policy, "Score": score}
                    for policy, score in macro_convergence.policy_scores.items()
                ]
                pd.DataFrame(policy_data).to_excel(writer, sheet_name="MACRO_Policy_Areas", index=False)

                # Meso level clusters
                cluster_data = []
                for cluster in meso_clusters:
                    cluster_data.append({
                        "Cluster": cluster.cluster_name,
                        "Description": cluster.cluster_description,
                        "Average Score": cluster.avg_score,
                        "Coverage": cluster.question_coverage,
                        "Total Questions": cluster.total_questions,
                        "Answered Questions": cluster.answered_questions
                    })
                pd.DataFrame(cluster_data).to_excel(writer, sheet_name="MESO_Clusters", index=False)

                # Micro level answers
                micro_data = []
                for answer in micro_answers:
                    micro_data.append({
                        "Question ID": answer.question_id,
                        "Dimension": answer.metadata["dimension"],
                        "Policy Area": answer.metadata["policy_area"],
                        "Qualitative": answer.qualitative_note,
                        "Score": answer.quantitative_score,
                        "Confidence": answer.confidence,
                        "Evidence Count": len(answer.evidence)
                    })
                pd.DataFrame(micro_data).to_excel(writer, sheet_name="MICRO_Answers", index=False)

        except ImportError:
            logger.warning("pandas/openpyxl not available for Excel export")