"""
Dashboard Data Generator for AtroZ Web Interface
Converts FARFAN analysis results to dashboard-compatible JSON format
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

from .report_assembly import MicroLevelAnswer, MesoLevelCluster, MacroLevelConvergence

logger = logging.getLogger(__name__)


class DashboardDataGenerator:
    """
    Generates JSON data files for the AtroZ web dashboard.

    Output structure:
    - dashboard_data.json: Real-time data feed
    - pdet_regions.json: Regional constellation data
    - timeline_data.json: Historical progression
    """

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.dashboard_dir = output_dir / "dashboard"
        self.dashboard_dir.mkdir(exist_ok=True, parents=True)

    def generate_dashboard_data(
            self,
            plan_name: str,
            micro_answers: List[MicroLevelAnswer],
            meso_clusters: List[MesoLevelCluster],
            macro_convergence: MacroLevelConvergence,
            plan_metadata: Dict[str, Any]
    ) -> Dict[str, Path]:
        """
        Generate all dashboard data files for a single plan.

        Args:
            plan_name: Name of the development plan
            micro_answers: All 300 micro-level answers
            meso_clusters: 4 meso-level clusters
            macro_convergence: Macro-level convergence data
            plan_metadata: Additional plan metadata

        Returns:
            Dict mapping file types to their paths
        """
        logger.info(f"Generating dashboard data for {plan_name}")

        output_files = {}

        # 1. Main dashboard data
        dashboard_data = self._generate_main_dashboard(
            plan_name,
            micro_answers,
            meso_clusters,
            macro_convergence,
            plan_metadata
        )

        dashboard_path = self.dashboard_dir / f"{plan_name}_dashboard.json"
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            json.dump(dashboard_data, f, indent=2, ensure_ascii=False)
        output_files['dashboard'] = dashboard_path

        # 2. PDET region constellation
        pdet_data = self._generate_pdet_constellation(
            plan_name,
            meso_clusters,
            macro_convergence
        )

        pdet_path = self.dashboard_dir / f"{plan_name}_pdet.json"
        with open(pdet_path, 'w', encoding='utf-8') as f:
            json.dump(pdet_data, f, indent=2, ensure_ascii=False)
        output_files['pdet'] = pdet_path

        # 3. Micro-level DNA helix data
        micro_data = self._generate_micro_helix(micro_answers)

        micro_path = self.dashboard_dir / f"{plan_name}_micro.json"
        with open(micro_path, 'w', encoding='utf-8') as f:
            json.dump(micro_data, f, indent=2, ensure_ascii=False)
        output_files['micro'] = micro_path

        # 4. Evidence stream ticker
        evidence_data = self._generate_evidence_stream(micro_answers)

        evidence_path = self.dashboard_dir / f"{plan_name}_evidence.json"
        with open(evidence_path, 'w', encoding='utf-8') as f:
            json.dump(evidence_data, f, indent=2, ensure_ascii=False)
        output_files['evidence'] = evidence_path

        logger.info(f"Dashboard data generated: {len(output_files)} files")
        return output_files

    def _generate_main_dashboard(
            self,
            plan_name: str,
            micro_answers: List[MicroLevelAnswer],
            meso_clusters: List[MesoLevelCluster],
            macro_convergence: MacroLevelConvergence,
            plan_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate main dashboard data structure"""

        return {
            "metadata": {
                "plan_name": plan_name,
                "timestamp": datetime.now().isoformat(),
                "version": "3.0.0",
                **plan_metadata
            },
            "macro": {
                "overall_score": int(macro_convergence.overall_score * 100),
                "classification": macro_convergence.plan_classification,
                "agenda_alignment": int(macro_convergence.agenda_alignment * 100),
                "convergence_by_dimension": {
                    dim: int(score * 100)
                    for dim, score in macro_convergence.convergence_by_dimension.items()
                },
                "convergence_by_policy_area": {
                    policy: int(score * 100)
                    for policy, score in macro_convergence.convergence_by_policy_area.items()
                },
                "critical_gaps": macro_convergence.critical_gaps,
                "strategic_recommendations": macro_convergence.strategic_recommendations[:5]
            },
            "meso": {
                "clusters": [
                    {
                        "id": cluster.cluster_name.lower().replace(" ", "_"),
                        "name": cluster.cluster_description,
                        "score": int(cluster.avg_score * 100),
                        "dimension_scores": {
                            dim: int(score * 100)
                            for dim, score in cluster.dimension_scores.items()
                        },
                        "coverage": int(cluster.question_coverage * 100),
                        "strengths": cluster.strengths,
                        "weaknesses": cluster.weaknesses,
                        "recommendations": cluster.recommendations[:3]
                    }
                    for cluster in meso_clusters
                ]
            },
            "micro": {
                "total_questions": len(micro_answers),
                "avg_score": int(sum(a.quantitative_score for a in micro_answers) / len(micro_answers) * 100),
                "by_dimension": self._aggregate_by_dimension(micro_answers),
                "by_policy_area": self._aggregate_by_policy_area(micro_answers),
                "top_questions": self._get_top_questions(micro_answers, top_n=10),
                "bottom_questions": self._get_bottom_questions(micro_answers, top_n=10)
            },
            "phylogram": self._generate_phylogram_data(macro_convergence),
            "radar_chart": self._generate_radar_data(macro_convergence)
        }

    def _generate_pdet_constellation(
            self,
            plan_name: str,
            meso_clusters: List[MesoLevelCluster],
            macro_convergence: MacroLevelConvergence
    ) -> Dict[str, Any]:
        """
        Generate PDET regional constellation data.

        Maps the 4 clusters to visual constellation points.
        """

        # Cluster to PDET region mapping (simplified)
        cluster_positions = [
            {"x": 25, "y": 35, "name": "CLUSTER 1: Paz y Seguridad"},
            {"x": 65, "y": 25, "name": "CLUSTER 2: Grupos Poblacionales"},
            {"x": 35, "y": 65, "name": "CLUSTER 3: Tierra y Ambiente"},
            {"x": 70, "y": 60, "name": "CLUSTER 4: DESC"}
        ]

        regions = []
        for i, cluster in enumerate(meso_clusters):
            position = cluster_positions[i] if i < len(cluster_positions) else {"x": 50, "y": 50}

            regions.append({
                "id": cluster.cluster_name.lower().replace(" ", "_"),
                "name": cluster.cluster_description,
                "x": position["x"],
                "y": position["y"],
                "score": int(cluster.avg_score * 100),
                "municipalities": len(cluster.policy_areas),  # Policy areas as "municipalities"
                "dimension_breakdown": {
                    dim: int(score * 100)
                    for dim, score in cluster.dimension_scores.items()
                },
                "coverage": int(cluster.question_coverage * 100),
                "status": self._get_status_color(cluster.avg_score)
            })

        # Add overall plan node at center
        regions.append({
            "id": "centro_plan",
            "name": "PLAN COMPLETO",
            "x": 50,
            "y": 45,
            "score": int(macro_convergence.overall_score * 100),
            "municipalities": sum(len(c.policy_areas) for c in meso_clusters),
            "dimension_breakdown": {
                dim: int(score * 100)
                for dim, score in macro_convergence.convergence_by_dimension.items()
            },
            "status": self._get_status_color(macro_convergence.overall_score),
            "is_center": True
        })

        return {
            "plan_name": plan_name,
            "regions": regions,
            "connections": self._generate_neural_connections(regions)
        }

    def _generate_micro_helix(
            self,
            micro_answers: List[MicroLevelAnswer]
    ) -> Dict[str, Any]:
        """Generate DNA helix data for micro-level questions"""

        # Sort questions by score for visual impact
        sorted_answers = sorted(
            micro_answers,
            key=lambda a: a.quantitative_score,
            reverse=True
        )

        helix_points = []
        for i, answer in enumerate(sorted_answers):
            angle = (i / len(sorted_answers)) * 360 * 4  # 4 full rotations
            y_position = (i / len(sorted_answers)) * 100

            helix_points.append({
                "question_id": answer.question_id,
                "angle": angle,
                "y_position": y_position,
                "score": int(answer.quantitative_score * 100),
                "qualitative": answer.qualitative_note,
                "dimension": answer.metadata.get("dimension", ""),
                "policy_area": answer.metadata.get("policy_area", ""),
                "confidence": int(answer.confidence * 100),
                "has_evidence": len(answer.evidence) > 0
            })

        return {
            "total_questions": len(helix_points),
            "helix_points": helix_points,
            "summary": {
                "excelente": sum(1 for a in micro_answers if a.qualitative_note == "EXCELENTE"),
                "bueno": sum(1 for a in micro_answers if a.qualitative_note == "BUENO"),
                "aceptable": sum(1 for a in micro_answers if a.qualitative_note == "ACEPTABLE"),
                "insuficiente": sum(1 for a in micro_answers if a.qualitative_note == "INSUFICIENTE")
            }
        }

    def _generate_evidence_stream(
            self,
            micro_answers: List[MicroLevelAnswer]
    ) -> Dict[str, Any]:
        """Generate evidence stream ticker data"""

        evidence_items = []

        for answer in micro_answers:
            if answer.evidence:
                for evidence_text in answer.evidence[:2]:  # Max 2 per question
                    evidence_items.append({
                        "question_id": answer.question_id,
                        "text": evidence_text[:150] + "..." if len(evidence_text) > 150 else evidence_text,
                        "score": int(answer.quantitative_score * 100),
                        "dimension": answer.metadata.get("dimension", ""),
                        "timestamp": datetime.now().isoformat()
                    })

        # Sort by score (highest first) and take top 50 for ticker
        evidence_items.sort(key=lambda x: x["score"], reverse=True)

        return {
            "total_evidence_items": len(evidence_items),
            "ticker_items": evidence_items[:50],
            "evidence_coverage": len([a for a in micro_answers if a.evidence]) / len(micro_answers)
        }

    def _generate_phylogram_data(
            self,
            macro_convergence: MacroLevelConvergence
    ) -> Dict[str, Any]:
        """Generate circular phylogram data for macro visualization"""

        # Create branches for each dimension
        branches = []
        for dim, score in macro_convergence.convergence_by_dimension.items():
            branches.append({
                "dimension": dim,
                "score": int(score * 100),
                "length": score,  # Branch length proportional to score
                "angle": len(branches) * (360 / 6),  # 6 dimensions = 60° each
                "color": self._get_dimension_color(dim)
            })

        return {
            "center_score": int(macro_convergence.overall_score * 100),
            "branches": branches,
            "rings": [
                {"radius": 40, "threshold": 85, "label": "EXCELENTE"},
                {"radius": 60, "threshold": 70, "label": "BUENO"},
                {"radius": 80, "threshold": 55, "label": "ACEPTABLE"}
            ]
        }

    def _generate_radar_data(
            self,
            macro_convergence: MacroLevelConvergence
    ) -> Dict[str, Any]:
        """Generate radar chart data for dimension analysis"""

        dimensions = list(macro_convergence.convergence_by_dimension.items())

        return {
            "axes": [dim for dim, _ in dimensions],
            "values": [int(score * 100) for _, score in dimensions],
            "max_value": 100,
            "reference_polygon": [70, 70, 70, 70, 70, 70]  # BUENO threshold
        }

    def _aggregate_by_dimension(
            self,
            micro_answers: List[MicroLevelAnswer]
    ) -> Dict[str, Dict[str, Any]]:
        """Aggregate micro answers by dimension"""

        from collections import defaultdict
        import statistics

        by_dimension = defaultdict(list)

        for answer in micro_answers:
            dimension = answer.metadata.get("dimension", "UNKNOWN")
            by_dimension[dimension].append(answer.quantitative_score)

        return {
            dim: {
                "avg_score": int(statistics.mean(scores) * 100),
                "count": len(scores),
                "min": int(min(scores) * 100),
                "max": int(max(scores) * 100)
            }
            for dim, scores in by_dimension.items()
        }

    def _aggregate_by_policy_area(
            self,
            micro_answers: List[MicroLevelAnswer]
    ) -> Dict[str, Dict[str, Any]]:
        """Aggregate micro answers by policy area"""

        from collections import defaultdict
        import statistics

        by_policy = defaultdict(list)

        for answer in micro_answers:
            policy = answer.metadata.get("policy_area", "UNKNOWN")
            by_policy[policy].append(answer.quantitative_score)

        return {
            policy: {
                "avg_score": int(statistics.mean(scores) * 100),
                "count": len(scores)
            }
            for policy, scores in by_policy.items()
        }

    def _get_top_questions(
            self,
            micro_answers: List[MicroLevelAnswer],
            top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """Get top N best-scoring questions"""

        sorted_answers = sorted(
            micro_answers,
            key=lambda a: a.quantitative_score,
            reverse=True
        )[:top_n]

        return [
            {
                "question_id": a.question_id,
                "score": int(a.quantitative_score * 100),
                "qualitative": a.qualitative_note,
                "confidence": int(a.confidence * 100)
            }
            for a in sorted_answers
        ]

    def _get_bottom_questions(
            self,
            micro_answers: List[MicroLevelAnswer],
            top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """Get bottom N worst-scoring questions"""

        sorted_answers = sorted(
            micro_answers,
            key=lambda a: a.quantitative_score
        )[:top_n]

        return [
            {
                "question_id": a.question_id,
                "score": int(a.quantitative_score * 100),
                "qualitative": a.qualitative_note,
                "gaps": ["Evidence missing", "Low confidence"]
            }
            for a in sorted_answers
        ]

    def _generate_neural_connections(
            self,
            regions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate neural connections between regions"""

        connections = []

        # Connect each region to center
        center_region = next((r for r in regions if r.get("is_center")), None)

        if center_region:
            for region in regions:
                if not region.get("is_center"):
                    connections.append({
                        "from": region["id"],
                        "to": center_region["id"],
                        "strength": (region["score"] + center_region["score"]) / 200,
                        "type": "radial"
                    })

        # Connect adjacent regions
        for i in range(len(regions) - 1):
            for j in range(i + 1, len(regions)):
                if not regions[i].get("is_center") and not regions[j].get("is_center"):
                    # Calculate distance
                    dx = regions[i]["x"] - regions[j]["x"]
                    dy = regions[i]["y"] - regions[j]["y"]
                    distance = (dx ** 2 + dy ** 2) ** 0.5

                    # Only connect if close enough
                    if distance < 40:
                        connections.append({
                            "from": regions[i]["id"],
                            "to": regions[j]["id"],
                            "strength": 1 - (distance / 40),
                            "type": "lateral"
                        })

        return connections

    def _get_status_color(self, score: float) -> str:
        """Get status color based on score"""
        if score >= 0.85:
            return "green"
        elif score >= 0.70:
            return "blue"
        elif score >= 0.55:
            return "yellow"
        else:
            return "red"

    def _get_dimension_color(self, dimension: str) -> str:
        """Get color for dimension"""
        colors = {
            "D1": "#39FF14",  # green-toxic
            "D2": "#00D4FF",  # blue-electric
            "D3": "#B2642E",  # copper
            "D4": "#17A589",  # copper-oxide
            "D5": "#C41E3A",  # atroz-red
            "D6": "#8B0000"   # blood
        }
        return colors.get(dimension, "#E5E7EB")

    def generate_batch_dashboard(
            self,
            all_results: List[Dict[str, Any]]
    ) -> Path:
        """
        Generate comparative dashboard for batch analysis.

        Args:
            all_results: List of all plan analysis results

        Returns:
            Path to batch dashboard JSON
        """
        logger.info(f"Generating batch dashboard for {len(all_results)} plans")

        batch_data = {
            "metadata": {
                "total_plans": len(all_results),
                "timestamp": datetime.now().isoformat(),
                "version": "3.0.0"
            },
            "rankings": {
                "by_overall_score": self._rank_plans(all_results, "overall_score"),
                "by_dimension": self._rank_by_dimension(all_results),
                "by_cluster": self._rank_by_cluster(all_results)
            },
            "statistics": {
                "avg_overall_score": self._calculate_avg_score(all_results),
                "distribution": self._calculate_distribution(all_results)
            },
            "comparison_matrix": self._generate_comparison_matrix(all_results)
        }

        batch_path = self.dashboard_dir / "batch_dashboard.json"
        with open(batch_path, 'w', encoding='utf-8') as f:
            json.dump(batch_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Batch dashboard generated: {batch_path}")
        return batch_path

    def _rank_plans(
            self,
            results: List[Dict[str, Any]],
            key: str
    ) -> List[Dict[str, Any]]:
        """Rank plans by a specific metric"""

        ranked = []
        for result in results:
            if result.get("status") != "failed" and result.get("macro_convergence"):
                macro = result["macro_convergence"]
                ranked.append({
                    "plan_name": result["plan_name"],
                    "score": int(macro.overall_score * 100),
                    "classification": macro.plan_classification
                })

        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked

    def _rank_by_dimension(self, results: List[Dict[str, Any]]) -> Dict[str, List]:
        """Rank plans by each dimension"""
        # Placeholder implementation
        return {"D1": [], "D2": [], "D3": [], "D4": [], "D5": [], "D6": []}

    def _rank_by_cluster(self, results: List[Dict[str, Any]]) -> Dict[str, List]:
        """Rank plans by cluster performance"""
        return {}

    def _calculate_avg_score(self, results: List[Dict[str, Any]]) -> int:
        """Calculate average score across all plans"""
        valid_results = [
            r for r in results
            if r.get("status") != "failed" and r.get("macro_convergence")
        ]

        if not valid_results:
            return 0

        total = sum(r["macro_convergence"].overall_score for r in valid_results)
        return int((total / len(valid_results)) * 100)

    def _calculate_distribution(self, results: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate distribution of plan classifications"""

        distribution = {
            "EXCELENTE": 0,
            "BUENO": 0,
            "ACEPTABLE": 0,
            "INSUFICIENTE": 0
        }

        for result in results:
            if result.get("status") != "failed" and result.get("macro_convergence"):
                classification = result["macro_convergence"].plan_classification
                distribution[classification] = distribution.get(classification, 0) + 1

        return distribution

    def _generate_comparison_matrix(self, results: List[Dict[str, Any]]) -> List[List[int]]:
        """Generate comparison matrix for visualization"""
        # Simplified 3x3 matrix for now
        return [[75, 68, 82], [71, 79, 65], [88, 73, 77]]
