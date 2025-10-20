"""
Homeostasis Dashboard - CI/CD Monitoring Interface
===================================================

Web-based visualization dashboard for validation gates:
- Dependency graph with test status coloring
- Contract coverage metrics (413 methods)
- Canary test grid (pass/fail/rebaseline)
- Circuit breaker status indicators
- Performance time-series charts
- Automated remediation suggestions

Author: Integration Team
Version: 1.0.0
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
from collections import defaultdict

from flask import Flask, render_template, jsonify, request
import networkx as nx
from dataclasses import asdict

from validation_gates import ValidationGatePipeline
from orchestrator.circuit_breaker import CircuitBreaker
from orchestrator.module_adapters import (
    ModulosAdapter, AnalyzerOneAdapter, DerekBeachAdapter,
    EmbeddingPolicyAdapter, SemanticChunkingPolicyAdapter,
    ContradictionDetectionAdapter, FinancialViabilityAdapter,
    PolicyProcessorAdapter, PolicySegmenterAdapter
)

logger = logging.getLogger(__name__)

app = Flask(__name__)


class HomeostasisDashboard:
    
    def __init__(self):
        self.pipeline = ValidationGatePipeline()
        self.circuit_breaker = CircuitBreaker()
        self.adapters = self._initialize_adapters()
        self.dependency_graph = self._build_dependency_graph()
        self.performance_history = self._load_performance_history()
        
    def _initialize_adapters(self) -> Dict[str, Any]:
        return {
            "teoria_cambio": ModulosAdapter(),
            "analyzer_one": AnalyzerOneAdapter(),
            "dereck_beach": DerekBeachAdapter(),
            "embedding_policy": EmbeddingPolicyAdapter(),
            "semantic_chunking_policy": SemanticChunkingPolicyAdapter(),
            "contradiction_detection": ContradictionDetectionAdapter(),
            "financial_viability": FinancialViabilityAdapter(),
            "policy_processor": PolicyProcessorAdapter(),
            "policy_segmenter": PolicySegmenterAdapter()
        }
    
    def _build_dependency_graph(self) -> nx.DiGraph:
        G = nx.DiGraph()
        
        adapter_methods = {
            "teoria_cambio": 51,
            "analyzer_one": 39,
            "dereck_beach": 89,
            "embedding_policy": 37,
            "semantic_chunking_policy": 18,
            "contradiction_detection": 52,
            "financial_viability": 60,
            "policy_processor": 34,
            "policy_segmenter": 33
        }
        
        for adapter, method_count in adapter_methods.items():
            G.add_node(adapter, type="adapter", method_count=method_count)
        
        dependencies = [
            ("policy_segmenter", "semantic_chunking_policy"),
            ("semantic_chunking_policy", "embedding_policy"),
            ("policy_processor", "embedding_policy"),
            ("contradiction_detection", "analyzer_one"),
            ("financial_viability", "teoria_cambio"),
            ("dereck_beach", "teoria_cambio"),
            ("analyzer_one", "embedding_policy")
        ]
        
        for source, target in dependencies:
            G.add_edge(source, target, type="binding")
        
        return G
    
    def _load_performance_history(self) -> Dict[str, List[Dict]]:
        history_path = Path("performance_history.json")
        if not history_path.exists():
            return {}
        
        with open(history_path) as f:
            return json.load(f)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        validation_results = self.pipeline.run_all()
        circuit_status = self.circuit_breaker.get_all_status()
        contract_coverage = self._calculate_contract_coverage()
        canary_grid = self._get_canary_test_grid()
        performance_metrics = self._get_performance_metrics()
        graph_data = self._serialize_dependency_graph(validation_results)
        
        return {
            "validation_results": validation_results,
            "circuit_breaker_status": circuit_status,
            "contract_coverage": contract_coverage,
            "canary_test_grid": canary_grid,
            "performance_metrics": performance_metrics,
            "dependency_graph": graph_data,
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_contract_coverage(self) -> Dict[str, Any]:
        contracts_dir = Path("contracts")
        total_methods = 413
        
        if not contracts_dir.exists():
            valid_contracts = 0
        else:
            valid_contracts = len(list(contracts_dir.glob("**/*.yaml")))
        
        coverage_fraction = valid_contracts / total_methods if total_methods > 0 else 0
        
        return {
            "total_methods": total_methods,
            "valid_contracts": valid_contracts,
            "missing_contracts": total_methods - valid_contracts,
            "coverage_percentage": coverage_fraction * 100,
            "coverage_fraction": f"{valid_contracts}/{total_methods}"
        }
    
    def _get_canary_test_grid(self) -> Dict[str, Dict[str, str]]:
        baselines_path = Path("baselines")
        grid = defaultdict(dict)
        
        if not baselines_path.exists():
            return {}
        
        for adapter_dir in baselines_path.iterdir():
            if not adapter_dir.is_dir():
                continue
            
            adapter_name = adapter_dir.name
            
            for method_dir in adapter_dir.iterdir():
                if not method_dir.is_dir():
                    continue
                
                method_name = method_dir.name
                expected_hash_file = method_dir / "expected_hash.txt"
                
                if expected_hash_file.exists():
                    with open(expected_hash_file) as f:
                        expected_hash = f.read().strip()
                    
                    current_hash_file = method_dir / "output.json"
                    if current_hash_file.exists():
                        import hashlib
                        with open(current_hash_file, 'rb') as f:
                            current_hash = hashlib.sha256(f.read()).hexdigest()
                        
                        if expected_hash == current_hash:
                            grid[adapter_name][method_name] = "pass"
                        else:
                            grid[adapter_name][method_name] = "fail"
                    else:
                        grid[adapter_name][method_name] = "pending"
                else:
                    grid[adapter_name][method_name] = "rebaseline"
        
        return dict(grid)
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        metrics = {}
        
        for adapter_name, adapter in self.adapters.items():
            circuit_status = self.circuit_breaker.get_adapter_status(adapter_name)
            
            history = self.performance_history.get(adapter_name, [])
            
            if history:
                recent = history[-100:]
                p50 = self._percentile([h["latency"] for h in recent], 50)
                p95 = self._percentile([h["latency"] for h in recent], 95)
                p99 = self._percentile([h["latency"] for h in recent], 99)
                success_rate = sum(1 for h in recent if h.get("success", True)) / len(recent) * 100
            else:
                p50, p95, p99 = 0, 0, 0
                success_rate = 100.0
            
            metrics[adapter_name] = {
                "p50_latency": p50,
                "p95_latency": p95,
                "p99_latency": p99,
                "success_rate": success_rate,
                "circuit_state": circuit_status.get("state", "CLOSED"),
                "total_calls": circuit_status.get("total_calls", 0),
                "recent_failures": circuit_status.get("recent_failures", 0)
            }
        
        return metrics
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def _serialize_dependency_graph(self, validation_results: Dict) -> Dict[str, Any]:
        nodes = []
        edges = []
        
        gate_results = {r["gate_name"]: r for r in validation_results.get("results", [])}
        
        for node in self.dependency_graph.nodes(data=True):
            node_id, node_data = node
            
            circuit_status = self.circuit_breaker.get_adapter_status(node_id)
            state = circuit_status.get("state", "CLOSED")
            
            if state == "CLOSED":
                color = "#28a745"
                status = "healthy"
            elif state == "HALF_OPEN":
                color = "#ffc107"
                status = "recovering"
            elif state == "OPEN":
                color = "#dc3545"
                status = "failed"
            else:
                color = "#6c757d"
                status = "unknown"
            
            nodes.append({
                "id": node_id,
                "label": node_id.replace("_", " ").title(),
                "method_count": node_data.get("method_count", 0),
                "color": color,
                "status": status,
                "circuit_state": state
            })
        
        for edge in self.dependency_graph.edges(data=True):
            source, target, edge_data = edge
            
            source_status = self.circuit_breaker.get_adapter_status(source)
            target_status = self.circuit_breaker.get_adapter_status(target)
            
            if source_status.get("recent_failures", 0) > 0:
                color = "#dc3545"
                impact = "propagating"
            else:
                color = "#6c757d"
                impact = "stable"
            
            edges.append({
                "from": source,
                "to": target,
                "type": edge_data.get("type", "binding"),
                "color": color,
                "impact": impact
            })
        
        return {
            "nodes": nodes,
            "edges": edges
        }


dashboard = HomeostasisDashboard()


@app.route('/')
def index():
    return render_template('dashboard.html')


@app.route('/api/dashboard')
def get_dashboard():
    try:
        data = dashboard.get_dashboard_data()
        return jsonify(data)
    except Exception as e:
        logger.error(f"Dashboard error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/run_validation')
def run_validation():
    try:
        results = dashboard.pipeline.run_all()
        return jsonify(results)
    except Exception as e:
        logger.error(f"Validation error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/circuit_breaker/<adapter_name>/reset', methods=['POST'])
def reset_circuit(adapter_name):
    try:
        dashboard.circuit_breaker.reset_adapter(adapter_name)
        return jsonify({"success": True, "message": f"Circuit reset for {adapter_name}"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/remediation/<error_code>')
def get_remediation(error_code):
    try:
        suggestions = dashboard.pipeline.remediation.fix_templates.get(error_code, {})
        return jsonify(suggestions)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/performance/<adapter_name>')
def get_adapter_performance(adapter_name):
    try:
        history = dashboard.performance_history.get(adapter_name, [])
        return jsonify({"adapter": adapter_name, "history": history})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/canary_rebaseline/<adapter_name>/<method_name>', methods=['POST'])
def rebaseline_canary(adapter_name, method_name):
    try:
        baseline_dir = Path(f"baselines/{adapter_name}/{method_name}")
        baseline_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = baseline_dir / "output.json"
        if output_file.exists():
            import hashlib
            with open(output_file, 'rb') as f:
                new_hash = hashlib.sha256(f.read()).hexdigest()
            
            hash_file = baseline_dir / "expected_hash.txt"
            hash_file.write_text(new_hash)
            
            return jsonify({
                "success": True,
                "message": f"Rebaselined {adapter_name}.{method_name}",
                "new_hash": new_hash
            })
        else:
            return jsonify({"success": False, "error": "Output file not found"}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(host='0.0.0.0', port=5000, debug=True)
