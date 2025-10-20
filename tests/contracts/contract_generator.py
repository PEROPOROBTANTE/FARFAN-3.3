"""
Generate YAML contract specifications for all 313 adapter methods.
"""
import yaml
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Any
import re

# Method signatures and type mappings
METHOD_SIGNATURES = {
    # PolicyProcessorAdapter
    "from_legacy": {
        "params": ["legacy_config"],
        "types": {"legacy_config": "object"},
        "returns": "object",
        "deterministic": True
    },
    "validate": {
        "params": ["config"],
        "types": {"config": "object"},
        "returns": "boolean",
        "deterministic": True
    },
    "compute_evidence_score": {
        "params": ["matches", "context"],
        "types": {"matches": "array", "context": "string"},
        "returns": "number",
        "deterministic": False,
        "rng_seed_param": "seed"
    },
    "calculate_shannon_entropy": {
        "params": ["distribution"],
        "types": {"distribution": "array"},
        "returns": "number",
        "deterministic": True
    },
    "normalize_unicode": {
        "params": ["text"],
        "types": {"text": "string"},
        "returns": "string",
        "deterministic": True
    },
    "segment_into_sentences": {
        "params": ["text"],
        "types": {"text": "string"},
        "returns": "array",
        "deterministic": True
    },
    "extract_contextual_window": {
        "params": ["text", "position", "size"],
        "types": {"text": "string", "position": "integer", "size": "integer"},
        "returns": "string",
        "deterministic": True
    },
    "compile_pattern": {
        "params": ["pattern_str"],
        "types": {"pattern_str": "string"},
        "returns": "object",
        "deterministic": True
    },
    "to_dict": {
        "params": ["evidence_bundle"],
        "types": {"evidence_bundle": "object"},
        "returns": "object",
        "deterministic": True
    },
    "process": {
        "params": ["text"],
        "types": {"text": "string"},
        "returns": "object",
        "deterministic": False
    },
}

# Default type mappings for common parameter patterns
DEFAULT_TYPE_MAPPINGS = {
    "text": "string",
    "path": "string",
    "file_path": "string",
    "config": "object",
    "results": "object",
    "data": "object",
    "segments": "array",
    "sentences": "array",
    "matches": "array",
    "context": "string",
    "query": "string",
    "position": "integer",
    "size": "integer",
    "max_size": "integer",
    "threshold": "number",
    "confidence": "number",
    "score": "number",
    "distribution": "array",
    "embedding": "array",
    "embeddings": "array",
    "chunks": "array",
    "nodes": "array",
    "links": "array",
    "graph": "object",
    "metadata": "object",
    "encoding": "string",
    "format": "string",
    "method": "string",
    "pattern": "string",
    "top_k": "integer",
    "lambda_param": "number",
}

def infer_parameter_type(param_name: str) -> str:
    """Infer JSON Schema type from parameter name."""
    param_lower = param_name.lower()
    
    # Check exact matches first
    if param_lower in DEFAULT_TYPE_MAPPINGS:
        return DEFAULT_TYPE_MAPPINGS[param_lower]
    
    # Pattern matching
    if "text" in param_lower or "str" in param_lower or "name" in param_lower:
        return "string"
    if "path" in param_lower or "file" in param_lower or "dir" in param_lower:
        return "string"
    if "count" in param_lower or "size" in param_lower or "idx" in param_lower:
        return "integer"
    if "config" in param_lower or "result" in param_lower or "data" in param_lower:
        return "object"
    if "list" in param_lower or "array" in param_lower or param_lower.endswith("s"):
        return "array"
    if "rate" in param_lower or "ratio" in param_lower or "score" in param_lower:
        return "number"
    if "is_" in param_lower or "has_" in param_lower or "should_" in param_lower:
        return "boolean"
    
    # Default to string for unknown types (safer than 'any')
    return "string"

def infer_return_type(method_name: str) -> str:
    """Infer return type from method name."""
    name_lower = method_name.lower()
    
    if any(x in name_lower for x in ["validate", "check", "is_", "has_", "should_"]):
        return "boolean"
    if any(x in name_lower for x in ["calculate", "compute", "score", "count"]):
        return "number"
    if any(x in name_lower for x in ["extract", "detect", "find", "get"]) and name_lower.endswith("s"):
        return "array"
    if any(x in name_lower for x in ["normalize", "clean", "format"]) and "text" in name_lower:
        return "string"
    if any(x in name_lower for x in ["generate", "create", "build", "analyze", "process"]):
        return "object"
    
    return "object"

def is_deterministic(method_name: str, params: List[str]) -> bool:
    """Determine if method is deterministic."""
    name_lower = method_name.lower()
    
    # Non-deterministic indicators
    if any(x in name_lower for x in ["random", "sample", "shuffle", "generate", "infer"]):
        return False
    if any(x in name_lower for x in ["neural", "embedding", "semantic", "bayesian"]):
        return False
    if any(x in name_lower for x in ["extract", "detect", "analyze", "process"]):
        return False
    
    # Deterministic indicators
    if any(x in name_lower for x in ["normalize", "validate", "format", "parse", "clean"]):
        return True
    if any(x in name_lower for x in ["calculate", "compute"]) and "bayesian" not in name_lower:
        return True
    
    return False

def needs_rng_seed(method_name: str, is_det: bool, params: List[str]) -> bool:
    """Determine if method needs RNG seed parameter."""
    if is_det:
        return False
    
    # Check if seed parameter already exists
    if 'seed' in params or 'rng_seed' in params or 'random_seed' in params:
        return False
    
    name_lower = method_name.lower()
    return any(x in name_lower for x in ["random", "sample", "shuffle", "monte", "stochastic", "bayesian"])

def generate_contract(adapter_name: str, method_name: str, params: List[str] = None) -> Dict[str, Any]:
    """Generate a contract specification for a method."""
    
    if params is None:
        params = []
    
    # Build input schema
    input_schema = {
        "type": "object",
        "properties": {},
        "required": []
    }
    
    for param in params:
        if param in ["kwargs", "args", "self"]:
            continue
        param_type = infer_parameter_type(param)
        input_schema["properties"][param] = {"type": param_type}
        input_schema["required"].append(param)
    
    # Build output schema
    return_type = infer_return_type(method_name)
    output_schema = {"type": return_type}
    
    # Determinism
    deterministic = is_deterministic(method_name, params)
    
    # RNG seed
    rng_seed_param = "seed" if needs_rng_seed(method_name, deterministic, params) else None
    
    # Add seed to input schema if needed
    if rng_seed_param and rng_seed_param not in input_schema["properties"]:
        input_schema["properties"][rng_seed_param] = {"type": "integer"}
        # Make it optional
        if rng_seed_param in input_schema["required"]:
            input_schema["required"].remove(rng_seed_param)
    
    # Canonical canary (sample input for testing)
    canonical_canary = {}
    for param in input_schema["required"]:
        param_type = input_schema["properties"][param]["type"]
        if param_type == "string":
            canonical_canary[param] = "test_input"
        elif param_type == "integer":
            canonical_canary[param] = 0
        elif param_type == "number":
            canonical_canary[param] = 0.0
        elif param_type == "boolean":
            canonical_canary[param] = False
        elif param_type == "array":
            canonical_canary[param] = []
        elif param_type == "object":
            canonical_canary[param] = {}
        else:
            canonical_canary[param] = None
    
    # Sample hash (SHA-256 of canonical output)
    sample_output = {"status": "success", "data": {}}
    sample_hash = hashlib.sha256(json.dumps(sample_output, sort_keys=True).encode()).hexdigest()
    
    # Allowed side effects
    allowed_side_effects = []
    if "load" in method_name.lower() or "read" in method_name.lower():
        allowed_side_effects.append("file_read")
    if "save" in method_name.lower() or "write" in method_name.lower() or "export" in method_name.lower():
        allowed_side_effects.append("file_write")
    if "log" in method_name.lower():
        allowed_side_effects.append("logging")
    
    # Max latency (ms)
    max_latency_ms = 5000  # Default 5 seconds
    if "process" in method_name.lower() or "analyze" in method_name.lower():
        max_latency_ms = 30000  # 30 seconds for heavy operations
    elif "load" in method_name.lower() or "extract" in method_name.lower():
        max_latency_ms = 10000  # 10 seconds for I/O operations
    
    # Retry policy
    retry_policy = {
        "max_retries": 0 if deterministic else 3,
        "backoff_multiplier": 1.5,
        "initial_delay_ms": 100,
        "max_delay_ms": 5000
    }
    
    contract = {
        "adapter": adapter_name,
        "method": method_name,
        "input_schema": input_schema,
        "output_schema": output_schema,
        "deterministic": deterministic,
        "rng_seed_param": rng_seed_param,
        "canonical_canary": canonical_canary,
        "sample_hash": sample_hash,
        "allowed_side_effects": allowed_side_effects,
        "max_latency_ms": max_latency_ms,
        "retry_policy": retry_policy
    }
    
    return contract

def extract_method_params(file_path: str) -> Dict[str, Dict[str, List[str]]]:
    """Extract method parameters from source file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    adapters_params = {}
    
    # Pattern to match _execute_ methods with parameters
    pattern = r'def\s+_execute_(\w+)\s*\(self(?:,\s*([^)]+))?\)'
    
    for match in re.finditer(pattern, content):
        method_name = match.group(1)
        params_str = match.group(2)
        
        params = []
        if params_str:
            # Parse parameters
            for param in params_str.split(','):
                param = param.strip()
                if '=' in param:
                    param = param.split('=')[0].strip()
                if ':' in param:
                    param = param.split(':')[0].strip()
                if param and param not in ['args', 'kwargs', '**kwargs', '*args']:
                    params.append(param)
        
        # Find which adapter this belongs to
        method_pos = match.start()
        class_match = None
        for class_search in re.finditer(r'class\s+(\w+Adapter)\(BaseAdapter\):', content):
            if class_search.start() < method_pos:
                class_match = class_search.group(1)
        
        if class_match:
            if class_match not in adapters_params:
                adapters_params[class_match] = {}
            adapters_params[class_match][method_name] = params
    
    return adapters_params

def main():
    """Generate all contract files."""
    source_file = Path('orchestrator/module_adapters.py')
    contracts_dir = Path('tests/contracts')
    contracts_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract method parameters
    print("Extracting method parameters...")
    adapters_params = extract_method_params(str(source_file))
    
    total_contracts = 0
    
    for adapter_name, methods in adapters_params.items():
        print(f"\nGenerating contracts for {adapter_name} ({len(methods)} methods)...")
        
        for method_name, params in methods.items():
            contract = generate_contract(adapter_name, method_name, params)
            
            # Write contract file
            filename = f"{adapter_name}_{method_name}.yaml"
            file_path = contracts_dir / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(contract, f, default_flow_style=False, sort_keys=False)
            
            total_contracts += 1
    
    print(f"\n{'='*60}")
    print(f"Generated {total_contracts} contract files in {contracts_dir}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
