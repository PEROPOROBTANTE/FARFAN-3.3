# Fault Injection Testing Framework - FARFAN 3.0

Framework completo de inyección de fallos para validar resiliencia del sistema FARFAN 3.0 contra los 9 adapters.

## Arquitectura

### Componentes Principales

#### 1. **Injectors** (`injectors.py`)
Cuatro categorías de inyección de fallos:

- **ContractFaultInjector**: Violaciones de contrato
  - Type mismatches (tipos incorrectos en args/returns)
  - Missing bindings (dependencias faltantes entre adapters)
  - Schema breaks (ModuleResult mal formado, YAML corrupto)

- **DeterminismFaultInjector**: Ruptura de determinismo
  - Seed corruption (corrompe random/numpy seeds)
  - Non-reproducible outputs (timestamps, ruido aleatorio)

- **FaultToleranceFaultInjector**: Fallos en fault tolerance
  - Circuit breaker stuck (estados atascados OPEN/CLOSED)
  - Wrong thresholds (umbrales muy bajos/altos)
  - Retry storms (reintentos excesivos sin backoff)
  - Timeout misconfigurations (prematuros/infinitos)

- **OperationalFaultInjector**: Fallos operacionales del sistema
  - Disk full (IOError en escritura)
  - Clock skew (manipulación de time.time())
  - Network partitions (connection failures)
  - Memory pressure (presión de memoria)

#### 2. **ResilienceValidator** (`resilience_validator.py`)
Validador que ejecuta test scenarios contra los 9 adapters:

**Validaciones:**
- ✅ **Circuit Breaker State Transitions**: Verifica secuencia CLOSED → OPEN → HALF_OPEN → RECOVERING → ISOLATED
- ✅ **Retry Backoff**: Valida exponential backoff con jitter (delay = base * 2^retry + random_jitter)
- ✅ **Timeout Enforcement**: Respeta max_latency_ms de contracts
- ✅ **Idempotency Detection**: Previene ejecuciones duplicadas
- ✅ **Graceful Degradation**: No cascading failures entre adapters

#### 3. **ChaosScenarioRunner** (`chaos_scenarios.py`)
Ejecuta chaos tests combinando múltiples fault types simultáneamente:

**Scenarios:**
1. **Partial Failure**: 1-3 adapters fallan simultáneamente
2. **Cascading Risk**: Fallo en dependency chain (policy_processor → semantic → analyzer_one)
3. **Network Partition**: Simulación de network issues
4. **Resource Exhaustion**: Memory + disk pressure
5. **Timing Issues**: Clock skew + timeouts
6. **Contract Violations**: Type mismatches + schema breaks combinados
7. **Determinism Break**: Seed corruption + random noise
8. **Combined Chaos**: Múltiples categorías simultáneas (extreme testing)

## Los 9 Adapters Validados

```python
ADAPTERS = [
    "teoria_cambio",              # ModulosAdapter (51 methods)
    "analyzer_one",               # AnalyzerOneAdapter (39 methods)
    "dereck_beach",               # DerekBeachAdapter (89 methods)
    "embedding_policy",           # EmbeddingPolicyAdapter (37 methods)
    "semantic_chunking_policy",   # SemanticChunkingPolicyAdapter (18 methods)
    "contradiction_detection",    # ContradictionDetectionAdapter (52 methods)
    "financial_viability",        # FinancialViabilityAdapter (60 methods)
    "policy_processor",           # PolicyProcessorAdapter (34 methods)
    "policy_segmenter"            # PolicySegmenterAdapter (33 methods)
]
```

**Total**: 413 methods across 9 adapters

## Uso

### Instalación
```bash
pip install pytest
```

### Ejecutar Tests Completos
```bash
# Todos los tests
pytest tests/test_fault_injection_framework.py -v

# Solo validaciones de circuit breaker
pytest tests/test_fault_injection_framework.py::TestResilienceValidator::test_validate_circuit_breaker_transitions -v

# Solo chaos scenarios
pytest tests/test_fault_injection_framework.py::TestChaosScenarioRunner -v

# Excluir tests lentos
pytest tests/test_fault_injection_framework.py -v -m "not slow"
```

### Uso Programático

#### Ejemplo 1: Validar un Adapter
```python
from tests.fault_injection import ResilienceValidator

# Crear validator
validator = ResilienceValidator()

# Ejecutar todas las validaciones para un adapter
results = validator.run_all_validations("teoria_cambio")

# Generar reporte
report = validator.generate_report()
print(f"Success rate: {report['summary']['success_rate']:.1%}")
```

#### Ejemplo 2: Inyectar Fallo Específico
```python
from tests.fault_injection import ContractFaultInjector

injector = ContractFaultInjector()

# Inyectar type mismatch
fault = injector.inject_type_mismatch(
    adapter_name="analyzer_one",
    method_name="analyze",
    expected_type=dict,
    injected_value="wrong_type_string"
)

# Verificar fallo inyectado
print(f"Fault injected: {fault.description}")
```

#### Ejemplo 3: Chaos Testing
```python
from tests.fault_injection import ChaosScenarioRunner

runner = ChaosScenarioRunner()

# Ejecutar un scenario específico
scenario = runner.build_combined_chaos_scenario()
result = runner.run_scenario(scenario)

print(f"Status: {result.status.value}")
print(f"Cascading failures: {len(result.cascading_failures)}")
print(f"Graceful degradation: {result.graceful_degradation}")

# O ejecutar todos los scenarios
results = runner.run_all_scenarios()
report = runner.generate_chaos_report()
```

## Circuit Breaker State Machine

```
       failures >= threshold
CLOSED ────────────────────────> OPEN
  ^                                 │
  │                                 │ recovery_timeout elapsed
  │                                 v
  └──── all test calls succeed ── HALF_OPEN
                                    │
                                    │ test call fails
                                    v
                                  OPEN (retry)
                                    │
                                    │ multiple failures
                                    v
                                  ISOLATED (critical)
```

## Validación de Retry Backoff

**Estrategia Esperada**: Exponential backoff with jitter

```python
delay = base_delay * (2 ** retry_count) + random_jitter
```

**Ejemplo**:
- Retry 0: ~100ms
- Retry 1: ~200ms (2x)
- Retry 2: ~400ms (2x)
- Retry 3: ~800ms (2x)
- Retry 4: ~1600ms (2x)

Jitter: ±20% para evitar thundering herd

## Contract Schema

Los contracts se definen en `orchestrator/execution_mapping.yaml`:

```yaml
execution_chain:
  - step: 1
    adapter: policy_segmenter
    method: segment
    args:
      - name: text
        type: str
        source: plan_text
    returns:
      type: List[Dict[str, Any]]
      binding: document_segments
    max_latency_ms: 5000
```

El framework valida:
- ✅ Tipos correctos en args/returns
- ✅ Bindings disponibles entre steps
- ✅ Timeouts respetados (max_latency_ms)
- ✅ ModuleResult con schema correcto

## Métricas Reportadas

### Por Adapter
- `state`: Estado del circuit breaker (CLOSED/OPEN/HALF_OPEN/etc)
- `success_rate`: Tasa de éxito (0.0 - 1.0)
- `avg_response_time`: Tiempo promedio de respuesta (ms)
- `recent_failures`: Fallos recientes (ventana de 60s)

### Por Scenario
- `status`: PASSED/FAILED/DEGRADED
- `cascading_failures`: Lista de adapters que fallaron por cascade
- `graceful_degradation`: bool - si hubo degradación graceful
- `circuit_breaker_states`: Estados finales de todos los adapters

### Globales
- `success_rate`: Porcentaje de validaciones pasadas
- `total_cascading_failures`: Total de cascading failures detectados
- `graceful_degradation_rate`: Porcentaje de scenarios con degradación graceful

## Recomendaciones del Sistema

El framework genera recomendaciones automáticas:

```python
recommendations = [
    "⚠️ Detected 3 cascading failures - review adapter isolation boundaries",
    "⚠️ Graceful degradation rate below 80% - improve fallback strategies",
    "⚠️ Circuit breakers stuck in OPEN/ISOLATED: ['teoria_cambio'] - review recovery timeout"
]
```

## Estructura de Archivos

```
tests/fault_injection/
├── __init__.py                 # Package exports
├── injectors.py                # 4 fault injectors
├── resilience_validator.py     # Resilience validation engine
├── chaos_scenarios.py          # Chaos testing scenarios
└── README.md                   # This file

tests/
└── test_fault_injection_framework.py  # Test suite completo
```

## Extensión

### Agregar Nuevo Fault Type

```python
# En injectors.py
def inject_custom_fault(self, adapter_name: str, **kwargs) -> InjectedFault:
    fault = InjectedFault(
        category=FaultCategory.OPERATIONAL,  # o nueva categoría
        severity=FaultSeverity.HIGH,
        description="Custom fault description",
        target_adapter=adapter_name,
        metadata=kwargs
    )
    self.injected_faults.append(fault)
    return fault
```

### Agregar Nuevo Chaos Scenario

```python
# En chaos_scenarios.py
def build_custom_scenario(self) -> ChaosScenario:
    scenario = ChaosScenario(
        name="custom_scenario",
        scenario_type=ChaosScenarioType.COMBINED_CHAOS,
        description="Custom chaos scenario",
        affected_adapters=["adapter1", "adapter2"]
    )
    
    # Inyectar fallos
    fault = self.contract_injector.inject_type_mismatch(...)
    scenario.injected_faults.append(fault)
    
    return scenario
```

## Notas de Implementación

### Circuit Breaker States
- **CLOSED**: Operación normal, todas las requests pasan
- **OPEN**: Bloqueando requests, esperando recovery timeout
- **HALF_OPEN**: Testing recovery con llamadas limitadas
- **RECOVERING**: En proceso de recuperación (transición)
- **ISOLATED**: Estado crítico, múltiples fallos de recovery

### Idempotency Detection
Se usa hash SHA256 de `adapter:method:input` para generar `execution_id` único:
```python
execution_key = f"{adapter_name}:{method_name}:{json.dumps(input, sort_keys=True)}"
execution_id = hashlib.sha256(execution_key.encode()).hexdigest()[:16]
```

### Graceful Degradation Criteria
1. ✅ Adapters no afectados permanecen en CLOSED
2. ✅ Fallback strategies disponibles (use_cached, alternative_adapters)
3. ✅ No excepciones no manejadas / crashes

## Troubleshooting

### "Circuit breaker no transiciona a HALF_OPEN"
→ Verificar `recovery_timeout` suficientemente largo (≥5s para testing)

### "Cascading failures detectados"
→ Revisar dependency graph y agregar circuit breakers en puntos de integración

### "Tests muy lentos"
→ Reducir `recovery_timeout` y `failure_threshold` para testing:
```python
CircuitBreaker(failure_threshold=3, recovery_timeout=5.0)
```

### "Determinism tests fallan"
→ Asegurar que tests usan seeds fijos antes de inyectar corruption

## Referencias

- Circuit Breaker Pattern: `orchestrator/circuit_breaker.py`
- Module Adapters: `orchestrator/module_adapters.py`
- Execution Mapping: `orchestrator/execution_mapping.yaml`
- Choreographer: `orchestrator/choreographer.py`

## Licencia

FARFAN 3.0 - Integration Team
Python 3.10+
