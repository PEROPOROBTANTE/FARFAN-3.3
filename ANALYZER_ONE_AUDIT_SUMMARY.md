# Analyzer_one.py Invocation Audit Summary

**Generated:** 2024-01-15  
**Audit File:** `analyzer_one_invocation_audit.json`

## Overview

Complete audit of all classes and methods in `Analyzer_one.py`, cross-referenced with their usage in `AnalyzerOneAdapter` (`orchestrator/module_adapters.py`) to identify invocation pattern violations.

## Statistics

| Metric | Count |
|--------|-------|
| **Total Classes** | 10 |
| **Total Methods** | 39 |
| **Static Methods** | 6 |
| **Class Methods** | 0 |
| **Instance Methods** | 23 |
| **Constructors** | 10 |
| **Violations Found** | 5 |
| ↳ High Severity | 3 |
| ↳ Medium Severity | 2 |
| ↳ Low Severity | 0 |

## Class Inventory

### 1. ValueChainLink
- **Type:** Dataclass
- **Methods:** 1 constructor
- **Line:** 77

### 2. MunicipalOntology
- **Type:** Class
- **Methods:** 1 constructor
- **Line:** 92
- **Invocation:** `MunicipalOntology()`

### 3. SemanticAnalyzer
- **Type:** Class  
- **Line:** 151
- **Methods:** 1 constructor + 1 public + 7 private
- **Constructor Signature:** `SemanticAnalyzer(ontology)`
- **Public Methods:**
  - `extract_semantic_cube(document_segments)` - line 158
- **Private Methods:**
  - `_empty_semantic_cube()` - line 212
  - `_vectorize_segments(segments)` - line 231
  - `_process_segment(segment, idx, vector)` - line 244
  - `_classify_value_chain_link(segment)` - line 282
  - `_classify_policy_domain(segment)` - line 303
  - `_classify_cross_cutting_themes(segment)` - line 317
  - `_calculate_semantic_complexity(semantic_cube)` - line 331

### 4. PerformanceAnalyzer
- **Type:** Class
- **Line:** 381
- **Methods:** 1 constructor + 1 public + 4 private
- **Constructor Signature:** `PerformanceAnalyzer(ontology)`
- **Public Methods:**
  - `analyze_performance(semantic_cube)` - line 388
- **Private Methods:**
  - `_calculate_throughput_metrics(segments, link_config)` - line 423
  - `_detect_bottlenecks(segments, link_config)` - line 462
  - `_calculate_loss_functions(metrics, link_config)` - line 496
  - `_generate_recommendations(performance_analysis)` - line 533

### 5. TextMiningEngine
- **Type:** Class
- **Line:** 557
- **Methods:** 1 constructor + 1 public + 4 private
- **Constructor Signature:** `TextMiningEngine(ontology)`
- **Public Methods:**
  - `diagnose_critical_links(semantic_cube, performance_analysis)` - line 574
- **Private Methods:**
  - `_identify_critical_links(performance_analysis)` - line 615
  - `_analyze_link_text(segments)` - line 640
  - `_assess_risks(segments, text_analysis)` - line 675
  - `_generate_interventions(link_name, risk_assessment, text_analysis)` - line 703

### 6. MunicipalAnalyzer
- **Type:** Class
- **Line:** 739
- **Methods:** 1 constructor + 1 public + 2 private
- **Constructor Signature:** `MunicipalAnalyzer()`
- **Public Methods:**
  - `analyze_document(document_path)` - line 747
- **Private Methods:**
  - `_load_document(document_path)` - line 784
  - `_generate_summary(semantic_cube, performance_analysis, critical_diagnosis)` - line 807

### 7. DocumentProcessor
- **Type:** Utility Class (Static Methods Only)
- **Line:** 980
- **Methods:** 3 static methods
- **Static Methods:**
  - `load_pdf(pdf_path)` - line 984
  - `load_docx(docx_path)` - line 1000
  - `segment_text(text, method='sentence')` - line 1016
- **Correct Invocation:** `DocumentProcessor.method_name()` (no instantiation)

### 8. ResultsExporter
- **Type:** Utility Class (Static Methods Only)
- **Line:** 1062
- **Methods:** 3 static methods
- **Static Methods:**
  - `export_to_json(results, output_path)` - line 1066
  - `export_to_excel(results, output_path)` - line 1076
  - `export_summary_report(results, output_path)` - line 1146
- **Correct Invocation:** `ResultsExporter.method_name()` (no instantiation)

### 9. ConfigurationManager
- **Type:** Class
- **Line:** 1255
- **Methods:** 1 constructor + 3 instance methods
- **Constructor Signature:** `ConfigurationManager(config_path)`
- **Instance Methods:**
  - `load_config()` - line 1261
  - `save_config(config)` - line 1276
  - `validate_config(config)` - line 1284

### 10. BatchProcessor
- **Type:** Class
- **Line:** 1306
- **Methods:** 1 constructor + 2 public + 1 private
- **Constructor Signature:** `BatchProcessor(analyzer)`
- **Public Methods:**
  - `process_directory(directory_path, pattern='*.txt')` - line 1310
  - `export_batch_results(batch_results, output_dir)` - line 1358
- **Private Methods:**
  - `_create_batch_summary(batch_results, output_path)` - line 1371

## Violations Found in AnalyzerOneAdapter

### 🔴 HIGH SEVERITY (3 violations)

#### Violation 1: Missing Required Parameter
- **Location:** `orchestrator/module_adapters.py:1730`
- **Code:** `analyzer = self.SemanticAnalyzer()`
- **Issue:** `SemanticAnalyzer` constructor requires `ontology` parameter
- **Fix:** 
  ```python
  ontology = self.MunicipalOntology()
  analyzer = self.SemanticAnalyzer(ontology)
  ```

#### Violation 2: Missing Required Parameter
- **Location:** `orchestrator/module_adapters.py:1750`
- **Code:** `analyzer = self.PerformanceAnalyzer()`
- **Issue:** `PerformanceAnalyzer` constructor requires `ontology` parameter
- **Fix:**
  ```python
  ontology = self.MunicipalOntology()
  analyzer = self.PerformanceAnalyzer(ontology)
  ```

#### Violation 3: Incorrect Parameter Type
- **Location:** `orchestrator/module_adapters.py:1820`
- **Code:** `complexity = analyzer._calculate_semantic_complexity(segment)`
- **Issue:** Method expects `semantic_cube` (Dict) but receives `segment` (str)
- **Fix:**
  ```python
  complexity = analyzer._calculate_semantic_complexity(semantic_cube)
  ```

### 🟡 MEDIUM SEVERITY (2 violations)

#### Violation 4: Incorrect Static Method Invocation
- **Location:** `orchestrator/module_adapters.py:1880`
- **Code:** `text = self.DocumentProcessor().load_pdf(file_path)`
- **Issue:** Static method called with unnecessary instantiation
- **Fix:**
  ```python
  text = self.DocumentProcessor.load_pdf(file_path)
  ```

#### Violation 5: Incorrect Static Method Invocation
- **Location:** `orchestrator/module_adapters.py:1900`
- **Code:** `self.ResultsExporter().export_to_json(results, path)`
- **Issue:** Static method called with unnecessary instantiation
- **Fix:**
  ```python
  self.ResultsExporter.export_to_json(results, path)
  ```

## Correct Invocation Patterns

### For Classes Requiring Ontology

```python
# SemanticAnalyzer, PerformanceAnalyzer, TextMiningEngine
ontology = self.MunicipalOntology()
analyzer = self.SemanticAnalyzer(ontology)
analyzer.extract_semantic_cube(document_segments)
```

### For Static Method Classes

```python
# DocumentProcessor, ResultsExporter
# DO NOT instantiate - call directly on class
text = self.DocumentProcessor.load_pdf(file_path)
self.ResultsExporter.export_to_json(results, path)
```

### For Regular Instance Classes

```python
# MunicipalAnalyzer, ConfigurationManager, BatchProcessor
analyzer = self.MunicipalAnalyzer()
results = analyzer.analyze_document(document_path)
```

## Recommendations

1. **Fix High Severity Violations First:** All 3 high-severity violations involve missing required constructor parameters that will cause runtime failures.

2. **Optimize Static Method Calls:** The 2 medium-severity violations involve unnecessary object instantiation for static methods, creating overhead without benefit.

3. **Type Safety:** Consider adding type hints to adapter methods to catch parameter type mismatches at development time.

4. **Parameter Validation:** Add runtime validation in adapter methods to verify correct parameter types before calling underlying module methods.

5. **Unit Tests:** Create unit tests for AnalyzerOneAdapter that verify correct instantiation and method invocation patterns.

## Files Generated

- **`analyzer_one_invocation_audit.json`** - Complete machine-readable audit report with full method inventory and correction matrix

---

**Next Steps:** Review and apply fixes from the correction matrix to `orchestrator/module_adapters.py` starting with high-severity violations.
