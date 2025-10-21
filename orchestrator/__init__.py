# coding=utf-8
"""
Legacy Orchestrator - DEPRECATED
=================================

This directory has been quarantined as part of adapter registry consolidation.

All orchestrator functionality has been moved to src.orchestrator with a
canonical ModuleAdapterRegistry implementation.

DO NOT USE THIS MODULE. Import from src.orchestrator instead.

Migration Guide:
----------------
OLD (deprecated):
    from orchestrator import core_orchestrator

NEW (correct):
    from src.orchestrator import FARFANOrchestrator
    # or
    from src.orchestrator.adapter_registry import ModuleAdapterRegistry
    from src.orchestrator.choreographer import ExecutionChoreographer

SIN_CARRETA Compliance:
-----------------------
This explicit ImportError replaces ambiguous import paths that could lead
to non-deterministic runtime selection depending on PYTHONPATH ordering.

Rationale:
----------
Two parallel trees (orchestrator/ and src/orchestrator/) created import
ambiguity. Consolidation to src.orchestrator provides deterministic imports
and a single source of truth for adapter registry implementation.

See: CODE_FIX_REPORT.md for full migration details.
"""

raise ImportError(
    "\n"
    "=" * 80 + "\n"
    "DEPRECATED: orchestrator module has been quarantined\n"
    "=" * 80 + "\n"
    "\n"
    "The top-level 'orchestrator' module is deprecated and has been replaced\n"
    "by 'src.orchestrator' with a canonical ModuleAdapterRegistry implementation.\n"
    "\n"
    "Please update your imports:\n"
    "\n"
    "  OLD: from orchestrator import core_orchestrator\n"
    "  NEW: from src.orchestrator.core_orchestrator import FARFANOrchestrator\n"
    "\n"
    "  OLD: from orchestrator.module_adapters import AdapterRegistry\n"
    "  NEW: from src.orchestrator.adapter_registry import ModuleAdapterRegistry\n"
    "\n"
    "See CODE_FIX_REPORT.md for full migration guide.\n"
    "\n"
    "=" * 80 + "\n"
)
