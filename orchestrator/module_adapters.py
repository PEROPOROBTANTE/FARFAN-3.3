# coding=utf-8
"""
Module Adapters - Adapter Registry for Module Instances
========================================================

Registers module instances, maintains a registry mapping module names to 
adapter instances, and provides lookup methods for dependency injection.

Author: FARFAN 3.0 Team
Version: 3.0.0
Python: 3.10+
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModuleAdapter:
    """Adapter wrapper for a module instance"""
    name: str
    module_instance: Any
    available: bool = True
    description: str = ""


class AdapterRegistry:
    """
    Registry for module adapter instances
    
    Features:
    - Registers module instances by name
    - Maintains mapping of module names to adapter instances
    - Provides lookup methods for dependency injection
    - Tracks adapter availability
    """

    def __init__(self):
        """Initialize empty adapter registry"""
        self._adapters: Dict[str, ModuleAdapter] = {}
        logger.info("AdapterRegistry initialized")

    def register(
        self,
        name: str,
        module_instance: Any,
        description: str = ""
    ):
        """
        Register a module adapter
        
        Args:
            name: Unique name for the module
            module_instance: Instance of the module/adapter
            description: Optional description of the adapter
        """
        adapter = ModuleAdapter(
            name=name,
            module_instance=module_instance,
            available=True,
            description=description
        )
        
        self._adapters[name] = adapter
        logger.info(f"Registered adapter: {name}")

    def get(self, name: str) -> Optional[Any]:
        """
        Get module instance by name
        
        Args:
            name: Name of the module adapter
            
        Returns:
            Module instance or None if not found
        """
        adapter = self._adapters.get(name)
        
        if adapter is None:
            logger.warning(f"Adapter not found: {name}")
            return None
        
        if not adapter.available:
            logger.warning(f"Adapter unavailable: {name}")
            return None
        
        return adapter.module_instance

    def has(self, name: str) -> bool:
        """Check if an adapter is registered"""
        return name in self._adapters

    def is_available(self, name: str) -> bool:
        """Check if an adapter is available"""
        adapter = self._adapters.get(name)
        return adapter is not None and adapter.available

    def set_availability(self, name: str, available: bool):
        """Set adapter availability status"""
        if name in self._adapters:
            self._adapters[name].available = available
            logger.info(f"Adapter {name} availability set to {available}")

    def get_all(self) -> Dict[str, Any]:
        """Get all registered adapter instances"""
        return {
            name: adapter.module_instance
            for name, adapter in self._adapters.items()
            if adapter.available
        }

    def get_available_names(self) -> List[str]:
        """Get list of available adapter names"""
        return [
            name for name, adapter in self._adapters.items()
            if adapter.available
        ]

    def get_all_names(self) -> List[str]:
        """Get list of all registered adapter names"""
        return list(self._adapters.keys())

    def unregister(self, name: str):
        """Unregister an adapter"""
        if name in self._adapters:
            del self._adapters[name]
            logger.info(f"Unregistered adapter: {name}")

    def clear(self):
        """Clear all registered adapters"""
        self._adapters.clear()
        logger.info("Cleared all adapters from registry")

    def get_status(self) -> Dict[str, Any]:
        """
        Get registry status information
        
        Returns:
            Dictionary with counts and availability information
        """
        total = len(self._adapters)
        available = sum(1 for a in self._adapters.values() if a.available)
        
        return {
            "total_adapters": total,
            "available_adapters": available,
            "unavailable_adapters": total - available,
            "adapters": {
                name: {
                    "available": adapter.available,
                    "description": adapter.description
                }
                for name, adapter in self._adapters.items()
            }
        }


if __name__ == "__main__":
    registry = AdapterRegistry()
    
    print("=" * 60)
    print("Module Adapter Registry")
    print("=" * 60)
    
    # Example usage
    class MockAdapter:
        def process(self):
            return "processed"
    
    registry.register("test_adapter", MockAdapter(), "Test adapter")
    registry.register("another_adapter", MockAdapter(), "Another test")
    
    status = registry.get_status()
    print(f"\nTotal adapters: {status['total_adapters']}")
    print(f"Available: {status['available_adapters']}")
    print(f"\nRegistered adapters:")
    for name in registry.get_all_names():
        available = "✓" if registry.is_available(name) else "✗"
        print(f"  {available} {name}")
