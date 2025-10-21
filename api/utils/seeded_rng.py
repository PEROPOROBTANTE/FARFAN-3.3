# coding=utf-8
"""
Seeded Random Number Generator Utilities
=========================================

SIN_CARRETA: Deterministic random number generation using mulberry32 and splitmix32
algorithms for reproducible sample data generation.

These algorithms provide:
- Fast performance (critical for API response times)
- Good statistical properties for sample data
- Deterministic output from fixed seeds
- No dependency on system random state

Author: FARFAN 3.0 Team
Version: 1.0.0
Python: 3.10+
"""

from typing import List, Tuple
import struct


class Mulberry32:
    """
    SIN_CARRETA: Mulberry32 PRNG implementation
    
    Fast, high-quality 32-bit generator with excellent statistical properties.
    Period: 2^32 - 1
    
    Reference: https://gist.github.com/tommyettinger/46a874533244883189143505d203312c
    """
    
    def __init__(self, seed: int):
        """
        Initialize generator with seed
        
        Args:
            seed: Integer seed value (will be converted to uint32)
        """
        self.state = seed & 0xFFFFFFFF
    
    def next_uint32(self) -> int:
        """
        Generate next random uint32
        
        Returns:
            Random integer in range [0, 2^32)
        """
        self.state = (self.state + 0x6D2B79F5) & 0xFFFFFFFF
        z = self.state
        z = ((z ^ (z >> 15)) * (z | 1)) & 0xFFFFFFFF
        z ^= z + ((z ^ (z >> 7)) * (z | 61)) & 0xFFFFFFFF
        return (z ^ (z >> 14)) & 0xFFFFFFFF
    
    def next_float(self) -> float:
        """
        Generate random float in range [0, 1)
        
        Returns:
            Random float
        """
        return self.next_uint32() / 0x100000000
    
    def next_int(self, min_val: int, max_val: int) -> int:
        """
        Generate random integer in range [min_val, max_val]
        
        Args:
            min_val: Minimum value (inclusive)
            max_val: Maximum value (inclusive)
            
        Returns:
            Random integer
        """
        range_size = max_val - min_val + 1
        return min_val + (self.next_uint32() % range_size)
    
    def choice(self, items: List) -> any:
        """
        Choose random item from list
        
        Args:
            items: List of items
            
        Returns:
            Random item
        """
        if not items:
            raise ValueError("Cannot choose from empty list")
        return items[self.next_int(0, len(items) - 1)]
    
    def shuffle(self, items: List) -> List:
        """
        Shuffle list in-place using Fisher-Yates algorithm
        
        Args:
            items: List to shuffle
            
        Returns:
            Shuffled list (same reference)
        """
        n = len(items)
        for i in range(n - 1, 0, -1):
            j = self.next_int(0, i)
            items[i], items[j] = items[j], items[i]
        return items


class SplitMix32:
    """
    SIN_CARRETA: SplitMix32 PRNG implementation
    
    Alternative fast generator with good avalanche properties.
    Useful for generating independent seed sequences.
    
    Reference: Based on SplitMix64 adapted to 32-bit
    """
    
    def __init__(self, seed: int):
        """
        Initialize generator with seed
        
        Args:
            seed: Integer seed value (will be converted to uint32)
        """
        self.state = seed & 0xFFFFFFFF
    
    def next_uint32(self) -> int:
        """
        Generate next random uint32
        
        Returns:
            Random integer in range [0, 2^32)
        """
        self.state = (self.state + 0x9E3779B9) & 0xFFFFFFFF
        z = self.state
        z = ((z ^ (z >> 16)) * 0x85EBCA6B) & 0xFFFFFFFF
        z = ((z ^ (z >> 13)) * 0xC2B2AE35) & 0xFFFFFFFF
        return (z ^ (z >> 16)) & 0xFFFFFFFF
    
    def next_float(self) -> float:
        """
        Generate random float in range [0, 1)
        
        Returns:
            Random float
        """
        return self.next_uint32() / 0x100000000
    
    def next_int(self, min_val: int, max_val: int) -> int:
        """
        Generate random integer in range [min_val, max_val]
        
        Args:
            min_val: Minimum value (inclusive)
            max_val: Maximum value (inclusive)
            
        Returns:
            Random integer
        """
        range_size = max_val - min_val + 1
        return min_val + (self.next_uint32() % range_size)


class SeededGenerator:
    """
    SIN_CARRETA: High-level interface for deterministic sample data generation
    
    Combines Mulberry32 and SplitMix32 for various use cases.
    """
    
    def __init__(self, seed: int, algorithm: str = "mulberry32"):
        """
        Initialize seeded generator
        
        Args:
            seed: Base seed for deterministic generation
            algorithm: "mulberry32" or "splitmix32"
        """
        self.seed = seed
        self.algorithm = algorithm
        
        if algorithm == "mulberry32":
            self.rng = Mulberry32(seed)
        elif algorithm == "splitmix32":
            self.rng = SplitMix32(seed)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}. Use 'mulberry32' or 'splitmix32'")
    
    def generate_region_ids(self, count: int) -> List[str]:
        """
        Generate deterministic region IDs
        
        Args:
            count: Number of regions
            
        Returns:
            List of region ID strings
        """
        return [f"REGION_{i+1:03d}" for i in range(count)]
    
    def generate_municipality_ids(self, region_id: str, count: int) -> List[str]:
        """
        Generate deterministic municipality IDs for a region
        
        Args:
            region_id: Parent region ID
            count: Number of municipalities
            
        Returns:
            List of municipality ID strings
        """
        region_num = int(region_id.split("_")[1])
        base = region_num * 100
        return [f"MUN_{base + i:05d}" for i in range(count)]
    
    def generate_score(self, min_val: float = 0.0, max_val: float = 100.0) -> float:
        """
        Generate random score
        
        Args:
            min_val: Minimum score
            max_val: Maximum score
            
        Returns:
            Random score with 2 decimal places
        """
        return round(min_val + self.rng.next_float() * (max_val - min_val), 2)
    
    def generate_name(self, prefix: str = "Entity") -> str:
        """
        Generate deterministic entity name
        
        Args:
            prefix: Name prefix
            
        Returns:
            Generated name
        """
        adjectives = [
            "Norte", "Sur", "Este", "Oeste", "Central", "Alto", "Bajo",
            "Grande", "Pequeño", "Nuevo", "Viejo", "Verde", "Dorado"
        ]
        nouns = [
            "Valle", "Monte", "Río", "Lago", "Sierra", "Campo", "Puerto",
            "Villa", "Ciudad", "Pueblo", "Distrito", "Región"
        ]
        
        adj = self.rng.choice(adjectives)
        noun = self.rng.choice(nouns)
        return f"{prefix} {adj} del {noun}"
    
    def generate_coordinates(self) -> Tuple[float, float]:
        """
        Generate random geographic coordinates for Colombia
        
        Returns:
            Tuple of (latitude, longitude)
        """
        # Colombia approximate bounds: lat 12.5N to -4.2N, lon -81.7W to -66.9W
        lat = round(-4.2 + self.rng.next_float() * 16.7, 6)
        lon = round(-81.7 + self.rng.next_float() * 14.8, 6)
        return lat, lon
    
    def choice(self, items: List) -> any:
        """
        Choose random item from list
        
        Args:
            items: List of items
            
        Returns:
            Random item
        """
        return self.rng.choice(items)


def create_seeded_generator(seed: int, algorithm: str = "mulberry32") -> SeededGenerator:
    """
    SIN_CARRETA: Factory function to create seeded generator
    
    Args:
        seed: Base seed value
        algorithm: "mulberry32" or "splitmix32"
        
    Returns:
        Configured SeededGenerator instance
    """
    return SeededGenerator(seed, algorithm)
