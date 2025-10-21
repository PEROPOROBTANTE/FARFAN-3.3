#!/usr/bin/env python3
"""
Canary Test Rebaselining Script for FARFAN 3.0
===============================================

Updates expected hash baselines for canary regression tests when intentional
changes are made to adapter outputs.

Usage:
    python cicd/rebaseline.py --method teoria_cambio
    python cicd/rebaseline.py --all
    python cicd/rebaseline.py --verify

Requires signed changelog entry before rebaselining.

Author: CI/CD Team
Version: 1.0.0
"""

import argparse
import hashlib
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CanaryRebaseline:
    """Rebaseline canary test expected hashes."""
    
    def __init__(self, baselines_path: Path = Path("baselines")):
        self.baselines_path = baselines_path
        self.baselines_path.mkdir(exist_ok=True, parents=True)
        self.changelog_path = Path("CHANGELOG_SIGNED.md")
    
    def rebaseline_method(self, method_name: str, force: bool = False):
        """Rebaseline a specific method's expected hash."""
        logger.info(f"Rebaselining method: {method_name}")
        
        # Check for signed changelog entry
        if not force and not self._has_changelog_entry(method_name):
            logger.error(
                f"No signed changelog entry found for {method_name}. "
                f"Add entry to {self.changelog_path} or use --force"
            )
            return False
        
        # Compute new hash from current output
        method_dir = self.baselines_path / method_name
        method_dir.mkdir(exist_ok=True, parents=True)
        
        output_file = method_dir / "output.json"
        if not output_file.exists():
            logger.error(f"Output file not found: {output_file}")
            logger.info("Generate baseline by running the method first")
            return False
        
        # Compute hash
        with open(output_file, 'rb') as f:
            new_hash = hashlib.sha256(f.read()).hexdigest()
        
        # Save new expected hash
        hash_file = method_dir / "expected_hash.txt"
        with open(hash_file, 'w') as f:
            f.write(new_hash)
        
        # Save metadata
        metadata_file = method_dir / "baseline_metadata.json"
        import json
        metadata = {
            "method": method_name,
            "hash": new_hash,
            "timestamp": datetime.now().isoformat(),
            "reason": "Rebaselined via cicd/rebaseline.py"
        }
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✓ Rebaselined {method_name}: {new_hash[:16]}...")
        return True
    
    def rebaseline_all(self, force: bool = False):
        """Rebaseline all methods with output files."""
        logger.info("Rebaselining all methods")
        
        count = 0
        for method_dir in self.baselines_path.iterdir():
            if not method_dir.is_dir():
                continue
            
            method_name = method_dir.name
            if self.rebaseline_method(method_name, force=force):
                count += 1
        
        logger.info(f"Rebaselined {count} methods")
        return count
    
    def verify_baselines(self) -> List[str]:
        """Verify all baselines match current outputs."""
        logger.info("Verifying baselines")
        
        mismatches = []
        
        for method_dir in self.baselines_path.iterdir():
            if not method_dir.is_dir():
                continue
            
            method_name = method_dir.name
            expected_hash_file = method_dir / "expected_hash.txt"
            output_file = method_dir / "output.json"
            
            if not expected_hash_file.exists():
                logger.warning(f"No expected hash for {method_name}")
                continue
            
            if not output_file.exists():
                logger.warning(f"No output file for {method_name}")
                continue
            
            with open(expected_hash_file) as f:
                expected_hash = f.read().strip()
            
            with open(output_file, 'rb') as f:
                current_hash = hashlib.sha256(f.read()).hexdigest()
            
            if expected_hash != current_hash:
                mismatches.append(method_name)
                logger.warning(f"✗ Mismatch: {method_name}")
                logger.warning(f"  Expected: {expected_hash[:16]}...")
                logger.warning(f"  Current:  {current_hash[:16]}...")
            else:
                logger.info(f"✓ Match: {method_name}")
        
        if mismatches:
            logger.warning(f"Found {len(mismatches)} mismatches")
        else:
            logger.info("All baselines match!")
        
        return mismatches
    
    def _has_changelog_entry(self, method_name: str) -> bool:
        """Check if method has a signed changelog entry."""
        if not self.changelog_path.exists():
            return False
        
        with open(self.changelog_path) as f:
            content = f.read()
        
        return method_name in content


def main():
    parser = argparse.ArgumentParser(
        description="Rebaseline canary test expected hashes"
    )
    parser.add_argument(
        "--method",
        type=str,
        help="Method name to rebaseline"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Rebaseline all methods"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify all baselines without updating"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip changelog verification"
    )
    
    args = parser.parse_args()
    
    rebaseline = CanaryRebaseline()
    
    if args.verify:
        mismatches = rebaseline.verify_baselines()
        return 1 if mismatches else 0
    elif args.all:
        count = rebaseline.rebaseline_all(force=args.force)
        return 0 if count > 0 else 1
    elif args.method:
        success = rebaseline.rebaseline_method(args.method, force=args.force)
        return 0 if success else 1
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    exit(main())
