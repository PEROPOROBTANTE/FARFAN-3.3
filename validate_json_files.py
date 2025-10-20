#!/usr/bin/env python3
"""
JSON Files Validator
Validates syntax and indentation consistency of all JSON files in the repository.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class JSONValidator:
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)
        self.results: List[Dict] = []

    def find_json_files(self) -> List[Path]:
        """Find all JSON files excluding venv directories."""
        json_files = []
        for json_file in self.root_path.rglob("*.json"):
            if "venv" not in json_file.parts and ".git" not in json_file.parts:
                json_files.append(json_file)
        return sorted(json_files)

    def detect_indentation(self, content: str) -> Optional[Tuple[int, bool]]:
        """
        Detect indentation type (2 or 4 spaces) and check consistency.
        Returns (indent_size, is_consistent)
        """
        lines = content.split("\n")
        indents = []

        for line in lines:
            if line.strip():
                match = re.match(r"^( +)\S", line)
                if match:
                    spaces = len(match.group(1))
                    indents.append(spaces)

        if not indents:
            return None

        indent_counts: Dict[int, int] = {}
        for indent in indents:
            indent_counts[indent] = indent_counts.get(indent, 0) + 1

        likely_base: Optional[int] = None
        for candidate in [2, 4]:
            multiples = [i for i in indent_counts.keys() if i % candidate == 0]
            if multiples:
                if likely_base is None or sum(
                    indent_counts.get(m, 0) for m in multiples
                ) > sum(
                    indent_counts.get(m, 0)
                    for m in indent_counts.keys()
                    if m % (likely_base or candidate) == 0
                ):
                    likely_base = candidate

        if likely_base is None:
            likely_base = min(indents) if indents else 2

        inconsistent = []
        for indent in indent_counts.keys():
            if indent % likely_base != 0:
                inconsistent.append(indent)

        is_consistent = len(inconsistent) == 0

        return likely_base, is_consistent

    def validate_file(self, file_path: Path) -> Dict:
        """Validate a single JSON file."""
        result = {
            "file": str(file_path),
            "valid_syntax": False,
            "syntax_error": None,
            "indentation_size": None,
            "indentation_consistent": False,
            "indentation_issues": [],
        }

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            try:
                json.loads(content)
                result["valid_syntax"] = True
            except json.JSONDecodeError as e:
                result["syntax_error"] = f"Line {e.lineno}, Column {e.colno}: {e.msg}"
                return result

            indent_info = self.detect_indentation(content)
            if indent_info:
                indent_size, is_consistent = indent_info
                result["indentation_size"] = indent_size
                result["indentation_consistent"] = is_consistent

                if not is_consistent:
                    lines = content.split("\n")
                    for i, line in enumerate(lines, 1):
                        if line.strip():
                            match = re.match(r"^( +)\S", line)
                            if match:
                                spaces = len(match.group(1))
                                if spaces % indent_size != 0:
                                    result["indentation_issues"].append(
                                        f"Line {i}: {spaces} spaces (expected multiple of {indent_size})"
                                    )

        except Exception as e:
            result["syntax_error"] = f"Error reading file: {str(e)}"

        return result

    def validate_all(self) -> List[Dict]:
        """Validate all JSON files."""
        json_files = self.find_json_files()
        print(f"Found {len(json_files)} JSON files to validate\n")

        for json_file in json_files:
            result = self.validate_file(json_file)
            self.results.append(result)

        return self.results

    def print_report(self):
        """Print validation report."""
        print("=" * 80)
        print("JSON VALIDATION REPORT")
        print("=" * 80)
        print()

        files_with_issues = []

        for result in self.results:
            if not result["valid_syntax"]:
                files_with_issues.append(result["file"])
                print(f"❌ {result['file']}")
                print(f"   SYNTAX ERROR: {result['syntax_error']}")
                print()
            elif not result["indentation_consistent"]:
                files_with_issues.append(result["file"])
                print(f"⚠️  {result['file']}")
                indent_size = result["indentation_size"]
                print(
                    f"   INDENTATION ISSUES: Uses {indent_size}-space "
                    "indentation but has inconsistencies"
                )
                for issue in result["indentation_issues"][:5]:
                    print(f"      {issue}")
                if len(result["indentation_issues"]) > 5:
                    extra = len(result["indentation_issues"]) - 5
                    print(f"      ... and {extra} more issues")
                print()
            else:
                print(f"✅ {result['file']}")
                indent_size = result["indentation_size"]
                print(f"   Valid syntax, consistent {indent_size}-space indentation")
                print()

        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total files checked: {len(self.results)}")
        print(f"Files with issues: {len(files_with_issues)}")
        print(f"Files OK: {len(self.results) - len(files_with_issues)}")
        print()

        if files_with_issues:
            print("Files requiring fixes:")
            for file in files_with_issues:
                print(f"  - {file}")
        else:
            print("✅ All JSON files are valid!")

    def fix_file(self, file_path: Path, indent: int = 2) -> bool:
        """Fix a JSON file by reformatting with consistent indentation."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=indent, ensure_ascii=False)
                f.write("\n")

            return True
        except Exception as e:
            print(f"Error fixing {file_path}: {e}")
            return False

    def fix_all_issues(self, indent: int = 2):
        """Fix all files with indentation issues."""
        print("\n" + "=" * 80)
        print("FIXING JSON FILES")
        print("=" * 80)
        print()

        fixed_count = 0
        failed_count = 0

        for result in self.results:
            if result["valid_syntax"] and not result["indentation_consistent"]:
                file_path = Path(result["file"])
                print(f"Fixing {file_path}...")
                if self.fix_file(file_path, indent):
                    fixed_count += 1
                    print(f"  ✅ Fixed with {indent}-space indentation")
                else:
                    failed_count += 1
                    print("  ❌ Failed to fix")

        print()
        print("=" * 80)
        print("FIX SUMMARY")
        print("=" * 80)
        print(f"Files fixed: {fixed_count}")
        print(f"Files failed: {failed_count}")


def main():
    validator = JSONValidator()
    validator.validate_all()
    validator.print_report()

    files_with_issues = [
        r
        for r in validator.results
        if not r["valid_syntax"] or not r["indentation_consistent"]
    ]

    if files_with_issues:
        fixable = [r for r in files_with_issues if r["valid_syntax"]]
        if fixable:
            response = input(
                "\nDo you want to fix indentation issues automatically? (y/n): "
            )
            if response.lower() == "y":
                indent_str = input(
                    "Choose indentation (2 or 4 spaces, default 2): "
                ).strip()
                indent = int(indent_str) if indent_str in ["2", "4"] else 2
                validator.fix_all_issues(indent)


if __name__ == "__main__":
    main()
