#!/bin/bash
# FARFAN 3.0 Delivery Package - Quick Verification Script

echo "=========================================="
echo "FARFAN 3.0 Delivery Package Verification"
echo "=========================================="
echo ""

# Check directory structure
echo "✓ Checking directory structure..."
required_dirs=("refactored_code" "tests" "reports" "documentation" "config" "diffs")
for dir in "${required_dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo "  ✅ $dir/ exists"
    else
        echo "  ❌ $dir/ missing"
    fi
done
echo ""

# Check key files
echo "✓ Checking key files..."
required_files=(
    "README_DELIVERY.md"
    "EXECUTION_INSTRUCTIONS.md"
    "reports/audit_trail.md"
    "reports/compatibility_matrix.csv"
    "reports/preservation_metrics.json"
    "config/requirements.txt"
    "config/execution_mapping.yaml"
)
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file exists"
    else
        echo "  ❌ $file missing"
    fi
done
echo ""

# Check refactored code
echo "✓ Checking refactored code modules..."
if [ -d "refactored_code/orchestrator" ]; then
    orchestrator_files=$(ls refactored_code/orchestrator/*.py 2>/dev/null | wc -l)
    echo "  📦 Orchestrator modules: $orchestrator_files"
fi
if [ -d "refactored_code" ]; then
    module_files=$(ls refactored_code/*.py 2>/dev/null | wc -l)
    echo "  📦 Analysis modules: $module_files"
fi
echo ""

# Check documentation
echo "✓ Checking documentation..."
if [ -d "documentation" ]; then
    doc_files=$(ls documentation/*.md 2>/dev/null | wc -l)
    echo "  📄 Documentation files: $doc_files"
fi
echo ""

echo "=========================================="
echo "Verification complete!"
echo ""
echo "Next steps:"
echo "1. Review README_DELIVERY.md for overview"
echo "2. Follow EXECUTION_INSTRUCTIONS.md for validation"
echo "3. Run: pip install -r config/requirements.txt"
echo "=========================================="
