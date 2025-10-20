#!/bin/bash
# Canary Regression Detection System - Full Pipeline
# ===================================================

echo "================================================================================"
echo "FARFAN 3.0 - CANARY REGRESSION DETECTION SYSTEM"
echo "================================================================================"
echo ""

# Step 1: Generate canaries (baseline)
echo "STEP 1: Generating baseline canaries for all 413 methods..."
echo "--------------------------------------------------------------------------------"
python tests/canary_generator.py
GENERATE_STATUS=$?

if [ $GENERATE_STATUS -eq 0 ]; then
    echo "✓ Canary generation complete"
else
    echo "✗ Canary generation failed"
    exit 1
fi

echo ""
echo "STEP 2: Running canary regression tests..."
echo "--------------------------------------------------------------------------------"
python tests/canary_runner.py
TEST_STATUS=$?

if [ $TEST_STATUS -eq 0 ]; then
    echo "✓ All canaries passed - no regressions detected"
    exit 0
else
    echo "✗ Canary violations detected"
fi

echo ""
echo "STEP 3: Generating fix operations..."
echo "--------------------------------------------------------------------------------"
python tests/canary_fix_generator.py

echo ""
echo "================================================================================"
echo "CANARY SYSTEM COMPLETE"
echo "================================================================================"
echo ""
echo "Reports generated:"
echo "  - tests/canaries/generation_report.json"
echo "  - tests/canaries/test_report.json"
echo "  - tests/canaries/fix_report.json"
echo ""
echo "To apply automatic fixes:"
echo "  python tests/canary_fix_generator.py --execute-rebaseline"
echo ""
echo "================================================================================"

exit $TEST_STATUS
