#!/bin/bash
"""
Auto-test Runner with File Monitoring
======================================

Monitors Python file modifications and automatically runs integration tests.

Uses pytest-watch or fallback to inotify-based monitoring.

Author: FARFAN 3.0 Dev Team
Version: 1.0.0
"""

# Check if pytest-watch is available
if command -v ptw &> /dev/null; then
    echo "🔍 Starting pytest-watch..."
    echo "   Monitoring: *.py files"
    echo "   Running: integration smoke tests"
    echo ""
    
    ptw --runner "pytest test_integration_smoke.py -v -m 'not slow'" \
        --patterns "*.py" \
        --ignore .git \
        --ignore venv \
        --ignore __pycache__ \
        --clear
else
    echo "⚠️  pytest-watch not found"
    echo "   Install with: pip install pytest-watch"
    echo ""
    echo "   Falling back to inotifywait..."
    
    # Check if inotifywait is available
    if ! command -v inotifywait &> /dev/null; then
        echo "❌ inotifywait not found"
        echo "   Install with: sudo apt-get install inotify-tools (Linux)"
        echo "   Or: brew install watch (macOS)"
        exit 1
    fi
    
    echo "🔍 Starting file monitor..."
    echo "   Monitoring: *.py files"
    echo ""
    
    # Run tests once initially
    pytest test_integration_smoke.py -v -m 'not slow'
    
    # Monitor for changes
    while true; do
        inotifywait -q -r -e modify,create,delete --exclude '(__pycache__|\.pyc|\.git|venv)' . | while read -r directory event file; do
            if [[ "$file" == *.py ]]; then
                echo ""
                echo "📝 Change detected: $file"
                echo "🔄 Running tests..."
                echo ""
                
                pytest test_integration_smoke.py -v -m 'not slow'
                
                echo ""
                echo "✅ Tests complete. Watching for changes..."
                echo ""
            fi
        done
    done
fi
