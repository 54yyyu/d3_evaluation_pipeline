#!/bin/bash
# Convenience script to run data loading tests
# Usage: ./run_tests.sh

set -e  # Exit on error

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║       D3 Data Loading Refactoring - Test Runner                   ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Determine Python command (python3 vs python)
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "❌ Error: Python not found in PATH"
    echo "   Please install Python 3.7 or higher"
    exit 1
fi

echo "Using Python: $PYTHON ($($PYTHON --version))"
echo ""

# Step 1: Check environment
echo "Step 1/2: Checking environment..."
echo "────────────────────────────────────────────────────────────────────"
$PYTHON check_test_env.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Environment check failed!"
    echo "   Please install missing dependencies:"
    echo "   pip install numpy h5py torch"
    exit 1
fi

echo ""
echo "Step 2/2: Running test suite..."
echo "────────────────────────────────────────────────────────────────────"
$PYTHON test_data_loading.py

if [ $? -eq 0 ]; then
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║  ✅ SUCCESS! All tests passed.                                     ║"
    echo "║                                                                    ║"
    echo "║  The data loading refactoring is working correctly!               ║"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    echo ""
    exit 0
else
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║  ❌ FAILURE! Some tests failed.                                    ║"
    echo "║                                                                    ║"
    echo "║  Please review the output above for details.                      ║"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    echo ""
    exit 1
fi
