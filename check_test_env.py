#!/usr/bin/env python3
"""
Quick environment check for data loading tests.
Run this before running the full test suite.
"""

import sys

def check_environment():
    """Check if all required dependencies are available."""
    print("="*70)
    print("ENVIRONMENT CHECK FOR DATA LOADING TESTS")
    print("="*70)

    checks = []

    # Check Python version
    print(f"\nPython version: {sys.version}")
    if sys.version_info >= (3, 7):
        checks.append(("Python >= 3.7", True, f"✓ {sys.version_info.major}.{sys.version_info.minor}"))
    else:
        checks.append(("Python >= 3.7", False, f"✗ {sys.version_info.major}.{sys.version_info.minor} (need >= 3.7)"))

    # Check numpy
    try:
        import numpy as np
        checks.append(("numpy", True, f"✓ {np.__version__}"))
    except ImportError:
        checks.append(("numpy", False, "✗ Not installed (run: pip install numpy)"))

    # Check h5py
    try:
        import h5py
        checks.append(("h5py", True, f"✓ {h5py.__version__}"))
    except ImportError:
        checks.append(("h5py", False, "✗ Not installed (run: pip install h5py)"))

    # Check torch
    try:
        import torch
        checks.append(("torch", True, f"✓ {torch.__version__}"))
    except ImportError:
        checks.append(("torch", False, "✗ Not installed (run: pip install torch)"))

    # Check if utils module is accessible
    try:
        from utils.helpers import extract_sequences
        checks.append(("utils.helpers", True, "✓ Module found"))
    except ImportError as e:
        checks.append(("utils.helpers", False, f"✗ Cannot import: {e}"))

    # Print results
    print("\nDependency Check:")
    print("-" * 70)
    for name, passed, message in checks:
        print(f"{name:20s} {message}")

    print("\n" + "="*70)

    # Summary
    all_passed = all(passed for _, passed, _ in checks)
    if all_passed:
        print("✅ ENVIRONMENT READY - You can run test_data_loading.py")
        print("   Run: python test_data_loading.py")
    else:
        print("❌ ENVIRONMENT NOT READY - Please install missing dependencies")
        failed = [name for name, passed, _ in checks if not passed]
        print(f"   Missing: {', '.join(failed)}")
        print("\n   Quick fix:")
        print("   pip install numpy h5py torch")

    print("="*70 + "\n")

    return all_passed


if __name__ == "__main__":
    success = check_environment()
    sys.exit(0 if success else 1)
