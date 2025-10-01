#!/usr/bin/env python3
"""
Test script for multi-oracle functionality validation.
Tests argument parsing and validation without needing actual model files.
"""

import argparse
import os
import sys

def parse_arguments():
    """Parse command line arguments - copied from main.py"""
    parser = argparse.ArgumentParser(description='Test multi-oracle argument parsing')

    parser.add_argument('--samples', type=str, default='test_samples.npz',
                       help='Path to samples file (.npz or .h5 format) containing synthetic sequences')

    parser.add_argument('--data', type=str, default='test_data.h5',
                       help='Path to data file (.h5 or .npz format) containing test/train sequences')

    parser.add_argument('--model', type=str, default='model1.ckpt',
                       help='Path to model checkpoint file (for single model) or first model (for multi-oracle)')

    parser.add_argument('--model2', type=str, default=None,
                       help='Path to second model checkpoint file (for multi-oracle setup)')

    parser.add_argument('--model3', type=str, default=None,
                       help='Path to third model checkpoint file (for multi-oracle setup)')

    parser.add_argument('--model-type', type=str, default='deepstarr',
                       choices=['deepstarr', 'mpralegnet', 'lentimpra', 'multi-oracle'],
                       help='Type of oracle model (deepstarr, mpralegnet, lentimpra, or multi-oracle for three models)')

    return parser.parse_args()

def validate_inputs(args):
    """Validate multi-oracle arguments - copied from main.py"""
    files_to_check = [
        (args.samples, 'Samples file'),
        (args.data, 'Data file'),
        (args.model, 'Model file')
    ]

    # Check for multi-oracle setup
    if args.model_type == 'multi-oracle':
        if args.model2 is None or args.model3 is None:
            print("Error: Multi-oracle mode requires --model2 and --model3 arguments")
            print("Usage: --model path1 --model2 path2 --model3 path3 --model-type multi-oracle")
            sys.exit(1)
        files_to_check.extend([
            (args.model2, 'Model 2 file'),
            (args.model3, 'Model 3 file')
        ])

    missing_files = []
    for file_path, description in files_to_check:
        if not os.path.exists(file_path):
            missing_files.append(f"{description}: {file_path}")

    if missing_files:
        print("Error: Missing required files:")
        for missing in missing_files:
            print(f"  - {missing}")
        print("\nNote: This is a test script - files don't need to exist for validation testing")
        return False
    return True

def test_multi_oracle_validation():
    """Test multi-oracle validation logic"""
    print("=== Testing Multi-Oracle Argument Parsing ===\n")

    # Test cases
    test_cases = [
        {
            'name': 'Valid single model (deepstarr)',
            'args': ['--model', 'model1.ckpt', '--model-type', 'deepstarr'],
            'should_pass': True
        },
        {
            'name': 'Valid multi-oracle with all three models',
            'args': ['--model', 'model1.ckpt', '--model2', 'model2.ckpt', '--model3', 'model3.ckpt', '--model-type', 'multi-oracle'],
            'should_pass': True
        },
        {
            'name': 'Invalid multi-oracle missing model2',
            'args': ['--model', 'model1.ckpt', '--model3', 'model3.ckpt', '--model-type', 'multi-oracle'],
            'should_pass': False
        },
        {
            'name': 'Invalid multi-oracle missing model3',
            'args': ['--model', 'model1.ckpt', '--model2', 'model2.ckpt', '--model-type', 'multi-oracle'],
            'should_pass': False
        }
    ]

    for test_case in test_cases:
        print(f"Testing: {test_case['name']}")
        print(f"Command: python test_multi_oracle.py {' '.join(test_case['args'])}")

        # Mock sys.argv
        original_argv = sys.argv
        sys.argv = ['test_multi_oracle.py'] + test_case['args']

        try:
            args = parse_arguments()
            print(f"✓ Argument parsing succeeded")
            print(f"  model_type: {args.model_type}")
            print(f"  model: {args.model}")
            if args.model2:
                print(f"  model2: {args.model2}")
            if args.model3:
                print(f"  model3: {args.model3}")

            # Test validation (without file existence check)
            if args.model_type == 'multi-oracle':
                if args.model2 is None or args.model3 is None:
                    print("✗ Validation failed: Missing model2 or model3")
                    if test_case['should_pass']:
                        print(f"  Expected to pass but failed!")
                    else:
                        print(f"  Expected to fail - correct!")
                else:
                    print("✓ Validation passed: All three models provided")
                    if test_case['should_pass']:
                        print(f"  Expected to pass - correct!")
                    else:
                        print(f"  Expected to fail but passed!")
            else:
                print("✓ Single model validation passed")

        except SystemExit:
            print("✗ Argument parsing failed (SystemExit)")
            if not test_case['should_pass']:
                print(f"  Expected to fail - correct!")
            else:
                print(f"  Expected to pass but failed!")
        except Exception as e:
            print(f"✗ Unexpected error: {e}")

        finally:
            sys.argv = original_argv

        print()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Normal argument parsing mode
        args = parse_arguments()
        validate_inputs(args)
        print("✓ Multi-oracle argument validation completed")
    else:
        # Test mode
        test_multi_oracle_validation()