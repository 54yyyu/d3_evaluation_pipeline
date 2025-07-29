#!/usr/bin/env python3
"""
Main runner script for D3 sequence analysis.

This script provides a unified interface to run all sequence analysis tasks:
- Attribution consistency analysis  
- Functional similarity analysis
- Motif enrichment and co-occurrence analysis

Usage:
    python main.py --samples samples.npz --data DeepSTARR_data.h5 --model oracle_DeepSTARR_DeepSTARR_data.ckpt
    
    Or with environment variables:
    SAMPLES_FILE=samples.npz DATA_FILE=DeepSTARR_data.h5 MODEL_FILE=oracle.ckpt python main.py
"""

import argparse
import os
import sys
from datetime import datetime
import numpy as np
import torch
import h5py
import pickle
import json

# Add parent directory to path to import modules (needed for deepstarr)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from deepstarr import PL_DeepSTARR
from utils.helpers import extract_data, numpy_to_tensor, load_deepstarr
from core.attribution_analysis import run_attribution_consistency_analysis
from core.functional_similarity import run_functional_similarity_analysis  
from core.motif_analysis import run_motif_analysis


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run D3 sequence analysis')
    
    parser.add_argument('--samples', type=str, 
                       default=os.getenv('SAMPLES_FILE', 'samples.npz'),
                       help='Path to samples NPZ file')
    
    parser.add_argument('--data', type=str,
                       default=os.getenv('DATA_FILE', 'DeepSTARR_data.h5'), 
                       help='Path to DeepSTARR data H5 file')
    
    parser.add_argument('--model', type=str,
                       default=os.getenv('MODEL_FILE', 'oracle_DeepSTARR_DeepSTARR_data.ckpt'),
                       help='Path to model checkpoint file')
    
    parser.add_argument('--output-dir', type=str,
                       default=os.getenv('OUTPUT_DIR', 'results'),
                       help='Output directory for results')
    
    parser.add_argument('--skip-attribution', action='store_true',
                       help='Skip attribution consistency analysis')
    
    parser.add_argument('--skip-functional', action='store_true', 
                       help='Skip functional similarity analysis')
    
    parser.add_argument('--skip-motif', action='store_true',
                       help='Skip motif analysis')
    
    return parser.parse_args()


def validate_inputs(args):
    """Validate that all required input files exist."""
    files_to_check = [
        (args.samples, 'Samples file'),
        (args.data, 'Data file'), 
        (args.model, 'Model file')
    ]
    
    missing_files = []
    for file_path, description in files_to_check:
        if not os.path.exists(file_path):
            missing_files.append(f"{description}: {file_path}")
    
    if missing_files:
        print("Error: Missing required files:")
        for missing in missing_files:
            print(f"  - {missing}")
        sys.exit(1)


def setup_output_directory(output_dir):
    """Create output directory with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    full_output_dir = os.path.join(output_dir, f"analysis_results_{timestamp}")
    os.makedirs(full_output_dir, exist_ok=True)
    return full_output_dir


def save_progress_file(output_dir, completed_analyses, all_results):
    """Save a progress file with completed analyses and summary."""
    progress_file = os.path.join(output_dir, "analysis_progress.json")
    
    # Create summary of results for JSON serialization
    results_summary = {}
    for analysis_name, results in all_results.items():
        summary = {}
        for key, value in results.items():
            if isinstance(value, (int, float, str, bool)):
                summary[key] = value
            elif hasattr(value, 'shape'):
                if value.shape == ():  # numpy scalar
                    summary[key] = value.item()
                else:  # numpy arrays
                    summary[key] = f"Array with shape {value.shape}"
            else:
                summary[key] = str(type(value).__name__)
        results_summary[analysis_name] = summary
    
    progress_data = {
        "timestamp": datetime.now().isoformat(),
        "completed_analyses": completed_analyses,
        "total_analyses": len([a for a in ["attribution", "functional", "motif"] if a not in completed_analyses]),
        "results_summary": results_summary,
        "output_directory": output_dir
    }
    
    with open(progress_file, 'w') as f:
        json.dump(progress_data, f, indent=2)
    
    print(f"✓ Progress saved to: {progress_file}")


def save_combined_results(output_dir, all_results):
    """Save all results in a combined pickle file."""
    combined_file = os.path.join(output_dir, "all_results_combined.pkl")
    
    with open(combined_file, 'wb') as f:
        pickle.dump(all_results, f)
    
    print(f"✓ Combined results saved to: {combined_file}")


def print_analysis_summary(analysis_name, results):
    """Print a summary of analysis results."""
    print(f"\n{analysis_name.replace('_', ' ').title()} Results:")
    for key, value in results.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.6f}")
        elif hasattr(value, 'shape'):
            if value.shape == ():  # numpy scalar
                print(f"  {key}: {value.item():.6f}")
            else:  # numpy arrays
                print(f"  {key}: Array with shape {value.shape}")
        else:
            print(f"  {key}: {type(value).__name__}")
    print(f"✓ {analysis_name.replace('_', ' ').title()} completed and saved")


def load_data_and_model(args):
    """Load all required data and model."""
    print("Loading data and model...")
    
    # Load data
    x_test, x_synthetic, x_train = extract_data(args.samples, args.data)
    
    # Convert to tensors
    x_test_tensor = numpy_to_tensor(x_test)
    x_synthetic_tensor = numpy_to_tensor(x_synthetic)
    x_train_tensor = numpy_to_tensor(x_train)
    
    # Load model
    deepstarr = load_deepstarr(args.model)
    
    # Load sample sequences for attribution analysis
    samples = np.load(args.samples)
    sample_seqs = samples['arr_0']
    sample_seqs = torch.tensor(sample_seqs, dtype=torch.float32)
    
    # Load test data for attribution analysis
    DeepSTARR_data = h5py.File(args.data, 'r')
    X_test = torch.tensor(np.array(DeepSTARR_data['X_test']).transpose(0,2,1), dtype=torch.float32)
    
    print(f"Loaded {len(x_test)} test sequences")
    print(f"Loaded {len(x_synthetic)} synthetic sequences") 
    print(f"Loaded {len(x_train)} training sequences")
    print(f"Loaded {len(sample_seqs)} sample sequences")
    
    return {
        'deepstarr': deepstarr,
        'x_test_tensor': x_test_tensor,
        'x_synthetic_tensor': x_synthetic_tensor, 
        'x_train_tensor': x_train_tensor,
        'sample_seqs': sample_seqs,
        'X_test': X_test
    }


def main():
    """Main analysis pipeline with on-the-fly result saving."""
    print("=== D3 Sequence Analysis Pipeline ===")
    
    # Parse arguments and validate inputs
    args = parse_arguments()
    validate_inputs(args)
    
    # Setup output directory
    output_dir = setup_output_directory(args.output_dir)
    print(f"Results will be saved to: {output_dir}")
    
    # Load data and model
    data = load_data_and_model(args)
    
    # Initialize tracking
    all_results = {}
    completed_analyses = []
    total_analyses = sum([
        not args.skip_attribution,
        not args.skip_functional, 
        not args.skip_motif
    ])
    current_analysis = 0
    
    print(f"\n=== Starting {total_analyses} Analysis Tasks ===")
    
    # Attribution Consistency Analysis
    if not args.skip_attribution:
        current_analysis += 1
        print(f"\n[{current_analysis}/{total_analyses}] --- Running Attribution Consistency Analysis ---")
        try:
            attribution_results = run_attribution_consistency_analysis(
                data['deepstarr'], 
                data['sample_seqs'],
                data['X_test'],
                output_dir
            )
            all_results['attribution_consistency'] = attribution_results
            completed_analyses.append('attribution_consistency')
            
            # Save progress immediately
            print_analysis_summary('attribution_consistency', attribution_results)
            save_progress_file(output_dir, completed_analyses, all_results)
            save_combined_results(output_dir, all_results)
            
        except Exception as e:
            import traceback
            print(f"✗ Attribution analysis failed: {e}")
            print("Full error traceback:")
            traceback.print_exc()
            print("Continuing with remaining analyses...")
    
    # Functional Similarity Analysis
    if not args.skip_functional:
        current_analysis += 1
        print(f"\n[{current_analysis}/{total_analyses}] --- Running Functional Similarity Analysis ---")
        try:
            functional_results = run_functional_similarity_analysis(
                data['deepstarr'],
                data['x_test_tensor'],
                data['x_synthetic_tensor'], 
                data['x_train_tensor'],
                output_dir
            )
            all_results['functional_similarity'] = functional_results
            completed_analyses.append('functional_similarity')
            
            # Save progress immediately
            print_analysis_summary('functional_similarity', functional_results)
            save_progress_file(output_dir, completed_analyses, all_results)
            save_combined_results(output_dir, all_results)
            
        except Exception as e:
            import traceback
            print(f"✗ Functional similarity analysis failed: {e}")
            print("Full error traceback:")
            traceback.print_exc()
            print("Continuing with remaining analyses...")
    
    # Motif Analysis
    if not args.skip_motif:
        current_analysis += 1
        print(f"\n[{current_analysis}/{total_analyses}] --- Running Motif Analysis ---")
        try:
            motif_results = run_motif_analysis(
                data['x_test_tensor'],
                data['x_synthetic_tensor'],
                output_dir
            )
            all_results['motif_analysis'] = motif_results
            completed_analyses.append('motif_analysis')
            
            # Save progress immediately
            print_analysis_summary('motif_analysis', motif_results)
            save_progress_file(output_dir, completed_analyses, all_results)
            save_combined_results(output_dir, all_results)
            
        except Exception as e:
            import traceback
            print(f"✗ Motif analysis failed: {e}")
            print("Full error traceback:")
            traceback.print_exc()
            print("Analysis completed with errors.")
    
    # Final summary
    print(f"\n=== Analysis Pipeline Complete ===")
    print(f"Completed: {len(completed_analyses)}/{total_analyses} analyses")
    print(f"All results saved to: {output_dir}")
    
    if completed_analyses:
        print(f"\n--- Final Results Summary ---")
        for analysis_name in completed_analyses:
            if analysis_name in all_results:
                results = all_results[analysis_name]
                print(f"\n{analysis_name.replace('_', ' ').title()}:")
                for key, value in results.items():
                    if isinstance(value, (int, float)):
                        print(f"  {key}: {value:.6f}")
                    elif hasattr(value, 'shape'):
                        if value.shape == ():  # numpy scalar
                            print(f"  {key}: {value.item():.6f}")
                        else:  # numpy arrays
                            print(f"  {key}: Array with shape {value.shape}")
                    else:
                        print(f"  {key}: {type(value).__name__}")
    
    if len(completed_analyses) < total_analyses:
        failed_count = total_analyses - len(completed_analyses)
        print(f"\n⚠️  {failed_count} analysis(es) failed - check output for details")
    else:
        print(f"\n✅ All analyses completed successfully!")


if __name__ == "__main__":
    main()