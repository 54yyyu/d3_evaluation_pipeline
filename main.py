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
# Legacy imports removed - using only modular imports now

# New modular imports
from core.functional.cond_gen_fidelity import run_conditional_generation_fidelity_analysis
from core.functional.frechet_distance import run_frechet_distance_analysis
from core.functional.predictive_dist_shift import run_predictive_distribution_shift_analysis
from core.sequence.percent_identity import run_percent_identity_analysis
from core.sequence.kmer_spectrum_shift import run_kmer_spectrum_shift_analysis
from core.sequence.discriminability import run_discriminability_analysis as run_discriminability_analysis_modular
from core.compositional.motif_enrichment import run_motif_enrichment_analysis
from core.compositional.motif_cooccurrence import run_motif_cooccurrence_analysis
from core.compositional.attribution_consistency import run_attribution_consistency_analysis as run_attribution_consistency_analysis_modular


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run D3 sequence analysis')
    
    parser.add_argument('--samples', type=str, 
                       default=os.getenv('SAMPLES_FILE', 'samples.npz'),
                       help='Path to samples NPZ file')
    
    parser.add_argument('--samples-batch', type=str,
                       help='Path to directory containing multiple NPZ files for batch processing')
    
    parser.add_argument('--data', type=str,
                       default=os.getenv('DATA_FILE', 'DeepSTARR_data.h5'), 
                       help='Path to DeepSTARR data H5 file')
    
    parser.add_argument('--model', type=str,
                       default=os.getenv('MODEL_FILE', 'oracle_DeepSTARR_DeepSTARR_data.ckpt'),
                       help='Path to model checkpoint file')
    
    parser.add_argument('--output-dir', type=str,
                       default=os.getenv('OUTPUT_DIR', 'results'),
                       help='Output directory for results')
    
    parser.add_argument('--functional', action='store_true',
                       help="""Run functional similarity tests: 
                        cond_gen_fidelity, frechet_distance, predictive_dist_shift""")
    
    parser.add_argument('--sequence', action='store_true', 
                       help="""Run sequence similarity tests: 
                        percent_identity, kmer_spectrum_shift, discriminability""")
    
    parser.add_argument('--compositional', action='store_true',
                       help="""Run compositional similarity tests: 
                        motif_enrichment, motif_cooccurrence, attribution_consistency""")
    
    parser.add_argument('--test', type=str, help="""Run specific test(s). 
                        Single test or comma-separated list. 
                        Available: cond_gen_fidelity, frechet_distance, predictive_dist_shift, 
                        percent_identity, kmer_spectrum_shift, discriminability, 
                        motif_enrichment, motif_cooccurrence, attribution_consistency""")
    
    parser.add_argument('--motif-db', type=str, 
                       default='JASPAR2024_CORE_non-redundant_pfms_meme.txt',
                       help='Path to motif database file for motif analysis (default: JASPAR2024_CORE_non-redundant_pfms_meme.txt)')
    
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


def run_batch_analysis(args):
    """Run analysis in batch mode on multiple NPZ files."""
    from utils.batch_helpers import discover_batch_samples, load_batch_sample
    from utils.helpers import extract_data, numpy_to_tensor, load_deepstarr
    
    print("=== D3 Sequence Analysis Pipeline - Batch Mode ===")
    
    # Discover batch samples (this may exit if CSV template is created)
    batch_samples = discover_batch_samples(args.samples_batch)
    
    # Validate that required files exist
    required_files = [
        (args.data, 'Data file'), 
        (args.model, 'Model file')
    ]
    
    missing_files = []
    for file_path, description in required_files:
        if not os.path.exists(file_path):
            missing_files.append(f"{description}: {file_path}")
    
    if missing_files:
        print("Error: Missing required files:")
        for missing in missing_files:
            print(f"  - {missing}")
        sys.exit(1)
    
    # Setup output directory
    output_dir = setup_output_directory(args.output_dir)
    print(f"Results will be saved to: {output_dir}")
    
    # Load model and test data once (they're shared across all samples)
    print("Loading model and test data...")
    deepstarr = load_deepstarr(args.model)
    
    # Load test and training data from the data file
    with h5py.File(args.data, 'r') as f:
        x_test = f['X_test'][()]
        x_train = f['X_train'][()]
    
    x_test_tensor = numpy_to_tensor(x_test)
    x_train_tensor = numpy_to_tensor(x_train)
    
    # For attribution analysis, convert test data to proper format
    X_test = torch.tensor(np.array(x_test).transpose(0,2,1), dtype=torch.float32)
    
    print(f"Loaded {len(x_test)} test sequences")
    print(f"Loaded {len(x_train)} training sequences")
    
    # Determine which tests to run
    if args.test:
        test_list = [t.strip() for t in args.test.split(',')]
        valid_tests = [
            'cond_gen_fidelity', 'frechet_distance', 'predictive_dist_shift',
            'percent_identity', 'kmer_spectrum_shift', 'discriminability',
            'motif_enrichment', 'motif_cooccurrence', 'attribution_consistency'
        ]
        
        invalid_tests = [t for t in test_list if t not in valid_tests]
        if invalid_tests:
            print(f"Error: Invalid test(s): {invalid_tests}")
            print(f"Valid tests: {valid_tests}")
            sys.exit(1)
    else:
        # Run tests based on similarity type flags
        test_list = []
        if not any([args.functional, args.sequence, args.compositional]):
            # If no specific flags provided, run all tests
            test_list = [
                'cond_gen_fidelity', 'frechet_distance', 'predictive_dist_shift',
                'percent_identity', 'kmer_spectrum_shift', 'discriminability',
                'motif_enrichment', 'motif_cooccurrence', 'attribution_consistency'
            ]
        else:
            if args.functional:
                test_list.extend(['cond_gen_fidelity', 'frechet_distance', 'predictive_dist_shift'])
            if args.sequence:
                test_list.extend(['percent_identity', 'kmer_spectrum_shift', 'discriminability'])
            if args.compositional:
                test_list.extend(['motif_enrichment', 'motif_cooccurrence', 'attribution_consistency'])
    
    print(f"\nProcessing {len(batch_samples)} samples with {len(test_list)} analyses each...")
    
    # Process each sample
    for i, sample_record in enumerate(batch_samples, 1):
        sample_name = sample_record['sample_name']
        print(f"\n[{i}/{len(batch_samples)}] Processing sample: {sample_name}")
        
        # Load sample data
        sample_result = load_batch_sample(args.samples_batch, sample_record)
        if sample_result is None:
            print(f"Skipping {sample_name} due to loading error")
            continue
            
        sample_name_loaded, npz_data = sample_result
        
        # Extract synthetic sequences from NPZ
        try:
            x_synthetic = np.transpose(npz_data['arr_0'], (0, 2, 1))  # Convert to (N, 4, L)
            x_synthetic_tensor = numpy_to_tensor(x_synthetic)
            
            # For attribution analysis
            sample_seqs = torch.tensor(npz_data['arr_0'], dtype=torch.float32)
            
            print(f"Loaded {len(x_synthetic)} synthetic sequences for {sample_name}")
            
        except Exception as e:
            print(f"Error processing {sample_name}: {e}")
            continue
        
        # Run each analysis for this sample
        for j, test_name in enumerate(test_list, 1):
            print(f"  [{j}/{len(test_list)}] Running {test_name.replace('_', ' ').title()}...")
            try:
                run_single_batch_test(test_name, deepstarr, x_test_tensor, x_synthetic_tensor, 
                                     x_train_tensor, sample_seqs, X_test, output_dir, sample_name, args.motif_db)
            except Exception as e:
                import traceback
                print(f"    ✗ {test_name} failed for {sample_name}: {e}")
                traceback.print_exc()
    
    print(f"\n=== Batch Analysis Complete ===")
    print(f"Results saved to: {output_dir}")
    print(f"Check CSV files for concise metrics and H5 files for detailed results")


def run_single_batch_test(test_name, deepstarr, x_test_tensor, x_synthetic_tensor, 
                         x_train_tensor, sample_seqs, X_test, output_dir, sample_name, motif_db_path):
    """Run a single test for batch mode."""
    if test_name == 'cond_gen_fidelity':
        run_conditional_generation_fidelity_analysis(
            deepstarr, x_test_tensor, x_synthetic_tensor, output_dir, sample_name)
    elif test_name == 'frechet_distance':
        run_frechet_distance_analysis(
            deepstarr, x_test_tensor, x_synthetic_tensor, output_dir, sample_name)
    elif test_name == 'predictive_dist_shift':
        run_predictive_distribution_shift_analysis(
            deepstarr, x_test_tensor, x_synthetic_tensor, output_dir, sample_name)
    elif test_name == 'percent_identity':
        run_percent_identity_analysis(
            x_synthetic_tensor, x_train_tensor, output_dir, sample_name)
    elif test_name == 'kmer_spectrum_shift':
        run_kmer_spectrum_shift_analysis(
            x_test_tensor, x_synthetic_tensor, output_dir=output_dir, sample_name=sample_name)
    elif test_name == 'discriminability':
        # Check if discriminability data exists, if not create it first
        discriminability_file = f'Discriminatability_{sample_name}.h5'
        if not os.path.exists(discriminability_file):
            from core.sequence.discriminability import prep_data_for_classification
            from utils.helpers import write_to_h5
            data_dict = prep_data_for_classification(x_test_tensor, x_synthetic_tensor)
            write_to_h5(discriminability_file, data_dict)
        run_discriminability_analysis_modular(
            output_dir=output_dir, h5_file=discriminability_file, sample_name=sample_name)
    elif test_name == 'motif_enrichment':
        run_motif_enrichment_analysis(
            x_test_tensor, x_synthetic_tensor, output_dir, motif_db_path, sample_name)
    elif test_name == 'motif_cooccurrence':
        run_motif_cooccurrence_analysis(
            x_test_tensor, x_synthetic_tensor, output_dir, motif_db_path, sample_name)
    elif test_name == 'attribution_consistency':
        run_attribution_consistency_analysis_modular(
            deepstarr, sample_seqs, X_test, output_dir, sample_name)


def main():
    """Main analysis pipeline with on-the-fly result saving."""
    print("=== D3 Sequence Analysis Pipeline ===")
    
    # Parse arguments and validate inputs
    args = parse_arguments()
    
    # Check if batch mode
    if args.samples_batch:
        run_batch_analysis(args)
        return
    
    validate_inputs(args)
    
    # Setup output directory
    output_dir = setup_output_directory(args.output_dir)
    print(f"Results will be saved to: {output_dir}")
    
    # Load data and model
    data = load_data_and_model(args)
    
    # Initialize tracking
    all_results = {}
    completed_analyses = []
    
    if args.test:
        # Parse comma-separated test list
        test_list = [t.strip() for t in args.test.split(',')]
        valid_tests = [
            'cond_gen_fidelity', 'frechet_distance', 'predictive_dist_shift',
            'percent_identity', 'kmer_spectrum_shift', 'discriminability',
            'motif_enrichment', 'motif_cooccurrence', 'attribution_consistency'
        ]
        
        # Validate all tests
        invalid_tests = [t for t in test_list if t not in valid_tests]
        if invalid_tests:
            print(f"Error: Invalid test(s): {invalid_tests}")
            print(f"Valid tests: {valid_tests}")
            sys.exit(1)
        
        total_analyses = len(test_list)
        if len(test_list) == 1:
            print(f"\n=== Running Single Test: {test_list[0]} ===")
        else:
            print(f"\n=== Running {len(test_list)} Selected Tests: {', '.join(test_list)} ===")
        
        # Run each selected test
        for i, test_name in enumerate(test_list, 1):
            print(f"\n[{i}/{total_analyses}] --- Running {test_name.replace('_', ' ').title()} ---")
            run_single_modular_test(test_name, data, output_dir, all_results, completed_analyses, args.motif_db)
    else:
        # Run tests based on similarity type flags
        run_similarity_type_tests(args, data, output_dir, all_results, completed_analyses)
        total_analyses = len(completed_analyses)  # Count what was actually run
    
    
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


def run_single_modular_test(test_name, data, output_dir, all_results, completed_analyses, motif_db_path='JASPAR2024_CORE_non-redundant_pfms_meme.txt'):
    """Run a single modular test."""
    try:
        if test_name == 'cond_gen_fidelity':
            results = run_conditional_generation_fidelity_analysis(
                data['deepstarr'], data['x_test_tensor'], data['x_synthetic_tensor'], output_dir)
        elif test_name == 'frechet_distance':
            results = run_frechet_distance_analysis(
                data['deepstarr'], data['x_test_tensor'], data['x_synthetic_tensor'], output_dir)
        elif test_name == 'predictive_dist_shift':
            results = run_predictive_distribution_shift_analysis(
                data['deepstarr'], data['x_test_tensor'], data['x_synthetic_tensor'], output_dir)
        elif test_name == 'percent_identity':
            results = run_percent_identity_analysis(
                data['x_synthetic_tensor'], data['x_train_tensor'], output_dir)
        elif test_name == 'kmer_spectrum_shift':
            results = run_kmer_spectrum_shift_analysis(
                data['x_test_tensor'], data['x_synthetic_tensor'], output_dir=output_dir)
        elif test_name == 'discriminability':
            # Check if discriminability data exists, if not create it first
            discriminability_file = 'Discriminatability.h5'
            if not os.path.exists(discriminability_file):
                print("Discriminability data not found. Creating it from existing data...")
                from core.sequence.discriminability import prep_data_for_classification
                from utils.helpers import write_to_h5
                data_dict = prep_data_for_classification(
                    data['x_test_tensor'], data['x_synthetic_tensor'])
                write_to_h5(discriminability_file, data_dict)
                print(f"Created discriminability data: {discriminability_file}")
            results = run_discriminability_analysis_modular(
                output_dir=output_dir, h5_file=discriminability_file)
        elif test_name == 'motif_enrichment':
            results = run_motif_enrichment_analysis(
                data['x_test_tensor'], data['x_synthetic_tensor'], output_dir, motif_db_path)
        elif test_name == 'motif_cooccurrence':
            results = run_motif_cooccurrence_analysis(
                data['x_test_tensor'], data['x_synthetic_tensor'], output_dir, motif_db_path)
        elif test_name == 'attribution_consistency':
            results = run_attribution_consistency_analysis_modular(
                data['deepstarr'], data['sample_seqs'], data['X_test'], output_dir)
        
        all_results[test_name] = results
        completed_analyses.append(test_name)
        print_analysis_summary(test_name, results)
        save_progress_file(output_dir, completed_analyses, all_results)
        save_combined_results(output_dir, all_results)
        
    except Exception as e:
        import traceback
        print(f"✗ {test_name} analysis failed: {e}")
        print("Full error traceback:")
        traceback.print_exc()


def run_similarity_type_tests(args, data, output_dir, all_results, completed_analyses):
    """Run tests based on similarity type flags."""
    tests_to_run = []
    
    # Determine which tests to run based on flags
    if not any([args.functional, args.sequence, args.compositional]):
        # If no specific flags provided, run all tests
        print("No specific similarity type flags provided. Running all tests.")
        tests_to_run = [
            'cond_gen_fidelity', 'frechet_distance', 'predictive_dist_shift',
            'percent_identity', 'kmer_spectrum_shift', 'discriminability',
            'motif_enrichment', 'motif_cooccurrence', 'attribution_consistency'
        ]
    else:
        # Run only requested similarity types
        if args.functional:
            tests_to_run.extend(['cond_gen_fidelity', 'frechet_distance', 'predictive_dist_shift'])
        
        if args.sequence:
            tests_to_run.extend(['percent_identity', 'kmer_spectrum_shift', 'discriminability'])
        
        if args.compositional:
            tests_to_run.extend(['motif_enrichment', 'motif_cooccurrence', 'attribution_consistency'])
    
    # Show which types are being run
    types_running = []
    if args.functional or not any([args.functional, args.sequence, args.compositional]):
        types_running.append("Functional")
    if args.sequence or not any([args.functional, args.sequence, args.compositional]):
        types_running.append("Sequence") 
    if args.compositional or not any([args.functional, args.sequence, args.compositional]):
        types_running.append("Compositional")
    
    print(f"\n=== Running {len(tests_to_run)} Tests ({', '.join(types_running)} Similarity) ===")
    
    for i, test_name in enumerate(tests_to_run, 1):
        print(f"\n[{i}/{len(tests_to_run)}] --- Running {test_name.replace('_', ' ').title()} ---")
        run_single_modular_test(test_name, data, output_dir, all_results, completed_analyses, args.motif_db)




if __name__ == "__main__":
    main()