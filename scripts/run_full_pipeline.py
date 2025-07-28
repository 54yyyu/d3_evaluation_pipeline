#!/usr/bin/env python3
"""
Run the full evaluation pipeline on synthetic sequences.

This script runs all evaluation metrics (functional similarity, sequence similarity,
and compositional similarity) on synthetic sequences and generates a comprehensive
evaluation report.
"""

import argparse
import sys
from pathlib import Path
import logging
import pickle
import json
from datetime import datetime
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.data_utils import DataLoader
from models.model_utils import load_oracle_model
from evaluators.functional_similarity import FunctionalSimilarityEvaluator
from evaluators.sequence_similarity import SequenceSimilarityEvaluator
from evaluators.compositional_similarity import CompositionalSimilarityEvaluator
from utils.config_utils import ConfigManager, get_config_for_dataset
from utils.common_utils import (
    setup_logging, ensure_directory_exists, set_random_seed, 
    format_results_for_display, create_run_metadata, profile_function_time
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the full evaluation pipeline on synthetic sequences"
    )
    
    # Data arguments
    parser.add_argument(
        "--samples-path",
        type=str,
        help="Path to synthetic sequences (NPZ file)"
    )
    parser.add_argument(
        "--dataset-path", 
        type=str,
        help="Path to dataset (H5 file)"
    )
    parser.add_argument(
        "--oracle-path",
        type=str,
        help="Path to oracle model checkpoint"
    )
    parser.add_argument(
        "--motif-database-path",
        type=str,
        help="Path to motif database (optional for compositional similarity)"
    )
    
    # Configuration arguments
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="deepstarr",
        help="Dataset name for loading default configuration"
    )
    
    # Evaluation selection
    parser.add_argument(
        "--skip-functional",
        action="store_true",
        help="Skip functional similarity evaluation"
    )
    parser.add_argument(
        "--skip-sequence",
        action="store_true",
        help="Skip sequence similarity evaluation"
    )
    parser.add_argument(
        "--skip-compositional",
        action="store_true",
        help="Skip compositional similarity evaluation"
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--output-format",
        choices=["pickle", "json", "both"],
        default="pickle",
        help="Format for saving results"
    )
    parser.add_argument(
        "--save-individual",
        action="store_true",
        help="Save individual evaluation results in addition to combined results"
    )
    
    # Computation arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for computations"
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for computation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    # Logging arguments
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="File to save logs"
    )
    
    return parser.parse_args()


def validate_required_args(args):
    """Validate that required arguments are provided when config file is not used."""
    if not args.config:
        required_args = ['samples_path', 'dataset_path', 'oracle_path']
        missing_args = []
        
        for arg in required_args:
            if not getattr(args, arg.replace('-', '_')):
                missing_args.append(f"--{arg.replace('_', '-')}")
        
        if missing_args:
            raise ValueError(
                f"The following arguments are required when no config file is provided: {', '.join(missing_args)}"
            )


def load_configuration(args):
    """Load and configure the evaluation configuration."""
    # Validate required arguments if no config file provided
    validate_required_args(args)
    
    # Load base configuration
    if args.config:
        config_manager = ConfigManager(args.config)
    else:
        config = get_config_for_dataset(args.dataset)
        config_manager = ConfigManager()
        config_manager.config = config
    
    # Update configuration with command line arguments (only if provided)
    update_kwargs = {}
    if args.samples_path:
        update_kwargs['samples_path'] = args.samples_path
    if args.dataset_path:
        update_kwargs['dataset_path'] = args.dataset_path
    if args.oracle_path:
        update_kwargs['oracle_path'] = args.oracle_path
    if args.motif_database_path:
        update_kwargs['motif_database_path'] = args.motif_database_path
    if args.output_dir != "results":  # Only update if changed from default
        update_kwargs['results_dir'] = args.output_dir
    if args.batch_size:
        update_kwargs['batch_size'] = args.batch_size
    if args.device != "auto":  # Only update if changed from default
        update_kwargs['device'] = args.device
    if args.seed != 42:  # Only update if changed from default
        update_kwargs['random_seed'] = args.seed
    
    if update_kwargs:
        config_manager.update(**update_kwargs)
    
    # Update evaluation flags
    config_manager.set('evaluation.run_functional_similarity', not args.skip_functional)
    config_manager.set('evaluation.run_sequence_similarity', not args.skip_sequence)
    config_manager.set('evaluation.run_compositional_similarity', not args.skip_compositional)
    
    # Resolve paths
    config_manager.resolve_paths()
    
    return config_manager


@profile_function_time
def load_data_and_model(config_manager):
    """Load data and model."""
    logger = logging.getLogger("evaluation_pipeline")
    
    # Load data
    logger.info("Loading data...")
    data_loader = DataLoader(config_manager.config)
    
    x_test, x_synthetic, x_train = data_loader.extract_evaluation_data(
        config_manager.get('data.samples_path'),
        config_manager.get('data.dataset_path')
    )
    
    logger.info(f"Loaded {len(x_synthetic)} synthetic sequences")
    logger.info(f"Loaded {len(x_test)} test sequences")
    logger.info(f"Loaded {len(x_train)} training sequences")
    
    # Print data summary
    data_summary = data_loader.get_data_summary(x_test, x_synthetic, x_train)
    logger.info(f"Data summary: {data_summary}")
    
    # Load oracle model
    oracle_path = config_manager.get('data.oracle_path')
    logger.info("Loading oracle model...")
    oracle_model = load_oracle_model(
        oracle_path,
        config_manager.get('model.type', 'deepstarr')
    )
    logger.info("Oracle model loaded successfully")
    
    return x_test, x_synthetic, x_train, oracle_model


@profile_function_time
def run_functional_similarity(x_test, x_synthetic, oracle_model, config_manager):
    """Run functional similarity evaluation."""
    logger = logging.getLogger("evaluation_pipeline")
    logger.info("Running functional similarity evaluation...")
    
    evaluator = FunctionalSimilarityEvaluator(config_manager.config)
    results = evaluator.evaluate(x_synthetic, x_test, oracle_model)
    
    logger.info("Functional similarity evaluation completed")
    return evaluator, results


@profile_function_time
def run_sequence_similarity(x_test, x_synthetic, x_train, config_manager):
    """Run sequence similarity evaluation."""
    logger = logging.getLogger("evaluation_pipeline")
    logger.info("Running sequence similarity evaluation...")
    
    evaluator = SequenceSimilarityEvaluator(config_manager.config)
    results = evaluator.evaluate(x_synthetic, x_test, x_train=x_train)
    
    logger.info("Sequence similarity evaluation completed")
    return evaluator, results


@profile_function_time
def run_compositional_similarity(x_test, x_synthetic, oracle_model, config_manager):
    """Run compositional similarity evaluation."""
    logger = logging.getLogger("evaluation_pipeline")
    logger.info("Running compositional similarity evaluation...")
    
    evaluator = CompositionalSimilarityEvaluator(config_manager.config)
    motif_db_path = config_manager.get('data.motif_database_path')
    
    results = evaluator.evaluate(
        x_synthetic, x_test, oracle_model, 
        motif_database_path=motif_db_path
    )
    
    logger.info("Compositional similarity evaluation completed")
    return evaluator, results


def combine_results(all_results: Dict[str, Dict[str, Any]], 
                   config_manager: ConfigManager) -> Dict[str, Any]:
    """Combine results from all evaluations."""
    combined = {
        "evaluation_summary": {
            "total_evaluations": len(all_results),
            "evaluation_types": list(all_results.keys())
        },
        "results": all_results,
        "metadata": create_run_metadata(config_manager.config)
    }
    
    return combined


def save_intermediate_results(eval_type: str, results: Dict[str, Any], 
                            config_manager: ConfigManager, 
                            output_format: str = "pickle"):
    """Save intermediate results as each evaluation completes."""
    logger = logging.getLogger("evaluation_pipeline")
    
    # Ensure output directory exists
    output_dir = config_manager.get('output.results_dir')
    ensure_directory_exists(output_dir)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create intermediate results structure
    intermediate_results = {
        "evaluation_type": eval_type,
        "results": results,
        "timestamp": timestamp,
        "metadata": create_run_metadata(config_manager.config)
    }
    
    if output_format in ["pickle", "both"]:
        intermediate_file = Path(output_dir) / f"intermediate_{eval_type}_{timestamp}.pkl"
        with open(intermediate_file, 'wb') as f:
            pickle.dump(intermediate_results, f)
        logger.info(f"✓ {eval_type} results saved to {intermediate_file}")
    
    if output_format in ["json", "both"]:
        intermediate_file = Path(output_dir) / f"intermediate_{eval_type}_{timestamp}.json"
        json_data = convert_for_json(intermediate_results)
        with open(intermediate_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        logger.info(f"✓ {eval_type} results saved to {intermediate_file}")


def save_results(all_results: Dict[str, Any], 
                evaluators: Dict[str, Any],
                config_manager: ConfigManager,
                save_individual: bool = False,
                output_format: str = "pickle"):
    """Save final combined evaluation results."""
    logger = logging.getLogger("evaluation_pipeline")
    
    # Ensure output directory exists
    output_dir = config_manager.get('output.results_dir')
    ensure_directory_exists(output_dir)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Save combined results
    combined_results = combine_results(all_results, config_manager)
    
    if output_format in ["pickle", "both"]:
        combined_file = Path(output_dir) / f"full_evaluation_{timestamp}.pkl"
        with open(combined_file, 'wb') as f:
            pickle.dump(combined_results, f)
        logger.info(f"Final combined results saved to {combined_file}")
    
    if output_format in ["json", "both"]:
        combined_file = Path(output_dir) / f"full_evaluation_{timestamp}.json"
        # Convert numpy arrays for JSON serialization
        json_data = convert_for_json(combined_results)
        with open(combined_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        logger.info(f"Final combined results saved to {combined_file}")
    
    # Save individual results if requested
    if save_individual:
        for eval_type, evaluator in evaluators.items():
            individual_format = "pickle" if output_format == "both" else output_format
            evaluator.save_results(output_dir, format=individual_format)
            logger.info(f"Individual {eval_type} results saved")


def convert_for_json(obj):
    """Convert numpy arrays and other non-JSON types for serialization."""
    import numpy as np
    
    if isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_for_json(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    else:
        return obj


def print_summary(all_results: Dict[str, Dict[str, Any]]):
    """Print summary of all evaluation results."""
    print("\n" + "="*80)
    print("FINAL EVALUATION PIPELINE SUMMARY")
    print("="*80)
    
    for eval_type, results in all_results.items():
        print(f"\n{eval_type.upper()} SIMILARITY:")
        print("-" * 40)
        print(format_results_for_display(results))
    
    print("\n" + "="*80)


def main():
    """Main function."""
    args = parse_arguments()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_level, args.log_file)
    
    try:
        # Set random seed
        set_random_seed(args.seed)
        logger.info(f"Random seed set to {args.seed}")
        
        # Load configuration
        logger.info("Loading configuration...")
        config_manager = load_configuration(args)
        
        # Validate required files exist
        from utils.common_utils import validate_file_exists
        
        # Get paths from config or args
        samples_path = args.samples_path or config_manager.get('data.samples_path')
        dataset_path = args.dataset_path or config_manager.get('data.dataset_path')
        oracle_path = args.oracle_path or config_manager.get('data.oracle_path')
        motif_database_path = args.motif_database_path or config_manager.get('data.motif_database_path')
        
        validate_file_exists(samples_path, "Samples file")
        validate_file_exists(dataset_path, "Dataset file")  
        validate_file_exists(oracle_path, "Oracle model")
        
        if motif_database_path:
            validate_file_exists(motif_database_path, "Motif database")
        
        # Load data and model
        x_test, x_synthetic, x_train, oracle_model = load_data_and_model(config_manager)
        
        # Run evaluations
        all_results = {}
        evaluators = {}
        
        # Functional similarity
        if config_manager.get('evaluation.run_functional_similarity', True):
            evaluator, results = run_functional_similarity(
                x_test, x_synthetic, oracle_model, config_manager
            )
            all_results['functional_similarity'] = results
            evaluators['functional_similarity'] = evaluator
            
            # Save intermediate results
            save_intermediate_results('functional_similarity', results, 
                                    config_manager, args.output_format)
            
            # Print intermediate summary
            print(f"\n{'='*60}")
            print("FUNCTIONAL SIMILARITY RESULTS (INTERMEDIATE)")
            print('='*60)
            print(format_results_for_display(results))
            print('='*60)
        
        # Sequence similarity
        if config_manager.get('evaluation.run_sequence_similarity', True):
            evaluator, results = run_sequence_similarity(
                x_test, x_synthetic, x_train, config_manager
            )
            all_results['sequence_similarity'] = results
            evaluators['sequence_similarity'] = evaluator
            
            # Save intermediate results
            save_intermediate_results('sequence_similarity', results, 
                                    config_manager, args.output_format)
            
            # Print intermediate summary
            print(f"\n{'='*60}")
            print("SEQUENCE SIMILARITY RESULTS (INTERMEDIATE)")
            print('='*60)
            print(format_results_for_display(results))
            print('='*60)
        
        # Compositional similarity
        if config_manager.get('evaluation.run_compositional_similarity', True):
            evaluator, results = run_compositional_similarity(
                x_test, x_synthetic, oracle_model, config_manager
            )
            all_results['compositional_similarity'] = results
            evaluators['compositional_similarity'] = evaluator
            
            # Save intermediate results
            save_intermediate_results('compositional_similarity', results, 
                                    config_manager, args.output_format)
            
            # Print intermediate summary
            print(f"\n{'='*60}")
            print("COMPOSITIONAL SIMILARITY RESULTS (INTERMEDIATE)")
            print('='*60)
            print(format_results_for_display(results))
            print('='*60)
        
        # Print summary
        print_summary(all_results)
        
        # Save results
        save_results(
            all_results, evaluators, config_manager,
            save_individual=args.save_individual,
            output_format=args.output_format
        )
        
        logger.info("Full evaluation pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation pipeline failed: {e}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()