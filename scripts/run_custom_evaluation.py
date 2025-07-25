#!/usr/bin/env python3
"""
Run custom evaluation combinations on synthetic sequences.

This script allows running custom combinations of evaluation metrics with
fine-grained control over parameters and subsets of data.
"""

import argparse
import sys
from pathlib import Path
import logging
import numpy as np
from typing import List, Dict, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.data_utils import DataLoader
from models.model_utils import load_oracle_model
from evaluators.functional_similarity import FunctionalSimilarityEvaluator
from evaluators.sequence_similarity import SequenceSimilarityEvaluator
from evaluators.compositional_similarity import CompositionalSimilarityEvaluator
from utils.config_utils import ConfigManager, get_config_for_dataset
from utils.common_utils import setup_logging, ensure_directory_exists, set_random_seed


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run custom evaluation combinations on synthetic sequences"
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
        help="Path to motif database"
    )
    
    # Custom evaluation selection
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        choices=[
            # Functional similarity metrics
            "conditional_fidelity", "frechet_distance", "distribution_shift",
            # Sequence similarity metrics  
            "percent_identity", "kmer_analysis", "discriminatability", "diversity",
            # Compositional similarity metrics
            "motif_enrichment", "motif_cooccurrence", "attribution_consistency"
        ],
        help="Specific metrics to run"
    )
    
    # Data subset options
    parser.add_argument(
        "--subset-synthetic",
        type=int,
        help="Number of synthetic sequences to use (random subset)"
    )
    parser.add_argument(
        "--subset-test",
        type=int,
        help="Number of test sequences to use (random subset)"
    )
    parser.add_argument(
        "--subset-train",
        type=int,
        help="Number of training sequences to use (random subset)"
    )
    
    # Custom parameter overrides
    parser.add_argument(
        "--kmer-lengths",
        type=int,
        nargs="+",
        default=[3, 4, 5],
        help="K-mer lengths for analysis"
    )
    parser.add_argument(
        "--attribution-samples",
        type=int,
        default=500,
        help="Number of samples for attribution analysis"
    )
    parser.add_argument(
        "--identity-batch-size",
        type=int,
        default=1000,
        help="Batch size for percent identity calculations"
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
    
    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="custom_eval",
        help="Prefix for output files"
    )
    parser.add_argument(
        "--output-format",
        choices=["pickle", "json"],
        default="pickle",
        help="Format for saving results"
    )
    
    # Computation arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Default batch size for computations"
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
        required_args = ['samples_path', 'dataset_path']
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
    
    # Update custom parameters
    config_manager.set('evaluation.kmer_lengths', args.kmer_lengths)
    config_manager.set('evaluation.attribution_max_samples', args.attribution_samples)
    config_manager.set('evaluation.identity_batch_size', args.identity_batch_size)
    
    # Resolve paths
    config_manager.resolve_paths()
    
    return config_manager


def subset_data(x_synthetic, x_test, x_train, args):
    """Create subsets of data based on arguments."""
    logger = logging.getLogger("evaluation_pipeline")
    
    # Set random seed for reproducible subsets
    np.random.seed(args.seed)
    
    # Subset synthetic sequences
    if args.subset_synthetic and len(x_synthetic) > args.subset_synthetic:
        indices = np.random.choice(len(x_synthetic), args.subset_synthetic, replace=False)
        x_synthetic = x_synthetic[indices]
        logger.info(f"Using subset of {len(x_synthetic)} synthetic sequences")
    
    # Subset test sequences
    if args.subset_test and len(x_test) > args.subset_test:
        indices = np.random.choice(len(x_test), args.subset_test, replace=False)
        x_test = x_test[indices]
        logger.info(f"Using subset of {len(x_test)} test sequences")
    
    # Subset training sequences
    if args.subset_train and x_train is not None and len(x_train) > args.subset_train:
        indices = np.random.choice(len(x_train), args.subset_train, replace=False)
        x_train = x_train[indices]
        logger.info(f"Using subset of {len(x_train)} training sequences")
    
    return x_synthetic, x_test, x_train


def get_metric_groups(metrics: List[str]) -> Dict[str, List[str]]:
    """Group metrics by evaluation type."""
    functional_metrics = ["conditional_fidelity", "frechet_distance", "distribution_shift"]
    sequence_metrics = ["percent_identity", "kmer_analysis", "discriminatability", "diversity"]
    compositional_metrics = ["motif_enrichment", "motif_cooccurrence", "attribution_consistency"]
    
    groups = {}
    
    if any(m in functional_metrics for m in metrics):
        groups["functional"] = [m for m in metrics if m in functional_metrics]
    
    if any(m in sequence_metrics for m in metrics):
        groups["sequence"] = [m for m in metrics if m in sequence_metrics]
    
    if any(m in compositional_metrics for m in metrics):
        groups["compositional"] = [m for m in metrics if m in compositional_metrics]
    
    return groups


def run_functional_metrics(metrics: List[str], x_test, x_synthetic, oracle_model, config_manager):
    """Run specific functional similarity metrics."""
    logger = logging.getLogger("evaluation_pipeline")
    logger.info(f"Running functional metrics: {metrics}")
    
    evaluator = FunctionalSimilarityEvaluator(config_manager.config)
    
    # Run full evaluation to get all metrics
    all_results = evaluator.evaluate(x_synthetic, x_test, oracle_model)
    
    # Filter to requested metrics
    metric_mapping = {
        "conditional_fidelity": ["conditional_generation_fidelity_mse"],
        "frechet_distance": ["frechet_distance"],
        "distribution_shift": ["predictive_distribution_shift_ks_statistic", "predictive_distribution_shift_p_value"]
    }
    
    filtered_results = {}
    for metric in metrics:
        for key in metric_mapping.get(metric, []):
            if key in all_results:
                filtered_results[key] = all_results[key]
    
    return filtered_results


def run_sequence_metrics(metrics: List[str], x_test, x_synthetic, x_train, config_manager):
    """Run specific sequence similarity metrics."""
    logger = logging.getLogger("evaluation_pipeline")
    logger.info(f"Running sequence metrics: {metrics}")
    
    evaluator = SequenceSimilarityEvaluator(config_manager.config)
    
    # Run full evaluation
    all_results = evaluator.evaluate(x_synthetic, x_test, x_train=x_train)
    
    # Filter to requested metrics
    metric_mapping = {
        "percent_identity": [k for k in all_results.keys() if "percent_identity" in k],
        "kmer_analysis": [k for k in all_results.keys() if "kmer_" in k],
        "discriminatability": [k for k in all_results.keys() if "discriminatability" in k],
        "diversity": [k for k in all_results.keys() if "diversity" in k]
    }
    
    filtered_results = {}
    for metric in metrics:
        for key in metric_mapping.get(metric, []):
            if key in all_results:
                filtered_results[key] = all_results[key]
    
    return filtered_results


def run_compositional_metrics(metrics: List[str], x_test, x_synthetic, oracle_model, config_manager):
    """Run specific compositional similarity metrics."""
    logger = logging.getLogger("evaluation_pipeline")
    logger.info(f"Running compositional metrics: {metrics}")
    
    evaluator = CompositionalSimilarityEvaluator(config_manager.config)
    motif_db_path = config_manager.get('data.motif_database_path')
    
    # Check what can be run
    can_run_motif = motif_db_path is not None
    can_run_attribution = oracle_model is not None
    
    # Run evaluation with available inputs
    all_results = evaluator.evaluate(
        x_synthetic, x_test, oracle_model if can_run_attribution else None,
        motif_database_path=motif_db_path if can_run_motif else None
    )
    
    # Filter to requested metrics
    metric_mapping = {
        "motif_enrichment": [k for k in all_results.keys() if "motif_enrichment" in k],
        "motif_cooccurrence": [k for k in all_results.keys() if "motif_cooccurrence" in k or "cooccurrence" in k],
        "attribution_consistency": [k for k in all_results.keys() if "attribution" in k]
    }
    
    filtered_results = {}
    for metric in metrics:
        if metric == "motif_enrichment" and not can_run_motif:
            logger.warning("Motif enrichment requested but no motif database provided")
            continue
        if metric == "motif_cooccurrence" and not can_run_motif:
            logger.warning("Motif co-occurrence requested but no motif database provided")
            continue
        if metric == "attribution_consistency" and not can_run_attribution:
            logger.warning("Attribution consistency requested but no oracle model provided")
            continue
            
        for key in metric_mapping.get(metric, []):
            if key in all_results:
                filtered_results[key] = all_results[key]
    
    return filtered_results


def save_custom_results(results: Dict[str, Any], config_manager: ConfigManager, args):
    """Save custom evaluation results."""
    logger = logging.getLogger("evaluation_pipeline")
    
    # Ensure output directory exists
    output_dir = config_manager.get('output.results_dir')
    ensure_directory_exists(output_dir)
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create output data
    output_data = {
        "custom_evaluation": {
            "requested_metrics": args.metrics,
            "parameters": {
                "subset_synthetic": args.subset_synthetic,
                "subset_test": args.subset_test,
                "subset_train": args.subset_train,
                "kmer_lengths": args.kmer_lengths,
                "attribution_samples": args.attribution_samples
            }
        },
        "results": results,
        "metadata": {
            "timestamp": timestamp,
            "config": config_manager.config
        }
    }
    
    # Save results
    if args.output_format == "pickle":
        import pickle
        output_file = Path(output_dir) / f"{args.output_prefix}_{timestamp}.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(output_data, f)
    else:
        import json
        output_file = Path(output_dir) / f"{args.output_prefix}_{timestamp}.json"
        # Convert for JSON serialization
        json_data = convert_for_json(output_data)
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=2)
    
    logger.info(f"Custom evaluation results saved to {output_file}")


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
        validate_file_exists(args.samples_path, "Samples file")
        validate_file_exists(args.dataset_path, "Dataset file")
        
        if args.oracle_path:
            validate_file_exists(args.oracle_path, "Oracle model")
        if args.motif_database_path:
            validate_file_exists(args.motif_database_path, "Motif database")
        
        # Load data
        logger.info("Loading data...")
        data_loader = DataLoader(config_manager.config)
        
        x_test, x_synthetic, x_train = data_loader.extract_evaluation_data(
            config_manager.get('data.samples_path'),
            config_manager.get('data.dataset_path')
        )
        
        # Create subsets if requested
        x_synthetic, x_test, x_train = subset_data(x_synthetic, x_test, x_train, args)
        
        # Load oracle model if needed
        oracle_model = None
        metric_groups = get_metric_groups(args.metrics)
        
        if "functional" in metric_groups or "compositional" in metric_groups:
            if args.oracle_path:
                logger.info("Loading oracle model...")
                oracle_model = load_oracle_model(
                    args.oracle_path,
                    config_manager.get('model.type', 'deepstarr')
                )
                logger.info("Oracle model loaded successfully")
            else:
                logger.warning("Oracle-dependent metrics requested but no oracle path provided")
        
        # Run requested evaluations
        all_results = {}
        
        if "functional" in metric_groups:
            functional_results = run_functional_metrics(
                metric_groups["functional"], x_test, x_synthetic, oracle_model, config_manager
            )
            all_results.update(functional_results)
        
        if "sequence" in metric_groups:
            sequence_results = run_sequence_metrics(
                metric_groups["sequence"], x_test, x_synthetic, x_train, config_manager
            )
            all_results.update(sequence_results)
        
        if "compositional" in metric_groups:
            compositional_results = run_compositional_metrics(
                metric_groups["compositional"], x_test, x_synthetic, oracle_model, config_manager
            )
            all_results.update(compositional_results)
        
        # Display results
        from utils.common_utils import format_results_for_display
        print("\n" + format_results_for_display(all_results))
        
        # Save results
        save_custom_results(all_results, config_manager, args)
        
        logger.info("Custom evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Custom evaluation failed: {e}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()