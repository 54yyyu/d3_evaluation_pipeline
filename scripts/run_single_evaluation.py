#!/usr/bin/env python3
"""
Run a single evaluation type on synthetic sequences.

This script allows running individual evaluation metrics (functional similarity,
sequence similarity, or compositional similarity) on synthetic sequences.
"""

import argparse
import sys
from pathlib import Path
import logging

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
        description="Run a single evaluation type on synthetic sequences"
    )
    
    # Required arguments
    parser.add_argument(
        "evaluation_type",
        choices=["functional", "sequence", "compositional"],
        help="Type of evaluation to run"
    )
    
    # Data arguments
    parser.add_argument(
        "--samples-path",
        type=str,
        required=True,
        help="Path to synthetic sequences (NPZ file)"
    )
    parser.add_argument(
        "--dataset-path", 
        type=str,
        required=True,
        help="Path to dataset (H5 file)"
    )
    parser.add_argument(
        "--oracle-path",
        type=str,
        help="Path to oracle model checkpoint (required for functional and compositional similarity)"
    )
    parser.add_argument(
        "--motif-database-path",
        type=str,
        help="Path to motif database (required for compositional similarity)"
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
        "--output-format",
        choices=["pickle", "json"],
        default="pickle",
        help="Format for saving results"
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


def load_configuration(args):
    """Load and configure the evaluation configuration."""
    # Load base configuration
    if args.config:
        config_manager = ConfigManager(args.config)
    else:
        config = get_config_for_dataset(args.dataset)
        config_manager = ConfigManager()
        config_manager.config = config
    
    # Update configuration with command line arguments
    config_manager.update(
        samples_path=args.samples_path,
        dataset_path=args.dataset_path,
        oracle_path=args.oracle_path,
        motif_database_path=args.motif_database_path,
        results_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
        random_seed=args.seed
    )
    
    # Resolve paths
    config_manager.resolve_paths()
    
    return config_manager


def load_data_and_model(config_manager, evaluation_type):
    """Load data and model based on evaluation type."""
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
    
    # Load oracle model if needed
    oracle_model = None
    if evaluation_type in ["functional", "compositional"]:
        oracle_path = config_manager.get('data.oracle_path')
        if oracle_path:
            logger.info("Loading oracle model...")
            oracle_model = load_oracle_model(
                oracle_path,
                config_manager.get('model.type', 'deepstarr')
            )
            logger.info("Oracle model loaded successfully")
        else:
            raise ValueError(f"Oracle model path required for {evaluation_type} similarity")
    
    return x_test, x_synthetic, x_train, oracle_model


def run_evaluation(evaluation_type, x_test, x_synthetic, x_train, oracle_model, config_manager):
    """Run the specified evaluation."""
    logger = logging.getLogger("evaluation_pipeline")
    
    logger.info(f"Running {evaluation_type} similarity evaluation...")
    
    # Create evaluator
    if evaluation_type == "functional":
        evaluator = FunctionalSimilarityEvaluator(config_manager.config)
        results = evaluator.evaluate(x_synthetic, x_test, oracle_model)
    
    elif evaluation_type == "sequence":
        evaluator = SequenceSimilarityEvaluator(config_manager.config)
        results = evaluator.evaluate(x_synthetic, x_test, x_train=x_train)
    
    elif evaluation_type == "compositional":
        evaluator = CompositionalSimilarityEvaluator(config_manager.config)
        motif_db_path = config_manager.get('data.motif_database_path')
        results = evaluator.evaluate(
            x_synthetic, x_test, oracle_model, 
            motif_database_path=motif_db_path
        )
    
    else:
        raise ValueError(f"Unknown evaluation type: {evaluation_type}")
    
    logger.info(f"{evaluation_type.capitalize()} similarity evaluation completed")
    
    return evaluator, results


def save_results(evaluator, results, config_manager):
    """Save evaluation results."""
    logger = logging.getLogger("evaluation_pipeline")
    
    # Ensure output directory exists
    output_dir = config_manager.get('output.results_dir')
    ensure_directory_exists(output_dir)
    
    # Save results
    output_format = config_manager.get('output.format', 'pickle')
    evaluator.save_results(output_dir, format=output_format)
    
    logger.info(f"Results saved to {output_dir}")


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
        
        # Load data and model
        x_test, x_synthetic, x_train, oracle_model = load_data_and_model(
            config_manager, args.evaluation_type
        )
        
        # Run evaluation
        evaluator, results = run_evaluation(
            args.evaluation_type, x_test, x_synthetic, x_train, 
            oracle_model, config_manager
        )
        
        # Display results
        from utils.common_utils import format_results_for_display
        print("\n" + format_results_for_display(results))
        
        # Save results
        save_results(evaluator, results, config_manager)
        
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()