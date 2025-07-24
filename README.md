# D3 Evaluation Pipeline

A comprehensive evaluation pipeline for assessing the quality of synthetic DNA sequences. This pipeline provides standardized metrics for evaluating generated sequences across functional, sequence-level, and compositional dimensions.

## Overview

The evaluation pipeline implements three main categories of evaluation metrics:

- **Functional Similarity**: How well synthetic sequences match the functional behavior of test sequences as measured by oracle models
- **Sequence Similarity**: Direct sequence-level comparison metrics including percent identity, k-mer analysis, and discriminatability
- **Compositional Similarity**: Analysis of motif content, co-occurrence patterns, and attribution consistency

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA (optional, for GPU acceleration)

### Install from source

```bash
git clone <repository-url>
cd evaluation_pipeline
pip install -e .
```

### Install with optional dependencies

```bash
# For motif analysis
pip install -e .[motif-analysis]

# For attribution analysis
pip install -e .[attribution]

# For visualization
pip install -e .[visualization]

# Install all optional dependencies
pip install -e .[all]
```

## Quick Start

### Basic Usage

Run the full evaluation pipeline:

```bash
python scripts/run_full_pipeline.py \
    --samples-path samples.npz \
    --dataset-path DeepSTARR_data.h5 \
    --oracle-path oracle_DeepSTARR_DeepSTARR_data.ckpt \
    --motif-database-path JASPAR2024_CORE_non-redundant_pfms_meme.txt \
    --output-dir results/
```

Run a single evaluation type:

```bash
python scripts/run_single_evaluation.py functional \
    --samples-path samples.npz \
    --dataset-path DeepSTARR_data.h5 \
    --oracle-path oracle_DeepSTARR_DeepSTARR_data.ckpt \
    --output-dir results/
```

### Using Configuration Files

Create a configuration file for your dataset:

```yaml
# my_config.yaml
data:
  samples_path: "my_samples.npz"
  dataset_path: "my_dataset.h5"
  oracle_path: "my_oracle.ckpt"
  motif_database_path: "motifs.txt"

evaluation:
  batch_size: 4096
  kmer_lengths: [3, 4, 5]
```

Run with configuration:

```bash
python scripts/run_full_pipeline.py --config my_config.yaml
```

## Input Data Format

### Required Inputs

1. **Synthetic Sequences** (`samples.npz`): Generated sequences in NPZ format
   - Shape: (N, A, L) or (N, L, A) where N=samples, A=alphabet_size (4), L=sequence_length

2. **Dataset** (`dataset.h5`): HDF5 file containing:
   - `X_test`: Test sequences (N, A, L) or (N, L, A)
   - `X_train`: Training sequences (N, A, L) or (N, L, A)
   - `Y_test`, `Y_train`: Labels (optional)

3. **Oracle Model** (`oracle.ckpt`): Trained model checkpoint
   - Currently supports DeepSTARR models
   - Must be compatible with PyTorch Lightning

### Optional Inputs

4. **Motif Database** (`.txt`): MEME format motif database
   - Required for compositional similarity analysis
   - Example: JASPAR database in MEME format

## Evaluation Metrics

### Functional Similarity

Measures how well synthetic sequences match functional behavior:

- **Conditional Generation Fidelity**: MSE between oracle predictions on synthetic vs test sequences
- **Fréchet Distance**: Distance between embedding distributions from oracle model
- **Predictive Distribution Shift**: KS test on base composition distributions

### Sequence Similarity

Direct sequence-level comparisons:

- **Percent Identity**: Cross-sequence identity calculations
- **K-mer Spectrum Analysis**: KL divergence and Jensen-Shannon distance of k-mer distributions
- **Discriminatability**: Prepares data for training classifiers to distinguish real vs synthetic

### Compositional Similarity

Analysis of sequence composition and regulatory elements:

- **Motif Enrichment**: Correlation of motif counts between synthetic and test sequences
- **Motif Co-occurrence**: Frobenius norm of covariance matrix differences
- **Attribution Consistency**: Consistency of gradient-based attribution maps

## Configuration

### Dataset-Specific Configurations

Pre-configured settings for common datasets:

```bash
# Use DeepSTARR configuration
python scripts/run_full_pipeline.py --dataset deepstarr --samples-path samples.npz
```

### Custom Configuration

Create custom YAML configurations:

```yaml
# config.yaml
data:
  samples_path: "samples.npz"
  dataset_path: "dataset.h5"
  oracle_path: "oracle.ckpt"

model:
  type: "deepstarr"
  embedding_layer: "model.batchnorm6"

evaluation:
  run_functional_similarity: true
  run_sequence_similarity: true
  run_compositional_similarity: true
  batch_size: 2000
  kmer_lengths: [3, 4, 5]

output:
  results_dir: "results"
  format: "pickle"
```

## Command Line Interface

### Full Pipeline

```bash
python scripts/run_full_pipeline.py [OPTIONS]

Options:
  --samples-path TEXT         Path to synthetic sequences (required)
  --dataset-path TEXT         Path to dataset (required)  
  --oracle-path TEXT          Path to oracle model (required)
  --motif-database-path TEXT  Path to motif database
  --config TEXT               Configuration file
  --dataset TEXT              Dataset name for default config
  --output-dir TEXT           Output directory
  --skip-functional           Skip functional similarity
  --skip-sequence            Skip sequence similarity  
  --skip-compositional       Skip compositional similarity
  --batch-size INTEGER       Batch size for computations
  --device [auto|cpu|cuda]   Computation device
  --seed INTEGER             Random seed
  --verbose                  Enable verbose logging
```

### Single Evaluation

```bash
python scripts/run_single_evaluation.py [functional|sequence|compositional] [OPTIONS]

Options:
  Similar to full pipeline, specific to the evaluation type
```

## Output Format

### Results Structure

```python
{
    "evaluation_summary": {
        "total_evaluations": 3,
        "evaluation_types": ["functional_similarity", "sequence_similarity", "compositional_similarity"]
    },
    "results": {
        "functional_similarity": {
            "conditional_generation_fidelity_mse": 0.123,
            "frechet_distance": 45.67,
            "predictive_distribution_shift_ks_statistic": 0.089
        },
        "sequence_similarity": {
            "max_percent_identity_vs_test": 0.89,
            "kmer_3_kullback_leibler_divergence": 0.045,
            "sequence_diversity_mean_similarity": 0.67
        },
        "compositional_similarity": {
            "motif_enrichment_pearson_r": 0.78,
            "motif_cooccurrence_frobenius_norm": 12.3,
            "attribution_consistency_score": 0.65
        }
    },
    "metadata": {
        "timestamp": "2024-01-15T10:30:00",
        "config": {...}
    }
}
```

### File Outputs

Results are saved in the specified output directory:

- `full_evaluation_YYYY-MM-DD_HH-MM-SS.pkl`: Complete results (pickle format)
- `full_evaluation_YYYY-MM-DD_HH-MM-SS.json`: Complete results (JSON format)  
- Individual evaluator files (if `--save-individual` specified)

## Extending the Pipeline

### Adding New Evaluators

1. Create a new evaluator class inheriting from `BaseEvaluator`:

```python
from src.evaluators.base_evaluator import BaseEvaluator

class MyEvaluator(BaseEvaluator):
    def __init__(self, config):
        super().__init__(config, "my_evaluation")
    
    def get_required_inputs(self):
        return {
            "x_synthetic": "Generated sequences",
            "x_test": "Test sequences"
        }
    
    def evaluate(self, x_synthetic, x_test, **kwargs):
        # Implement your evaluation logic
        results = {"my_metric": 0.5}
        self.update_results(results)
        return results
```

2. Add to the pipeline in the main scripts

### Adding New Model Types

Extend `model_utils.py` to support additional oracle model types:

```python
def _load_my_model(self, model_path):
    # Implementation for loading your model type
    return model
```

## Examples

### Example 1: DeepSTARR Evaluation

```bash
# Full pipeline with DeepSTARR data
python scripts/run_full_pipeline.py \
    --dataset deepstarr \
    --samples-path generated_sequences.npz \
    --output-dir results/deepstarr_eval/
```

### Example 2: Custom Evaluation

```bash
# Run only sequence similarity with custom batch size
python scripts/run_single_evaluation.py sequence \
    --samples-path samples.npz \
    --dataset-path my_data.h5 \
    --batch-size 8192 \
    --output-dir results/
```

### Example 3: Programmatic Usage

```python
from src.evaluators.functional_similarity import FunctionalSimilarityEvaluator
from src.data.data_utils import DataLoader
from src.models.model_utils import load_oracle_model

# Load data
loader = DataLoader(config)
x_test, x_synthetic, x_train = loader.extract_evaluation_data(
    "samples.npz", "dataset.h5"
)

# Load model
oracle = load_oracle_model("oracle.ckpt", "deepstarr")

# Run evaluation
evaluator = FunctionalSimilarityEvaluator(config)
results = evaluator.evaluate(x_synthetic, x_test, oracle)
print(results)
```

## Performance Considerations

### Memory Usage

- Large datasets may require smaller batch sizes
- Use `--batch-size` to control memory usage
- Attribution analysis is memory-intensive; consider reducing `attribution_max_samples`

### Computational Requirements

- Functional similarity: Requires GPU for oracle model inference
- Sequence similarity: CPU-intensive for large datasets
- Compositional similarity: Varies by motif database size

### Optimization Tips

- Use GPU when available (`--device cuda`)
- Adjust batch sizes based on available memory
- For large datasets, consider running evaluations separately
- Use configuration files to avoid respecifying parameters

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch sizes or use CPU
2. **Motif analysis fails**: Ensure motif database is in correct MEME format
3. **Model loading fails**: Check model checkpoint compatibility
4. **Import errors**: Install optional dependencies for specific analyses

### Debug Mode

Enable verbose logging for debugging:

```bash
python scripts/run_full_pipeline.py --verbose --log-file debug.log [other options]
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this evaluation pipeline in your research, please cite:

```
@software{d3_evaluation_pipeline,
  title={D3 Evaluation Pipeline: Comprehensive Assessment of Synthetic DNA Sequences},
  author={D3 Analysis Team},
  year={2024},
  url={https://github.com/your-org/d3-analysis}
}
```