# D3 Sequence Analysis Pipeline

This reorganized analysis pipeline provides a unified interface to run all sequence evaluation tasks.

## Structure

```
analysis/
├── main.py                    # Main runner script
├── core/                      # Core analysis modules
│   ├── attribution_analysis.py
│   ├── functional_similarity.py
│   └── motif_analysis.py
├── utils/                     # Utility functions
│   ├── helpers.py
│   └── seq_evals_func_motifs.py
└── training/                  # Training scripts (separated)
    ├── EvoAug_run_train.py
    └── finetune_run_train.py
```

## Usage

### Single Sample Mode

```bash
# Run all tests (default)
python main.py --samples samples.npz --data DeepSTARR_data.h5 --model oracle_DeepSTARR_DeepSTARR_data.ckpt

# Run tests by similarity type
python main.py --functional --samples samples.npz --data DeepSTARR_data.h5 --model oracle_DeepSTARR_DeepSTARR_data.ckpt
python main.py --sequence --samples samples.npz --data DeepSTARR_data.h5 --model oracle_DeepSTARR_DeepSTARR_data.ckpt
python main.py --compositional --samples samples.npz --data DeepSTARR_data.h5 --model oracle_DeepSTARR_DeepSTARR_data.ckpt

# Run specific tests
python main.py --test cond_gen_fidelity --samples samples.npz --data DeepSTARR_data.h5 --model oracle_DeepSTARR_DeepSTARR_data.ckpt
python main.py --test "cond_gen_fidelity,frechet_distance" --samples samples.npz --data DeepSTARR_data.h5 --model oracle_DeepSTARR_DeepSTARR_data.ckpt
```

### Batch Mode (NEW!)

Process multiple samples at once with organized CSV and HDF5 outputs:

```bash
# First run: Creates CSV template and exits for customization
python main.py --samples-batch /path/to/batch_folder --data DeepSTARR_data.h5 --model oracle_model.ckpt

# Second run: Processes all samples after optional CSV editing
python main.py --samples-batch /path/to/batch_folder --data DeepSTARR_data.h5 --model oracle_model.ckpt

# Run specific analysis types in batch mode
python main.py --samples-batch /path/to/batch_folder --functional --data DeepSTARR_data.h5 --model oracle_model.ckpt
python main.py --samples-batch /path/to/batch_folder --test "motif_enrichment,percent_identity" --data DeepSTARR_data.h5 --model oracle_model.ckpt
```

#### Supported Directory Structures

**Flat Structure**: Each NPZ file is one sample
```
batch_folder/
├── sample1.npz
├── sample2.npz
├── sample3.npz
└── metadata.csv (auto-generated)
```

**Nested Structure**: Multiple NPZ files per sample (multiple data points)
```
batch_folder/
├── sample1/
│   ├── run_001.npz
│   ├── run_002.npz
│   └── run_003.npz
├── sample2/
│   ├── run_001.npz
│   └── run_002.npz
└── metadata.csv (auto-generated)
```

### With Environment Variables
```bash
export SAMPLES_FILE=samples.npz
export DATA_FILE=DeepSTARR_data.h5  
export MODEL_FILE=oracle_DeepSTARR_DeepSTARR_data.ckpt
python analysis/main.py
```

### Selective Analysis

```bash
# Run only functional similarity tests
python main.py --functional

# Run multiple similarity types
python main.py --functional --sequence

# Run specific tests
python main.py --test frechet_distance
python main.py --test "motif_enrichment,percent_identity"

# Custom output directory
python main.py --output-dir my_results

# Custom motif database for compositional similarity tests
python main.py --compositional --motif-db /path/to/custom_motifs.meme
python main.py --test motif_enrichment --motif-db /path/to/custom_motifs.meme
```

## Analysis Components

The pipeline is organized by similarity type as described in the evaluation framework paper:

#### Functional Similarity
- **Conditional Generation Fidelity** - MSE between oracle predictions 
- **Fréchet Distance** - Distribution comparison of oracle embeddings
- **Predictive Distribution Shift** - Kolmogorov-Smirnov test on predictions

#### Sequence Similarity  
- **Percent Identity** - Normalized Hamming distance for memorization/diversity
- **k-mer Spectrum Shift** - Jensen-Shannon divergence of k-mer frequencies
- **Discriminability** - Binary classifier AUROC score

#### Compositional Similarity
- **Motif Enrichment** - Pearson correlation of motif occurrence counts
- **Motif Co-occurrence** - Frobenius norm of motif covariance matrices
- **Attribution Consistency** - KL divergence of attribution patterns

## Output

Results are saved **immediately** after each analysis completes to timestamped directories under `results/` (or custom `--output-dir`):

### Single Sample Mode Output
```
results/analysis_results_2024-01-15_14-30-25/
├── attribution_consistency_2024-01-15_14-31-45.pkl     # Individual analysis results
├── functional_similarity_2024-01-15_14-35-12.pkl       # (saved immediately)
├── motifs_2024-01-15_14-38-30.pkl                      # 
├── all_results_combined.pkl                            # Combined results (updated after each analysis)
└── analysis_progress.json                              # Progress tracking (updated after each analysis)
```

### Batch Mode Output (NEW!)

Batch mode creates **two output formats** for each analysis:

```
results/analysis_results_2024-01-15_14-30-25/
├── motif_enrichment.csv                    # Concise metrics: samples as columns, metrics as rows
├── motif_enrichment.h5                     # Full results: comprehensive data per sample
├── cond_gen_fidelity.csv                   # Key metric: conditional_generation_fidelity_mse
├── cond_gen_fidelity.h5                    # Full results: predictions, embeddings, etc.
├── percent_identity.csv                    # Key metric: average_max_percent_identity_samples_vs_training
├── percent_identity.h5                     # Full results: identity matrices, detailed stats
└── ... (all other analyses)
```

#### Concise CSV Format
Each analysis produces a CSV with **key metrics only**:

| Analysis | Key Metrics | Description |
|----------|-------------|-------------|
| `attribution_consistency.csv` | KLD, KLD_concat | KL divergence values |
| `motif_cooccurrence.csv` | frobenius_norm | Matrix comparison metric |
| `motif_enrichment.csv` | pearson_r_statistic | Correlation coefficient |
| `cond_gen_fidelity.csv` | conditional_generation_fidelity_mse | Mean squared error |
| `frechet_distance.csv` | frechet_distance | Distribution distance |
| `predictive_dist_shift.csv` | predictive_distribution_shift_ks_statistic | KS test statistic |
| `discriminability.csv` | auroc | Area under ROC curve |
| `kmer_spectrum_shift.csv` | js_distance | Jensen-Shannon distance |
| `percent_identity.csv` | average_max_percent_identity_samples_vs_training | Identity score |

**CSV Structure**: 
- **Columns**: Sample names
- **Rows**: Metric values (multiple rows for nested structure with multiple NPZ files per sample)

#### Full HDF5 Format
Each analysis produces an HDF5 file with **comprehensive results**:
- Arrays and matrices (e.g., motif counts, identity matrices)
- Detailed statistics and intermediate computations
- Sample-specific groups for organized data access

### Progress Tracking
- **`analysis_progress.json`**: Real-time progress with completed analyses and summary
- **`all_results_combined.pkl`**: Updated after each analysis completes
- **Individual files**: Each analysis saves its own timestamped pickle file

### Benefits
- **Fault tolerance**: If one analysis fails, others continue and results are preserved
- **Real-time monitoring**: Check progress without waiting for completion
- **Partial results**: Access completed analyses immediately
- **Resume capability**: Know exactly which analyses completed successfully
- **Batch processing**: Handle multiple samples efficiently with organized outputs
- **Dual formats**: Concise CSVs for quick analysis + comprehensive HDF5 for detailed examination
- **Flexible structure**: Support both flat and nested directory organizations

## Batch Mode Workflow

### 1. Initial Setup
```bash
# First run creates metadata template
python main.py --samples-batch /path/to/samples --data data.h5 --model model.ckpt
# Output: "CSV wasn't found, template created, can modify the sample names before running again"
```

### 2. Optional Customization
Edit the auto-generated `metadata.csv` to customize sample names:
```csv
sample_name,file_path,created_date
my_sample_1,sample1.npz,2024-01-15_10-30-45
my_experiment_2,sample2.npz,2024-01-15_10-30-45
control_group,sample3.npz,2024-01-15_10-30-45
```

### 3. Run Analysis
```bash
# Process all samples with customized names
python main.py --samples-batch /path/to/samples --data data.h5 --model model.ckpt
```

### 4. Results
- **Concise CSVs**: Quick overview with key metrics
- **Full HDF5 files**: Comprehensive data for detailed analysis
- **Sample columns**: Easy comparison across experiments

## Requirements

Install using pip:

```bash
# Basic installation with pymemesuite fallback
pip install -e .

# Install with memelite support (recommended)
pip install -e .[memelite]

# Full installation with all optional dependencies
pip install -e .[all]
```

Or using the pyproject.toml:

```bash
pip install -e .
```

## Package Fallback System

The analysis pipeline uses an intelligent fallback system for motif analysis:

1. **Primary**: Uses `memelite` functions (`motif_count_memelite`, `make_occurrence_matrix_memelite`)
   - Faster, more modern implementation
   - Same API as tangermeme but with memelite backend

2. **Fallback**: Uses `pymemesuite` functions if memelite is unavailable
   - Ensures compatibility when memelite can't be installed
   - Original implementation preserved

The fallback is automatic and transparent - no code changes needed.

## Oracle Model Requirements

Each test's dependency on the oracle model:

| Test Name | Requires Oracle Model | Similarity Type | Script Location |
|-----------|----------------------|-----------------|------------------|
| **Conditional Generation Fidelity** | ✅ Yes | Functional | `core/functional/cond_gen_fidelity.py` |
| **Fréchet Distance** | ✅ Yes | Functional | `core/functional/frechet_distance.py` |
| **Predictive Distribution Shift** | ✅ Yes | Functional | `core/functional/predictive_dist_shift.py` |
| **Percent Identity** | ❌ No | Sequence | `core/sequence/percent_identity.py` |
| **k-mer Spectrum Shift** | ❌ No | Sequence | `core/sequence/kmer_spectrum_shift.py` |
| **Discriminability** | ❌ No* | Sequence | `core/sequence/discriminability.py` |
| **Motif Enrichment** | ❌ No* | Compositional | `core/compositional/motif_enrichment.py` |
| **Motif Co-occurrence** | ❌ No* | Compositional | `core/compositional/motif_cooccurrence.py` |
| **Attribution Consistency** | ✅ Yes | Compositional | `core/compositional/attribution_consistency.py` |

*Discriminability trains its own binary classifier, so it doesn't require the oracle model.

**Motif tests require a motif database file (defaults to JASPAR2024_CORE_non-redundant_pfms_meme.txt, can be customized with `--motif-db`).

### Tests Requiring Oracle Model (4/9)
These tests need the trained DeepSTARR model to compute predictions or embeddings:
- Conditional Generation Fidelity
- Fréchet Distance  
- Predictive Distribution Shift
- Attribution Consistency

### Tests Not Requiring Oracle Model (5/9)
These tests work directly with sequence data:
- Percent Identity
- k-mer Spectrum Shift
- Discriminability (uses own classifier)
- Motif Enrichment
- Motif Co-occurrence