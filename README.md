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

### Basic Usage

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

### File Structure
```
results/analysis_results_2024-01-15_14-30-25/
├── attribution_consistency_2024-01-15_14-31-45.pkl     # Individual analysis results
├── functional_similarity_2024-01-15_14-35-12.pkl       # (saved immediately)
├── motifs_2024-01-15_14-38-30.pkl                      # 
├── all_results_combined.pkl                            # Combined results (updated after each analysis)
└── analysis_progress.json                              # Progress tracking (updated after each analysis)
```

### Progress Tracking
- **`analysis_progress.json`**: Real-time progress with completed analyses and summary
- **`all_results_combined.pkl`**: Updated after each analysis completes
- **Individual files**: Each analysis saves its own timestamped pickle file

### Benefits
- **Fault tolerance**: If one analysis fails, others continue and results are preserved
- **Real-time monitoring**: Check progress without waiting for completion
- **Partial results**: Access completed analyses immediately
- **Resume capability**: Know exactly which analyses completed successfully

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
| **Predictive Distribution Shift** | ❌ No | Functional | `core/functional/predictive_dist_shift.py` |
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
- Attribution Consistency

### Tests Not Requiring Oracle Model (5/9)
These tests work directly with sequence data:
- Predictive Distribution Shift
- Percent Identity
- k-mer Spectrum Shift
- Discriminability (uses own classifier)
- Motif Enrichment
- Motif Co-occurrence