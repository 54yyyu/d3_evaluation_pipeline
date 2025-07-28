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
│   └── seq_evals_improved.py
└── training/                  # Training scripts (separated)
    ├── EvoAug_run_train.py
    └── finetune_run_train.py
```

## Usage

### Basic Usage
```bash
python analysis/main.py --samples samples.npz --data DeepSTARR_data.h5 --model oracle_DeepSTARR_DeepSTARR_data.ckpt
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
# Skip attribution analysis
python analysis/main.py --skip-attribution

# Run only motif analysis
python analysis/main.py --skip-attribution --skip-functional

# Custom output directory
python analysis/main.py --output-dir my_results
```

## Analysis Components

1. **Attribution Consistency Analysis** - Evaluates attribution consistency using SHAP and entropic information
2. **Functional Similarity Analysis** - Measures fidelity, Frechet distance, distribution shift, and k-mer statistics  
3. **Motif Analysis** - Performs motif enrichment and co-occurrence analysis

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