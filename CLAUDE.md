# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

D3 Sequence Analysis Pipeline — a unified evaluation pipeline for DNA sequence generation models. Compares synthetic sequences against reference (test) data using 9 analyses across 3 similarity types: functional, sequence, and compositional.

## Commands

```bash
# Install (editable)
pip install -e .          # basic
pip install -e .[all]     # with all extras (memelite, dev tools)

# Format & lint
black . --line-length=100
flake8 . --max-line-length=100 --extend-ignore=E203,W503

# Tests
pytest tests/
pytest tests/test_foo.py::test_bar  # single test
```

## Architecture

**Entry point:** `main.py` — CLI orchestrator with three operating modes:
1. **Single sample:** `--samples` + `--data` + `--model`
2. **Batch:** `--samples-batch` (directory of sample files)
3. **Multi-oracle:** `--model` + `--model2` + `--model3` (3 MPRALegNet models)

**Oracle models** (top-level):
- `deepstarr.py` — DeepSTARR model (249bp sequences)
- `mpralegnet.py` — MPRALegNet/LentIMPRA model (230bp sequences)

**Analysis modules** (`core/`), organized by similarity type:
- `core/functional/` — Requires oracle model: conditional generation fidelity (MSE), Fréchet distance (embeddings), predictive distribution shift (KS test)
- `core/sequence/` — No oracle needed: percent identity (Hamming), k-mer spectrum shift (JSD), discriminability (AUROC)
- `core/compositional/` — Motif enrichment (Pearson), motif co-occurrence (Frobenius), attribution consistency (KL divergence, requires oracle)

All `run_*_analysis()` functions follow the same signature pattern: `(oracle_model, x_test_tensor, x_synthetic_tensor, ...)` returning a dict of metric results.

**Utilities** (`utils/`):
- `helpers.py` — Data loading, file format detection (NPZ/HDF5/PT), encoding detection (index vs one-hot), model loading. Uses a priority-based key resolution system for multi-key files.
- `batch_helpers.py` — Batch sample discovery, CSV metadata generation, supports flat and nested directory structures.
- `seq_evals_func_motifs.py` — Motif counting via memelite (preferred) with pymemesuite fallback.

## Data Formats

- Sequences can be one-hot `(N, L, 4)` or `(N, 4, L)` (auto-transposed) or index-encoded
- Supported file types: `.npz`, `.h5`/`.hdf5`, `.pt`
- Batch mode outputs CSV + HDF5; single mode outputs pickles

## Code Style

- **Black** with 100-char line length, targeting Python 3.8–3.11
- **Flake8** with 100-char lines, ignoring E203 and W503
- Snake_case for functions/variables, CamelCase for classes
