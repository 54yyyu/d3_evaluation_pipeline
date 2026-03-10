# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

D3 Sequence Analysis Pipeline — a bioinformatics evaluation framework for comparing synthetic DNA sequences against real genomic data. It computes 9 metrics across three similarity dimensions (functional, sequence, compositional) using oracle models (DeepSTARR, MPRALegNet/lentimpra, SEI).

## Commands

```bash
# Install
pip install -e .            # basic install
pip install -e .[memelite]  # recommended (faster motif analysis)
pip install -e .[all]       # includes dev dependencies

# Run analysis (single sample)
python main.py --samples samples.npz --data data.h5 --model model.ckpt
python main.py --samples samples.npz --data data.h5 --model model.ckpt --model-type lentimpra

# Run specific tests
python main.py --test cond_gen_fidelity --samples s.npz --data d.h5 --model m.ckpt
python main.py --test "motif_enrichment,percent_identity" --samples s.npz --data d.h5 --model m.ckpt

# Run by similarity type
python main.py --functional --samples s.npz --data d.h5 --model m.ckpt
python main.py --sequence --compositional --samples s.npz --data d.h5 --model m.ckpt

# Batch mode (multiple sample files in a directory)
python main.py --samples-batch /path/to/batch_folder --data d.h5 --model m.ckpt

# Formatting & tests
black --line-length 100 .
flake8 --max-line-length 100 .
pytest                       # runs tests in tests/ with coverage on core/ and utils/
```

## Architecture

**Entry point**: `main.py` — CLI runner that loads data/model, determines which analyses to run, and orchestrates execution. Supports single-sample and batch modes.

**`core/`** — 9 analysis modules organized by similarity type:
- `core/functional/` — Requires oracle model: `cond_gen_fidelity`, `frechet_distance`, `predictive_dist_shift`
- `core/sequence/` — Model-agnostic: `percent_identity`, `kmer_spectrum_shift`, `discriminability`
- `core/compositional/` — `motif_enrichment`, `motif_cooccurrence` (need motif DB file), `attribution_consistency` (needs oracle model)

Each module exports a `run_*_analysis()` function called from `main.py`. In batch mode, results append to per-analysis CSV (key metrics) and HDF5 (full data) files.

**`utils/helpers.py`** — Data loading (`extract_data`, `extract_lentimpra_data`, `extract_sei_data`), tensor conversion, model loading. Handles both NPZ and HDF5 formats with various key naming conventions.

**`utils/batch_helpers.py`** — Batch sample discovery from directories (flat or nested structure), metadata CSV generation.

**`utils/seq_evals_func_motifs.py`** — Motif scanning functions with memelite/pymemesuite fallback.

**Oracle models** (top-level files):
- `deepstarr.py` — DeepSTARR (249bp sequences), PyTorch Lightning
- `mpralegnet.py` — MPRALegNet (230bp sequences), PyTorch Lightning
- `sei.py` — SEI (4096bp sequences, padded), plain PyTorch

## Key Conventions

- Data tensors use shape `(N, 4, seq_len)` — 4 channels for one-hot encoded nucleotides (ACGT)
- Input files may be NPZ or HDF5 with inconsistent key names (`arr_0`, `first_sample`, `X_test`, `x_test`, `onehot_test`, etc.) — the code tries multiple keys
- SEI sequences are center-padded to 4096bp with 0.25 uniform background
- Results saved to timestamped directories under `results/` (or `--output-dir`)
- Motif analyses default to JASPAR2024 database file; override with `--motif-db`
- `--model-type` accepts: `deepstarr` (default), `mpralegnet`, `lentimpra`, `sei`
