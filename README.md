# Spatial Gradient Benchmark

Multi-model spatial gradient comparison benchmark for neural field methods in spatial transcriptomics. Evaluates five neural field models on spatial gradient detection across the DLPFC dataset.

## Background

This benchmark compares AINR against four published neural field methods---GASTON, SUICA, STINR, and GASTON-Mix---on the task of spatial gradient identification. Spatial gradient detection is a unique capability of neural field approaches: by modeling gene expression as a continuous function of spatial coordinates, these methods enable direct computation of spatial gradients through automatic differentiation.

## Directory Structure

```
spatial_gradient_benchmark/
├── config.py              # Shared configuration (paths, slices, seeds)
├── data_utils.py          # Data loading + gradient computation utilities
├── visualization.py       # Unified visualization (gradient maps, GT comparison)
├── setup_models.py        # Download all 4 comparison models
├── run_all.py             # Batch run all experiments
├── compare_results.py     # Multi-model comparison and analysis
├── results/               # Experiment outputs
├── ainr/run_gradient.py   # AINR gradient experiment
├── gaston/run_gradient.py # GASTON gradient experiment
├── stinr/run_gradient.py  # STINR gradient experiment
├── suica/run_gradient.py  # SUICA gradient experiment
├── gaston_mix/run_gradient.py  # GASTON-Mix gradient experiment
└── models/                # Comparison model repositories (run setup_models.py)
    ├── gaston/src/        # GASTON official code
    ├── gaston_mix/src/    # GASTON-Mix official code
    ├── stinr/STINR/       # STINR official code
    └── suica/             # SUICA official code
```

## Prerequisites

### Data Directory

```
<PROJECT_ROOT>/
├── Data/                          # DLPFC Visium data
│   └── 151673/
│       └── filtered_feature_bc_matrix.h5
├── DLPFC_annotations/             # Ground truth annotations
│   └── 151673_truth.txt
└── spatial_gradient_benchmark/    # This project
```

### Conda Environment

```bash
conda create -n AINR python=3.10 -y && conda activate AINR

# Core dependencies
pip install torch scanpy anndata numpy pandas matplotlib scipy scikit-learn

# SUICA additional dependencies
pip install lightning omegaconf rich kornia einops pyro-ppl tensorboard

# GASTON (optional pip install)
pip install gaston-spatial
```

## Quick Start

### 1. Download Comparison Model Repositories

```bash
cd spatial_gradient_benchmark
python setup_models.py
```

### 2. Run Individual Models

```bash
python stinr/run_gradient.py --slice 151673      # STINR
python gaston/run_gradient.py --slice 151673     # GASTON
python gaston_mix/run_gradient.py --slice 151673 # GASTON-Mix
python suica/run_gradient.py --slice 151673      # SUICA
```

### 3. Batch Run

```bash
# 4 slices x 4 models = 16 experiments
python run_all.py --slices 151673 151674 151675 151676 --models gaston suica stinr gaston_mix
```

## Model Parameters Reference

| Parameter | STINR | GASTON | GASTON-Mix | SUICA |
|-----------|-------|--------|------------|-------|
| Training | 14001 steps | 10000 epochs | 10000 epochs | 200 + 2000 epochs |
| Optimizer | Adamax lr=1e-3 | Adam lr=1e-3 | Adam lr=1e-3 | Adam lr=1e-5 / AdamW lr=1e-4 |
| Architecture | DeconvNet (INR+Deconv) | Isodepth NF (2→20→20→1) | MoE NF (Linear PE→1) | GAE + FFN INR |
| Gradient Source | d(latent)/d(coord) | d(isodepth)/d(coord) | d(isodepth_k)/d(coord) | d(embed)/d(coord) |
| Loss | Poisson_NLL+L2(mid)+L2(recon) | MSE | MSE + Routing | L1+MSE |
| Seed | 1 | 112 | 112 | 8848 |
| Data | 4-slice joint (73-76h5ad) | Single slice (HVG=2000) | Single slice (HVG=2000) | Single slice (all genes) |

## Output

Each model x slice combination generates files under `results/<Model>_gradient/<slice>/`:
- `spatial_gradient_values.csv` — Spot-level gradient values
- `gradient_boundary.pdf` — Spatial gradient heatmap
- `gradient_vs_gt.pdf` — Gradient vs Ground Truth dual panel
- `gradient_histogram.pdf` — Gradient distribution histogram
- `summary.csv` — Experiment parameters and statistics

## Metrics

| Metric | Description | Direction |
|--------|-------------|-----------|
| **Gradient Mean** | Average gradient magnitude | No absolute standard |
| **Contrast Ratio** | Boundary spot gradient / interior spot gradient | Higher is better (>1) |
| **AUROC** | Gradient as boundary detection score ranking | Higher is better |
| **Moran's I** | Spatial autocorrelation of gradient values | Higher is better |

## Evaluation Metrics

Three quantitative metrics for gradient quality assessment:
- **Contrast Ratio**: Ratio of mean gradient at boundary spots (adjacent to different layers) to interior spots. Values >1 indicate boundary enhancement.
- **AUROC**: Area under ROC curve using gradient as boundary detection score against ground truth layer boundaries.
- **Moran's I**: Spatial autocorrelation of gradient values (+1 = structured, 0 = random noise).

## Citation

If you use this benchmark in your research, please cite both the AINR paper and the original papers of the compared methods (GASTON, SUICA, STINR, GASTON-Mix).

## License

This project is provided for research purposes. See individual model directories for their respective licenses.
