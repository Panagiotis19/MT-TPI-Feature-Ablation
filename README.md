# Feature Ablation of a Physics-Informed Multi-Task Transformer for UAV Trajectory Prediction and State Identification

BSc Thesis, Department of Electrical and Computer Engineering, University of Cyprus, 2026.

**Author:** Panagiotis Averof  
**Supervisor:** Prof. George Ellinas  
**Original model and dataset:** Dr. Nicolas Souli and the KIOS Research and Innovation Center of Excellence

## Overview

This repository contains the source code, dataset, and experimental results for a systematic feature ablation study of an existing physics-informed multi-task Transformer model. The model jointly performs UAV trajectory forecasting and flight-state classification. The thesis re-implements the model in PyTorch and conducts 190 ablation experiments to identify a reduced 17-feature input configuration that strictly improves on the 20-feature baseline.

## Repository Structure

| File | Description |
|---|---|
| `Transformer_physicsinformed_pytorch.py` | PyTorch implementation of the model and training loop |
| `scheduler_ablation.py` | Cascaded feature ablation scheduler (190 experiments) |
| `scheduler_hyperparameter.py` | Hyperparameter grid runner across (FH, WS) configurations |
| `scan_results.py` | Aggregates per-experiment results |
| `plot_3D.py`, `plot_CDF.py`, `plot_MAE.py` | Visualisation scripts |
| `Framework.png` | Architecture diagram |
| `Weather_Dataset_final.csv` | Input dataset |
| `feature_ablation_results.html` | Interactive dashboard of all ablation results |
| `TABLES.html` | Summary tables for the thesis |
| `Trajectory Dashboard.html` | Trajectory visualisation dashboard |
| `all_results.json` | Aggregated results in JSON format |
| `requirements.txt` | Python dependencies |

## Installation

    git clone https://github.com/yourusername/MT-TPI-Feature-Ablation.git
    cd MT-TPI-Feature-Ablation
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

## Usage

Run the baseline training:

    python Transformer_physicsinformed_pytorch.py

Run the cascaded ablation study:

    python scheduler_ablation.py

Run the hyperparameter sweep:

    python scheduler_hyperparameter.py

## Key Results

The recommended 17-feature configuration removes `orientation_y`, `wind_angle`, and `angular_z`. At the reference configuration (FH=30, WS=90), it achieves:

- Classification accuracy improvement: +0.61% over baseline
- Trajectory MAE improvement: -0.0001 over baseline
- 15% reduction in input dimensionality
- The only configuration in 190 experiments that strictly dominates the baseline on both heads

## Citation

If you use this code or its findings, please cite:

    Averof, P. (2026). Feature Ablation of a Physics-Informed Multi-Task Transformer 
    for UAV Trajectory Prediction and State Identification. BSc Thesis, 
    University of Cyprus.

## License

MIT License. See LICENSE file for details.

## Acknowledgments

The model architecture and the dataset were provided by Dr. Nicolas Souli and the KIOS Research and Innovation Center of Excellence, University of Cyprus.
