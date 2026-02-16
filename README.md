# PKPD Paracetamol Simulation

Small project for my **bachelors's thesis** .

## Overview
This repository contains a simple PK/PD (pharmacokinetic/pharmacodynamic) simulation for paracetamol, plus a least‑squares fitting script.

## Files
- `ParacetamolRazlaganjePkPd.py` — PK/PD simulation.
- `LeastSquaresOptimization.py` — parameter fitting via least squares.
- `paracetamol_true.csv` — sample/ground‑truth data.

## Quick Start
```bash
python3 ParacetamolRazlaganjePkPd.py
python3 LeastSquaresOptimization.py
```

## Notes
Adjust parameters inside the scripts to change the model assumptions or fit targets.
