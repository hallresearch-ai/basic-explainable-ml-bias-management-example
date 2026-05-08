# Basic Explainable ML Example with Bias Management

This repository is a small educational example for training an interpretable
supervised model and checking basic group fairness metrics. It uses a credit
line delinquency dataset, trains a monotonic calibrated linear TensorFlow
Lattice model, explains the fitted model with calibrator curves, and compares a
few simple bias remediation strategies.

The notebook keeps demographic columns available for analysis, but excludes
them from model training. The goal is to demonstrate a compact workflow for
model transparency and bias management, not to provide compliance or legal
advice.

## Background Reading

This example is related to monotonic, shape-constrained modeling and practical
bias testing/remediation workflows. Useful background references include:

- "Fast and Flexible Monotonic Functions with Ensembles of Lattices":
  https://jmlr.org/papers/volume17/15-243/15-243.pdf

- *Machine Learning for High-Risk Applications*, Chapter 10, "Testing and
  Remediating Bias with XGBoost":
  https://www.oreilly.com/library/view/machine-learning-for/9781098102425/

## Contents

- `Monotonic_TFL_Explainability_Bias_Code_Apache2.ipynb`: Jupyter notebook that
  trains the monotonic TensorFlow Lattice model, explains it, tests group
  fairness metrics, and compares remediation strategies.
- `utils.py`: Helper functions for monotonicity detection, model construction,
  threshold selection, fairness tables, plotting, and model comparison.
- `credit_line_increase.csv`: Example credit line delinquency dataset. The
  target column is `DELINQ_NEXT`; demographic columns are used for analysis and
  excluded from model features.
- `requirements.txt`: Python dependencies for running the notebook.
- `LICENSE`: Apache License 2.0.

## What the Notebook Demonstrates

The notebook walks through a basic explainability and bias management workflow:

1. Load the credit line data and map coded demographic fields to readable
   labels.
2. Define modeling columns so demographic fields are retained for analysis but
   excluded from model training.
3. Split the data into training and validation sets.
4. Derive simple monotonicity constraints from training-set Spearman
   correlations.
5. Train a monotonic calibrated linear TensorFlow Lattice model.
6. Explain the model with training curves, ROC and score-distribution plots,
   output weights, and learned calibrator curves.
7. Compute validation-set group metrics by race, including adverse impact ratio
   (AIR), false positive rate (FPR), and false negative rate (FNR).
8. Apply three simple remediation strategies:
   - race-based reweighing
   - small one-feature-drop search
   - small hyperparameter search
9. Compare validation AUC, selected cutoff, and group fairness metrics across
   the baseline and remediated model variants.

## Setup

From the repository root, create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Start Jupyter:

```bash
jupyter notebook
```

Then open `Monotonic_TFL_Explainability_Bias_Code_Apache2.ipynb` and run the
cells in order.

## License

This repository is licensed under the Apache License, Version 2.0. See
`LICENSE` for details.
