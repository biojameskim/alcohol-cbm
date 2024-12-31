#!/bin/bash

# Name: run_logreg_and_visualize.py

python src/logreg_matrices.py; # Train and extract metrics for all 4 logreg models (SC, FC, FCgsr, demographics)
python src/logreg_ensemble.py; # Train and extract metrics for ensemble model

python scripts/roc_curve.py; # Generate ROC curves for all logreg models (SC, FC, FCgsr, demographics, ensemble)
python scripts/violin_plot.py; # Generate violin plots of (accuracy, balanced accuracy, roc_aucs) for all logreg models 
python scripts/sig_testing.py; # Visualize significant edges (coefficients) for SC, FC, FCgsr