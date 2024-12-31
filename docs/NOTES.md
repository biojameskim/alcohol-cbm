# Notes

6 Models being trained.
- SC, FC, FCgsr, demographics, ensemble, simple ensemble

We have 4 datasets that logreg was run on:
control_only: 1000 iterations
control_moderate: 1000 iterations
control_moderate (male): 1000 iterations
control_moderate (female): 100 iterations

Permutation test was done on each of these 4 datasets (for 100 iterations):
- So the y (target labels) was permuted 100 times
control_only: 100 iterations
control_moderate: 100 iterations
control_moderate (male): 100 iterations
control_moderate (female): 100 iterations