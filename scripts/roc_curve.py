import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(matrix_type, control_only, male, female, save_fig=False):
  """
  Plot ROC curve.

  Args:
  - true_labels: Aggregated true labels from nested cross-validation.
  - pred_probs: Aggregated predicted probabilities from nested cross-validation.
  """
  if control_only:
    file_name = 'control'
  else:
    file_name = 'control_moderate'
  
  if male:
    sex = '_male'
  elif female:
    sex = '_female'
  else:
    sex = ''

  true_labels = np.load(f'results/{matrix_type}/logreg/{file_name}/logreg_{matrix_type}_{file_name}{sex}_true_labels.npy')
  pred_probs = np.load(f'results/{matrix_type}/logreg/{file_name}/logreg_{matrix_type}_{file_name}{sex}_pred_probs.npy')

  fpr, tpr, _ = roc_curve(true_labels, pred_probs)
  roc_auc = auc(fpr, tpr)

  plt.figure()
  plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
  plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title(f'ROC Curve for {matrix_type} matrix ({file_name}{sex})')
  plt.legend(loc="lower right")
  if save_fig:
    plt.savefig(f'figures/roc_curve/{file_name}/roc_curve_{matrix_type}_{file_name}{sex}.png')
  else:
    plt.show()

def plot_roc_curve_combined(matrix_types, control_only, male, female, save_fig=False):
  """
  Plot multiple ROC curves on the same plot.

  Args:
  - matrix_types: List of matrix types to plot.
  - control_only: Boolean indicating whether to use control only data.
  - male: Boolean indicating whether to filter for male.
  - female: Boolean indicating whether to filter for female.
  - save_fig: Boolean indicating whether to save the figure.
  """
  if control_only:
      file_name = 'control'
  else:
      file_name = 'control_moderate'
  
  if male:
      sex = '_male'
  elif female:
      sex = '_female'
  else:
      sex = ''

  plt.figure()

  for matrix_type in matrix_types:
    true_labels = np.load(f'results/{matrix_type}/logreg/{file_name}/logreg_{matrix_type}_{file_name}{sex}_true_labels.npy')
    pred_probs = np.load(f'results/{matrix_type}/logreg/{file_name}/logreg_{matrix_type}_{file_name}{sex}_pred_probs.npy')
    roc_auc_scores = np.load(f'results/{matrix_type}/logreg/{file_name}/logreg_{matrix_type}_{file_name}{sex}_roc_aucs.npy')

    fpr, tpr, _ = roc_curve(true_labels, pred_probs)
    roc_auc = np.mean(roc_auc_scores, axis=0)

    if matrix_type == 'demos':
      matrix_type = 'Demographics'
    elif matrix_type == 'simple_ensemble':
      matrix_type = 'Ensemble'

    plt.plot(fpr, tpr, lw=2, label=f'{matrix_type} (AUC = {roc_auc:.2f})')

  plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title(f'ROC Curves for Model Type (Male)')
  plt.legend(loc="lower right")
  
  if save_fig:
      plt.savefig(f'figures/roc_curve/{file_name}/roc_curve_{file_name}{sex}.png')
  else:
      plt.show()

if __name__ == "__main__":  
  CONTROL_ONLY = False
  MALE = False
  FEMALE = False
  SAVE_FIG = True

  # plot_roc_curve('SC', control_only=CONTROL_ONLY, male=MALE, female=FEMALE, save_fig=SAVE_FIG)
  # plot_roc_curve('FC', control_only=CONTROL_ONLY, male=MALE, female=FEMALE, save_fig=SAVE_FIG)
  # plot_roc_curve('FCgsr', control_only=CONTROL_ONLY, male=MALE, female=FEMALE, save_fig=SAVE_FIG)
  # plot_roc_curve('demos', control_only=CONTROL_ONLY, male=MALE, female=FEMALE, save_fig=SAVE_FIG)

  # plot_roc_curve('ensemble', control_only=CONTROL_ONLY, male=MALE, female=FEMALE, save_fig=SAVE_FIG)
  # plot_roc_curve('simple_ensemble', control_only=CONTROL_ONLY, male=MALE, female=FEMALE, save_fig=SAVE_FIG)

  plot_roc_curve_combined(['SC', 'FC', 'demos', 'simple_ensemble'], control_only=CONTROL_ONLY, male=MALE, female=FEMALE, save_fig=SAVE_FIG)