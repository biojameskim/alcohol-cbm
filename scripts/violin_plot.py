import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def create_violin_plot_all(metric, control_only, male, female, save_fig=False):
  """
  create_violin_plot() creates a violin plot of [metric] for SC, FC, FCgsr, and (simple) demographics models.
  [metric] should be a string of either 'accuracies', 'balanced_accuracies', or 'roc_aucs'.
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

  SC_metric = np.load(f'results/SC/logreg/{file_name}/logreg_SC_{file_name}{sex}_{metric}.npy')
  FC_metric = np.load(f'results/FC/logreg/{file_name}/logreg_FC_{file_name}{sex}_{metric}.npy')
  FCgsr_metric = np.load(f'results/FCgsr/logreg/{file_name}/logreg_FCgsr_{file_name}{sex}_{metric}.npy')
  demos_metric = np.load(f'results/demos/logreg/{file_name}/logreg_demos_{file_name}{sex}_{metric}.npy')

  # Create a DataFrame for easier plotting
  data = pd.DataFrame({
      'Scores': np.concatenate([SC_metric, FC_metric, FCgsr_metric, demos_metric]),
      'Category': ['SC'] * len(SC_metric) + ['FC'] * len(FC_metric) + ['FCgsr'] * len(FCgsr_metric) + ['demos'] * len(demos_metric)
  })

  if metric == "accuracies":
    desc = "Accuracy"
  elif metric == "balanced_accuracies":
    desc = "Balanced Accuracy"
  else:
    desc = "ROC AUC"

  # Create a violin plot with the four arrays side-by-side
  plt.figure(figsize=(10, 6))
  sns.violinplot(x='Category', y='Scores', data=data, hue='Category', palette='muted', legend=False)
  plt.title(f'{desc} Scores for Connectivity Matrices and Demographics ({file_name}{sex})')
  plt.xlabel('Model Type')
  plt.ylabel(f'{desc} Scores')
  plt.ylim(0.45, 0.70)
  if save_fig:
    plt.savefig(f'figures/violin_plot/{file_name}/violin_plot_{metric}_{file_name}{sex}.png')
  else:
    plt.show()

def create_violin_plot_ensemble(simple, control_only, male, female, save_fig=False):
  """
  Create a violin plot of accuracies, balanced accuracies, and ROC AUCs for ensemble
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

  if simple:
    title = 'simple ensemble'
    ensemble_acc = np.load(f'results/simple_ensemble/logreg/{file_name}/logreg_simple_ensemble_{file_name}{sex}_accuracies.npy')
    ensemble_bal_acc = np.load(f'results/simple_ensemble/logreg/{file_name}/logreg_simple_ensemble_{file_name}{sex}_balanced_accuracies.npy')
    ensemble_roc_aucs = np.load(f'results/simple_ensemble/logreg/{file_name}/logreg_simple_ensemble_{file_name}{sex}_roc_aucs.npy')
  else:
    title = 'ensemble'
    ensemble_acc = np.load(f'results/ensemble/logreg/{file_name}/logreg_ensemble_{file_name}{sex}_accuracies.npy')
    ensemble_bal_acc = np.load(f'results/ensemble/logreg/{file_name}/logreg_ensemble_{file_name}{sex}_balanced_accuracies.npy')
    ensemble_roc_aucs = np.load(f'results/ensemble/logreg/{file_name}/logreg_ensemble_{file_name}{sex}_roc_aucs.npy')

  data = pd.DataFrame({
      'Scores': np.concatenate([ensemble_acc, ensemble_bal_acc, ensemble_roc_aucs]),
      'Category': ['Accuracies'] * len(ensemble_acc) + ['Balanced Accuracies'] * len(ensemble_bal_acc) + ['ROC AUCs'] * len(ensemble_roc_aucs)
  })

  # Create a violin plot with the three arrays side-by-side
  plt.figure(figsize=(10, 6))
  sns.violinplot(x='Category', y='Scores', data=data, hue='Category', palette='muted', legend=False)
  plt.title(f'Metrics for {title} model ({file_name}{sex})')
  plt.xlabel('Metrics')
  plt.ylabel(f'Scores')
  plt.ylim(0.45, 0.70)
  if save_fig:
    plt.savefig(f'figures/violin_plot/{file_name}/violin_plot_{title}_{file_name}{sex}.png')
  else:
    plt.show()

def create_violin_plot(metric, control_only, male, female, save_fig=False):
  """
  create_violin_plot() creates a violin plot of [metric] for SC, FC, demographics, and simple ensemble models
  [metric] should be a string of either 'accuracies', 'balanced_accuracies', or 'roc_aucs'.
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

  SC_metric = np.load(f'results/SC/logreg/{file_name}/logreg_SC_{file_name}{sex}_{metric}.npy')
  FC_metric = np.load(f'results/FC/logreg/{file_name}/logreg_FC_{file_name}{sex}_{metric}.npy')
  demos_metric = np.load(f'results/demos/logreg/{file_name}/logreg_demos_{file_name}{sex}_{metric}.npy')
  ensemble_metric = np.load(f'results/simple_ensemble/logreg/{file_name}/logreg_simple_ensemble_{file_name}{sex}_{metric}.npy')

  max_values = np.max([np.max(SC_metric), np.max(FC_metric), np.max(demos_metric), np.max(ensemble_metric)])
  print(f'Max values: {max_values}')
  min_values = np.min([np.min(SC_metric), np.min(FC_metric), np.min(demos_metric), np.min(ensemble_metric)])
  print(f'Min values: {min_values}')

  # Create a DataFrame for easier plotting
  data = pd.DataFrame({
      'Scores': np.concatenate([SC_metric, FC_metric, demos_metric, ensemble_metric]),
      'Category': ['SC'] * len(SC_metric) + ['FC'] * len(FC_metric) + ['Demographics'] * len(demos_metric) + ['Ensemble'] * len(ensemble_metric)
  })

  if metric == "accuracies":
    desc = "Accuracy"
  elif metric == "balanced_accuracies":
    desc = "Balanced Accuracy"
  else:
    desc = "ROC AUC"

  # Create a violin plot with the four arrays side-by-side
  plt.figure(figsize=(10, 6))
  sns.violinplot(x='Category', y='Scores', data=data, hue='Category', palette='muted', legend=False)
  plt.title(f'{desc} Scores for Model Type')
  plt.xlabel('Model Type')
  plt.ylabel(f'{desc} Scores')
  plt.ylim(0.45, 0.70)
  if save_fig:
    plt.savefig(f'figures/violin_plot/{file_name}/violin_plot_{metric}_{file_name}{sex}.png')
  else:
    plt.show()

if __name__ == '__main__':
  CONTROL_ONLY = False
  MALE = True
  FEMALE = False
  SAVE_FIG = True
  
  ### These only show SC, FC, ensemble (simple), and demographics (the significant models)
  create_violin_plot('accuracies', control_only=CONTROL_ONLY, male=MALE, female=FEMALE, save_fig=SAVE_FIG)
  create_violin_plot('balanced_accuracies', control_only=CONTROL_ONLY, male=MALE, female=FEMALE, save_fig=SAVE_FIG)
  create_violin_plot('roc_aucs', control_only=CONTROL_ONLY, male=MALE, female=FEMALE, save_fig=SAVE_FIG)

  ### These show all models (SC, FC, FCgsr, demographics)
  # create_violin_plot_all('accuracies', control_only=CONTROL_ONLY, male=MALE, female=FEMALE, save_fig=SAVE_FIG)
  # create_violin_plot_all('balanced_accuracies', control_only=CONTROL_ONLY, male=MALE, female=FEMALE, save_fig=SAVE_FIG)
  # create_violin_plot_all('roc_aucs', control_only=CONTROL_ONLY, male=MALE, female=FEMALE, save_fig=SAVE_FIG)

  ### These show ensemble and simple ensemble models
  # create_violin_plot_ensemble(simple=False, control_only=CONTROL_ONLY, male=MALE, female=FEMALE, save_fig=SAVE_FIG)
  # create_violin_plot_ensemble(simple=True, control_only=CONTROL_ONLY, male=MALE, female=FEMALE, save_fig=SAVE_FIG)