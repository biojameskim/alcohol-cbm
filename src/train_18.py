"""
Name: logreg_combined.py
Purpose: Perform logistic regression on connectivity matrices (SC, FC, FCgsr), demographics models.
      - Load and standardize the data
      - Create a logistic regression model and metrics
      - Save the results
      - Create a report
"""

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import time

def scale_data(X):
  """
  [scale_data] scales the input data [X] using the StandardScaler.
  """
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)
  return X_scaled


def create_metrics_report(metrics, num_splits, num_repeats, control_only, male, female, save_to_file=False):
  """
  [create_metrics_report] creates a report of the metrics for all models from the logistic regression
  """
  if control_only:
    control_str = "Baseline subjects with cahalan=='control' only"
  else:
    control_str = "Baseline subjects with cahalan=='control' or cahalan=='moderate'"

  if male:
    sex_str = "Male subjects only"
  elif female:
    sex_str = "Female subjects only"
  else:
    sex_str = "All subjects"

  report_lines = [
    "Results for logistic regression on matrices:\n",
    f"Number of Splits: {num_splits}",
    f"Number of Repeats: {num_repeats}\n",
    
    f"{control_str}",
    f"{sex_str}\n",

    "SC:",
    f"Mean accuracy: {np.mean(metrics['SC']['accuracies'])}",
    f"Std accuracy: {np.std(metrics['SC']['accuracies'])}",
    f"Mean balanced accuracy: {np.mean(metrics['SC']['balanced_accuracies'])}",
    f"Std balanced accuracy: {np.std(metrics['SC']['balanced_accuracies'])}",
    f"Mean ROC AUC: {np.mean(metrics['SC']['roc_aucs'])}",
    f"Std ROC AUC: {np.std(metrics['SC']['roc_aucs'])}\n",

    "FC:",
    f"Mean accuracy: {np.mean(metrics['FC']['accuracies'])}",
    f"Std accuracy: {np.std(metrics['FC']['accuracies'])}",
    f"Mean balanced accuracy: {np.mean(metrics['FC']['balanced_accuracies'])}",
    f"Std balanced accuracy: {np.std(metrics['FC']['balanced_accuracies'])}",
    f"Mean ROC AUC: {np.mean(metrics['FC']['roc_aucs'])}",
    f"Std ROC AUC: {np.std(metrics['FC']['roc_aucs'])}\n",

    "FCgsr:",
    f"Mean accuracy: {np.mean(metrics['FCgsr']['accuracies'])}",
    f"Std accuracy: {np.std(metrics['FCgsr']['accuracies'])}",
    f"Mean balanced accuracy: {np.mean(metrics['FCgsr']['balanced_accuracies'])}",
    f"Std balanced accuracy: {np.std(metrics['FCgsr']['balanced_accuracies'])}",
    f"Mean ROC AUC: {np.mean(metrics['FCgsr']['roc_aucs'])}",
    f"Std ROC AUC: {np.std(metrics['FCgsr']['roc_aucs'])}\n",

    "Demographics:",
    f"Mean accuracy: {np.mean(metrics['demos']['accuracies'])}",
    f"Std accuracy: {np.std(metrics['demos']['accuracies'])}",
    f"Mean balanced accuracy: {np.mean(metrics['demos']['balanced_accuracies'])}",
    f"Std balanced accuracy: {np.std(metrics['demos']['balanced_accuracies'])}",
    f"Mean ROC AUC: {np.mean(metrics['demos']['roc_aucs'])}",
    f"Std ROC AUC: {np.std(metrics['demos']['roc_aucs'])}\n",
  ]

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

  if save_to_file:
    with open(f'results/reports/logreg_metrics/logreg_matrices_metrics_report_{file_name}{sex}.txt', 'w') as report_file:
      report_file.write("\n".join(report_lines))
  else:
    print("\n".join(report_lines))

def create_model_and_metrics(X_dict, y, num_splits, num_repeats, random_ints, permute):
  """
  [create_model_and_metrics] performs logistic regression on the matrices and ensemble model.
  Returns the metrics for each model and the ensemble model.
  """
  # Initialize metrics dictionaries
  metrics = {}
  for model in MODELS:
    metrics[model] = {"accuracies": np.empty(num_repeats), 
                      "balanced_accuracies": np.empty(num_repeats), 
                      "roc_aucs": np.empty(num_repeats), 
                      "all_true_labels": [],
                      "all_pred_probs": []}

  C_values = np.logspace(-4, 4, 15)
  outer_loop_metrics = {}

  for repeat_idx in range(num_repeats):
    outer_kf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=random_ints[repeat_idx])

    for model_type in MODELS:
      outer_loop_metrics[model_type] = {"accuracies": np.empty(num_splits),
                                        "balanced_accuracies": np.empty(num_splits),
                                        "roc_aucs": np.empty(num_splits)}

    if permute:
      y = np.random.permutation(y)

    n_samples = len(y) # StratifiedKFold stratifies on y so X doesn't matter in split (hence np.zeros(n_samples) below)

    for fold_idx, (train_index, test_index) in enumerate(outer_kf.split(np.zeros(n_samples), y)):
      for model_type in MODELS:  
        X_train, X_test = X_dict[model_type][train_index], X_dict[model_type][test_index]
        y_train, y_test = y[train_index], y[test_index]

        inner_kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=random_ints[repeat_idx + fold_idx])
        model = LogisticRegressionCV(penalty='l2', Cs=C_values, cv=inner_kf, max_iter=100, n_jobs=-1)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Collect true labels and predicted probabilities for ROC curve
        metrics[model_type]['all_true_labels'].extend(y_test)
        metrics[model_type]['all_pred_probs'].extend(y_prob)

        accuracy = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        outer_loop_metrics[model_type]['accuracies'][fold_idx] = accuracy
        outer_loop_metrics[model_type]['balanced_accuracies'][fold_idx] = bal_acc
        outer_loop_metrics[model_type]['roc_aucs'][fold_idx] = roc_auc

    # Save (average) metrics for individual models for this repeat 
    for model_type in MODELS:
      metrics[model_type]['accuracies'][repeat_idx] = np.mean(outer_loop_metrics[model_type]['accuracies'])
      metrics[model_type]['balanced_accuracies'][repeat_idx] = np.mean(outer_loop_metrics[model_type]['balanced_accuracies'])
      metrics[model_type]['roc_aucs'][repeat_idx] = np.mean(outer_loop_metrics[model_type]['roc_aucs'])

    if (repeat_idx + 1) % 10 == 0:
        print(f"Finished repeat {repeat_idx + 1} of {num_repeats}")
    
  return metrics

if __name__ == "__main__":
  N_SPLITS = 5
  N_REPEATS = 100
  RANDOM_STATE = 42
  SAVE_RESULTS = False
  MODELS = ['SC', 'FC', 'FCgsr', 'demos']
  PERMUTE = False # If True, permute the labels (for permutation test)
  print("Permute: ", PERMUTE)
  CONTROL_ONLY = False # Set to False to include moderate group as well
  
  # BOTH of these sex flags cannot be True at the same time
  MALE = False # If True, only include male subjects
  FEMALE = False # If True, only include female subjects

  # Set random seed for reproducibility
  np.random.seed(RANDOM_STATE) 
  random_ints = np.random.randint(0, 1000, N_REPEATS + N_SPLITS) 

  if CONTROL_ONLY:
    file_name = 'control'
  else:
    file_name = 'control_moderate'

  if MALE:
    sex = '_male'
  elif FEMALE:
    sex = '_female'
  else:
    sex = ''

  # Load and standardize data (e.g., X_dict['SC'] = X_SC_scaled)

#   X_dict = {model: scale_data(np.load(f'data/18data/before18baseline_X_{model}_{file_name}.npy')) for model in MODELS}
#   y = np.load(f'data/18data/before18baseline_y_aligned_{file_name}.npy')
#   print("group1. baseline <18. age of event >18.")

    X_dict = {model: scale_data(np.load(f'data/18data/comp_X_{model}_{file_name}.npy')) for model in MODELS}
    y = np.load(f'data/18data/comp_y_aligned_{file_name}.npy')
    print("group1. everyone else.")


  print(X_dict['SC'].shape, X_dict['FC'].shape, X_dict['FCgsr'].shape, X_dict['demos'].shape, y.shape)
  print("\n")

  print("Running logistic regression on matrices...")
  if CONTROL_ONLY:
    print("Baseline subjects with cahalan=='control' only")
  else:
    print("Baseline subjects with cahalan=='control' or cahalan=='moderate'")

  if MALE:
    print("Male subjects only")
  elif FEMALE:
    print("Female subjects only")
  else:
    print("All subjects")
  print("\n")

  start = time.time()
  metrics = create_model_and_metrics(X_dict=X_dict, y=y, num_splits=N_SPLITS, num_repeats=N_REPEATS, random_ints=random_ints, permute=PERMUTE)
  end = time.time()
  print(f"Finished in {end - start} seconds\n")

  print("Creating report...")
  create_metrics_report(metrics=metrics, num_splits=N_SPLITS, num_repeats=N_REPEATS, control_only=CONTROL_ONLY, male=MALE, female=FEMALE, save_to_file=SAVE_RESULTS)