"""
Name: test_sex.py
Purpose: Perform logistic regression on connectivity matrices (SC, FC, FCgsr), demographics, and ensemble model
        for the purpose of testing the combined model on male only vs. female only test set
      - Load and standardize the data
      - Create a logistic regression model and metrics
      - Save the results
      - Create a report
"""

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict, BaseCrossValidator
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import time
from sklearn.pipeline import Pipeline

class SexStratifiedKFold(BaseCrossValidator):
    def __init__(self, n_splits=5, random_state=None):
        self.n_splits = n_splits
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits * 2  # Male + female splits

    def split(self, X, y, sex_data, stratification_key):
        """
        Parameters:
        -----------
        X : array-like
            Training data (not used, exists for compatibility)
        y : array-like
            Target values
        sex_data : array-like
            Binary indicator of sex (0=female, 1=male)
        stratification_key : array-like
            Combined label+site keys for stratification
        """
        indices = np.arange(len(y))
        male_mask = (sex_data == 1)
        male_indices = indices[male_mask]
        female_indices = indices[~male_mask]
        
        # Get male and female stratification keys
        male_strat_keys = [stratification_key[i] for i in male_indices]
        female_strat_keys = [stratification_key[i] for i in female_indices]

        rng = np.random.RandomState(self.random_state)
        
        # Use StratifiedKFold with stratification_key
        male_skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, 
                                 random_state=rng.randint(1e6))
        female_skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                                   random_state=rng.randint(1e6))

        # Male test folds
        for _, test_idx in male_skf.split(male_indices, male_strat_keys):
            test = male_indices[test_idx]
            train = np.concatenate([
                male_indices[~np.isin(male_indices, test)],
                female_indices
            ])
            yield train, test

        # Female test folds
        for _, test_idx in female_skf.split(female_indices, female_strat_keys):
            test = female_indices[test_idx]
            train = np.concatenate([
                male_indices,
                female_indices[~np.isin(female_indices, test)]
            ])
            yield train, test

def scale_data(X_train, X_test):
  """
  [scale_data] scales the input data [X] using the StandardScaler.
  Uses the parameters from train data to standardize both train and test. 
  """
  scaler = StandardScaler()
  X_scaled_train = scaler.fit_transform(X_train)
  X_scaled_test = scaler.transform(X_test)

  return X_scaled_train, X_scaled_test

def create_metrics_report(metrics, ensemble_metrics, num_splits, num_repeats, permute, save_to_file=False):
  """
  [create_metrics_report] creates a report of the metrics for all models from the logistic regression
  """

  report_lines = [
    "Sex-Specific Results for logistic regression on matrices and ensemble:\n",
    f"Number of Splits: {num_splits}",
    f"Number of Repeats: {num_repeats}\n",

    "Male Test Results:\n",

    "Median metrics:\n",
    f"SC Median accuracy: {np.median(metrics['male']['SC']['accuracies'])}",
    f"FC Median accuracy: {np.median(metrics['male']['FC']['accuracies'])}",
    f"Demographics Median accuracy: {np.median(metrics['male']['demos']['accuracies'])}",
    f"Ensemble Median accuracy: {np.median(ensemble_metrics['male']['accuracies'])}\n",

    f"SC Median balanced accuracy: {np.median(metrics['male']['SC']['balanced_accuracies'])}",
    f"FC Median balanced accuracy: {np.median(metrics['male']['FC']['balanced_accuracies'])}",
    f"Demographics Median balanced accuracy: {np.median(metrics['male']['demos']['balanced_accuracies'])}",
    f"Ensemble Median balanced accuracy: {np.median(ensemble_metrics['male']['balanced_accuracies'])}\n",

    f"SC Median ROC AUC: {np.median(metrics['male']['SC']['roc_aucs'])}",
    f"FC Median ROC AUC: {np.median(metrics['male']['FC']['roc_aucs'])}",
    f"Demographics Median ROC AUC: {np.median(metrics['male']['demos']['roc_aucs'])}",
    f"Ensemble Median ROC AUC: {np.median(ensemble_metrics['male']['roc_aucs'])}\n",

    f"SC Median PR AUC: {np.median(metrics['male']['SC']['pr_aucs'])}",
    f"FC Median PR AUC: {np.median(metrics['male']['FC']['pr_aucs'])}",
    f"Demographics Median PR AUC: {np.median(metrics['male']['demos']['pr_aucs'])}",
    f"Ensemble Median PR AUC: {np.median(ensemble_metrics['male']['pr_aucs'])}\n",

    "Mean metrics:\n",
    "SC:",
    f"Mean accuracy: {np.mean(metrics['male']['SC']['accuracies'])}",
    f"Std accuracy: {np.std(metrics['male']['SC']['accuracies'])}",
    f"Mean balanced accuracy: {np.mean(metrics['male']['SC']['balanced_accuracies'])}",
    f"Std balanced accuracy: {np.std(metrics['male']['SC']['balanced_accuracies'])}",
    f"Mean ROC AUC: {np.mean(metrics['male']['SC']['roc_aucs'])}",
    f"Std ROC AUC: {np.std(metrics['male']['SC']['roc_aucs'])}",
    f"Mean PR AUC: {np.mean(metrics['male']['SC']['pr_aucs'])}",
    f"Std PR AUC: {np.std(metrics['male']['SC']['pr_aucs'])}\n",

    "FC:",
    f"Mean accuracy: {np.mean(metrics['male']['FC']['accuracies'])}",
    f"Std accuracy: {np.std(metrics['male']['FC']['accuracies'])}",
    f"Mean balanced accuracy: {np.mean(metrics['male']['FC']['balanced_accuracies'])}",
    f"Std balanced accuracy: {np.std(metrics['male']['FC']['balanced_accuracies'])}",
    f"Mean ROC AUC: {np.mean(metrics['male']['FC']['roc_aucs'])}",
    f"Std ROC AUC: {np.std(metrics['male']['FC']['roc_aucs'])}",
    f"Mean PR AUC: {np.mean(metrics['male']['FC']['pr_aucs'])}",
    f"Std PR AUC: {np.std(metrics['male']['FC']['pr_aucs'])}\n",

    "Demographics:",
    f"Mean accuracy: {np.mean(metrics['male']['demos']['accuracies'])}",
    f"Std accuracy: {np.std(metrics['male']['demos']['accuracies'])}",
    f"Mean balanced accuracy: {np.mean(metrics['male']['demos']['balanced_accuracies'])}",
    f"Std balanced accuracy: {np.std(metrics['male']['demos']['balanced_accuracies'])}",
    f"Mean ROC AUC: {np.mean(metrics['male']['demos']['roc_aucs'])}",
    f"Std ROC AUC: {np.std(metrics['male']['demos']['roc_aucs'])}",
    f"Mean PR AUC: {np.mean(metrics['male']['demos']['pr_aucs'])}",
    f"Std PR AUC: {np.std(metrics['male']['demos']['pr_aucs'])}\n",

    "Ensemble:",
    f"Mean accuracy: {np.mean(ensemble_metrics['male']['accuracies'])}",
    f"Std accuracy: {np.std(ensemble_metrics['male']['accuracies'])}",
    f"Mean balanced accuracy: {np.mean(ensemble_metrics['male']['balanced_accuracies'])}",
    f"Std balanced accuracy: {np.std(ensemble_metrics['male']['balanced_accuracies'])}",
    f"Mean ROC AUC: {np.mean(ensemble_metrics['male']['roc_aucs'])}",
    f"Std ROC AUC: {np.std(ensemble_metrics['male']['roc_aucs'])}",
    f"Mean PR AUC: {np.mean(ensemble_metrics['male']['pr_aucs'])}",
    f"Std PR AUC: {np.std(ensemble_metrics['male']['pr_aucs'])}\n",

    "Female Test Results:\n",

    "Median metrics:\n",
    f"SC Median accuracy: {np.median(metrics['female']['SC']['accuracies'])}",
    f"FC Median accuracy: {np.median(metrics['female']['FC']['accuracies'])}",
    f"Demographics Median accuracy: {np.median(metrics['female']['demos']['accuracies'])}",
    f"Ensemble Median accuracy: {np.median(ensemble_metrics['female']['accuracies'])}\n",

    f"SC Median balanced accuracy: {np.median(metrics['female']['SC']['balanced_accuracies'])}",
    f"FC Median balanced accuracy: {np.median(metrics['female']['FC']['balanced_accuracies'])}",
    f"Demographics Median balanced accuracy: {np.median(metrics['female']['demos']['balanced_accuracies'])}",
    f"Ensemble Median balanced accuracy: {np.median(ensemble_metrics['female']['balanced_accuracies'])}\n",

    f"SC Median ROC AUC: {np.median(metrics['female']['SC']['roc_aucs'])}",
    f"FC Median ROC AUC: {np.median(metrics['female']['FC']['roc_aucs'])}",
    f"Demographics Median ROC AUC: {np.median(metrics['female']['demos']['roc_aucs'])}",
    f"Ensemble Median ROC AUC: {np.median(ensemble_metrics['female']['roc_aucs'])}\n",

    f"SC Median PR AUC: {np.median(metrics['female']['SC']['pr_aucs'])}",
    f"FC Median PR AUC: {np.median(metrics['female']['FC']['pr_aucs'])}",
    f"Demographics Median PR AUC: {np.median(metrics['female']['demos']['pr_aucs'])}",
    f"Ensemble Median PR AUC: {np.median(ensemble_metrics['female']['pr_aucs'])}\n",

    "Mean metrics:\n",
    "SC:",
    f"Mean accuracy: {np.mean(metrics['female']['SC']['accuracies'])}",
    f"Std accuracy: {np.std(metrics['female']['SC']['accuracies'])}",
    f"Mean balanced accuracy: {np.mean(metrics['female']['SC']['balanced_accuracies'])}",
    f"Std balanced accuracy: {np.std(metrics['female']['SC']['balanced_accuracies'])}",
    f"Mean ROC AUC: {np.mean(metrics['female']['SC']['roc_aucs'])}",
    f"Std ROC AUC: {np.std(metrics['female']['SC']['roc_aucs'])}",
    f"Mean PR AUC: {np.mean(metrics['female']['SC']['pr_aucs'])}",
    f"Std PR AUC: {np.std(metrics['female']['SC']['pr_aucs'])}\n",

    "FC:",
    f"Mean accuracy: {np.mean(metrics['female']['FC']['accuracies'])}",
    f"Std accuracy: {np.std(metrics['female']['FC']['accuracies'])}",
    f"Mean balanced accuracy: {np.mean(metrics['female']['FC']['balanced_accuracies'])}",
    f"Std balanced accuracy: {np.std(metrics['female']['FC']['balanced_accuracies'])}",
    f"Mean ROC AUC: {np.mean(metrics['female']['FC']['roc_aucs'])}",
    f"Std ROC AUC: {np.std(metrics['female']['FC']['roc_aucs'])}",
    f"Mean PR AUC: {np.mean(metrics['female']['FC']['pr_aucs'])}",
    f"Std PR AUC: {np.std(metrics['female']['FC']['pr_aucs'])}\n",

    "Demographics:",
    f"Mean accuracy: {np.mean(metrics['female']['demos']['accuracies'])}",
    f"Std accuracy: {np.std(metrics['female']['demos']['accuracies'])}",
    f"Mean balanced accuracy: {np.mean(metrics['female']['demos']['balanced_accuracies'])}",
    f"Std balanced accuracy: {np.std(metrics['female']['demos']['balanced_accuracies'])}",
    f"Mean ROC AUC: {np.mean(metrics['female']['demos']['roc_aucs'])}",
    f"Std ROC AUC: {np.std(metrics['female']['demos']['roc_aucs'])}",
    f"Mean PR AUC: {np.mean(metrics['female']['demos']['pr_aucs'])}",
    f"Std PR AUC: {np.std(metrics['female']['demos']['pr_aucs'])}\n",

    "Ensemble:",
    f"Mean accuracy: {np.mean(ensemble_metrics['female']['accuracies'])}",
    f"Std accuracy: {np.std(ensemble_metrics['female']['accuracies'])}",
    f"Mean balanced accuracy: {np.mean(ensemble_metrics['female']['balanced_accuracies'])}",
    f"Std balanced accuracy: {np.std(ensemble_metrics['female']['balanced_accuracies'])}",
    f"Mean ROC AUC: {np.mean(ensemble_metrics['female']['roc_aucs'])}",
    f"Std ROC AUC: {np.std(ensemble_metrics['female']['roc_aucs'])}",
    f"Mean PR AUC: {np.mean(ensemble_metrics['female']['pr_aucs'])}",
    f"Std PR AUC: {np.std(ensemble_metrics['female']['pr_aucs'])}\n",
  ]

  if save_to_file and permute:
    with open(f'results/reports/permutation_test/permuted_sex_test_logreg_metrics_report_control_moderate.txt', 'w') as report_file:
      report_file.write("\n".join(report_lines))
  elif save_to_file and not permute:
    with open(f'results/reports/logreg_metrics/sex_test_logreg_metrics_report_control_moderate.txt', 'w') as report_file:
      report_file.write("\n".join(report_lines))
  else:
    print("\n".join(report_lines))

def create_model_and_metrics(X_dict, y, site_data, num_splits, num_repeats, random_ints, permute):
  """
  [create_model_and_metrics] performs logistic regression on the matrices and ensemble model.
  Returns the metrics for each model and the ensemble model.
  """
  # Initialize metrics dictionaries
  metrics = {
    'male': {model: {"accuracies": np.empty(num_repeats), 
                      "balanced_accuracies": np.empty(num_repeats), 
                      "roc_aucs": np.empty(num_repeats), 
                      "pr_aucs": np.empty(num_repeats), 
                      "all_true_labels": [],
                      "all_pred_probs": []} 
             for model in MODELS},
    'female': {model: {"accuracies": np.empty(num_repeats), 
                      "balanced_accuracies": np.empty(num_repeats), 
                      "roc_aucs": np.empty(num_repeats), 
                      "pr_aucs": np.empty(num_repeats), 
                      "all_true_labels": [],
                      "all_pred_probs": []}
             for model in MODELS}
  }
    
  ensemble_metrics = {
    'male': {"accuracies": np.empty(num_repeats),
            "balanced_accuracies": np.empty(num_repeats), 
            "roc_aucs": np.empty(num_repeats),
            "pr_aucs": np.empty(num_repeats), 
            "all_true_labels": [],
            "all_pred_probs": []},
    'female': {"accuracies": np.empty(num_repeats),
            "balanced_accuracies": np.empty(num_repeats), 
            "roc_aucs": np.empty(num_repeats),
            "pr_aucs": np.empty(num_repeats), 
            "all_true_labels": [],
            "all_pred_probs": []}
  }

  C_values = np.logspace(-4, 4, 15)
  outer_loop_metrics = {}

  for repeat_idx in range(num_repeats):
    outer_kf = SexStratifiedKFold(n_splits=num_splits, shuffle=True, random_state=random_ints[repeat_idx])

    for model_type in MODELS:
      outer_loop_metrics[model_type] = {"accuracies": np.empty(num_splits),
                                        "balanced_accuracies": np.empty(num_splits),
                                        "roc_aucs": np.empty(num_splits),
                                        "pr_aucs": np.empty(num_splits)}
    
    ensemble_outer_loop_metrics = {"accuracies": np.empty(num_splits),
                                    "balanced_accuracies": np.empty(num_splits),
                                    "roc_aucs": np.empty(num_splits),
                                    "pr_aucs": np.empty(num_splits)}

    if permute:
      y = np.random.permutation(y)

    n_samples = len(y) # StratifiedKFold stratifies on y so X doesn't matter in split (hence np.zeros(n_samples) below)
    simple_site_data = np.argmax(site_data, axis=1) # reduce complexity by reducing site data to a single column
    stratification_key = [str(a) + '_' + str(b) for a, b in zip(y, simple_site_data)]

    for fold_idx, (train_index, test_index) in enumerate(outer_kf.split(np.zeros(n_samples), y, sex_data, stratification_key)):
      test_sex = 'male' if sex_data[test_index[0]] == 1 else 'female'

      

      base_models = [] # Store the base models for the ensemble model in this fold
      for model_type in MODELS:
        X_train, X_test = X_dict[model_type][train_index], X_dict[model_type][test_index]
        y_train, y_test = y[train_index], y[test_index]

        inner_kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=random_ints[repeat_idx + fold_idx])
        
        # Create a pipeline that includes standardization
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegressionCV(penalty='l2', Cs=C_values, cv=inner_kf, max_iter=100, n_jobs=-1))
        ])

        pipeline.fit(X_train, y_train)
        base_models.append(pipeline)

        # Get predictions from the pipeline
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        # Collect true labels and predicted probabilities for ROC curve
        metrics[test_sex][model_type]['all_true_labels'].extend(y_test)
        metrics[test_sex][model_type]['all_pred_probs'].extend(y_prob)

        accuracy = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)

        outer_loop_metrics[model_type]['accuracies'][fold_idx] = accuracy
        outer_loop_metrics[model_type]['balanced_accuracies'][fold_idx] = bal_acc
        outer_loop_metrics[model_type]['roc_aucs'][fold_idx] = roc_auc
        outer_loop_metrics[model_type]['pr_aucs'][fold_idx] = pr_auc

      # Ensemble model START
      # Modified Ensemble model implementation with out-of-fold predictions
      # Generate out-of-fold predictions for training the meta-model
      train_preds = np.column_stack([
          cross_val_predict(
              Pipeline([
                  ('scaler', StandardScaler()),
                  ('classifier', LogisticRegressionCV(penalty='l2', Cs=C_values, cv=inner_kf, max_iter=100, n_jobs=-1))
              ]),
              X_dict[model_type][train_index], y_train,
              method='predict_proba', cv=inner_kf, n_jobs=-1
          )[:, 1]
          for model_type in MODELS
      ])

      # Use previously trained base models to generate test predictions
      test_preds = np.column_stack([
          base_models[i].predict_proba(X_dict[model_type][test_index])[:, 1]
          for i, model_type in enumerate(MODELS)
      ])
      # Train meta-model on unbiased OOF predictions
      inner_kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=random_ints[repeat_idx + fold_idx])
      ensemble_model = LogisticRegressionCV(penalty='l2', Cs=C_values, cv=inner_kf, max_iter=100, n_jobs=-1)
      ensemble_model.fit(train_preds, y_train)

      # Generate predictions using meta-model
      y_pred = ensemble_model.predict(test_preds)
      y_prob = ensemble_model.predict_proba(test_preds)[:, 1]  # Probability of class 1

      ensemble_metrics[test_sex]['all_true_labels'].extend(y_test)
      ensemble_metrics[test_sex]['all_pred_probs'].extend(y_prob)

      accuracy = accuracy_score(y_test, y_pred)
      bal_acc = balanced_accuracy_score(y_test, y_pred)
      roc_auc = roc_auc_score(y_test, y_prob)
      pr_auc = average_precision_score(y_test, y_prob)

      ensemble_outer_loop_metrics['accuracies'][fold_idx] = accuracy
      ensemble_outer_loop_metrics['balanced_accuracies'][fold_idx] = bal_acc
      ensemble_outer_loop_metrics['roc_aucs'][fold_idx] = roc_auc
      ensemble_outer_loop_metrics['pr_aucs'][fold_idx] = pr_auc
      # Ensemble model END

    # Save (average) metrics for individual models for this repeat 
    for model_type in MODELS:
      metrics[test_sex][model_type]['accuracies'][repeat_idx] = np.mean(outer_loop_metrics[model_type]['accuracies'])
      metrics[test_sex][model_type]['balanced_accuracies'][repeat_idx] = np.mean(outer_loop_metrics[model_type]['balanced_accuracies'])
      metrics[test_sex][model_type]['roc_aucs'][repeat_idx] = np.mean(outer_loop_metrics[model_type]['roc_aucs'])
      metrics[test_sex][model_type]['pr_aucs'][repeat_idx] = np.mean(outer_loop_metrics[model_type]['pr_aucs'])
    
    # Save (average) metrics for ensemble model for this repeat
    ensemble_metrics[test_sex]['accuracies'][repeat_idx] = np.mean(ensemble_outer_loop_metrics['accuracies'])
    ensemble_metrics[test_sex]['balanced_accuracies'][repeat_idx] = np.mean(ensemble_outer_loop_metrics['balanced_accuracies'])
    ensemble_metrics[test_sex]['roc_aucs'][repeat_idx] = np.mean(ensemble_outer_loop_metrics['roc_aucs'])
    ensemble_metrics[test_sex]['pr_aucs'][repeat_idx] = np.mean(ensemble_outer_loop_metrics['pr_aucs'])

    if (repeat_idx + 1) % 10 == 0:
        print(f"Finished repeat {repeat_idx + 1} of {num_repeats}")
    
  return metrics, ensemble_metrics

if __name__ == "__main__":
  N_SPLITS = 5
  N_REPEATS = 100
  RANDOM_STATE = 42
  SAVE_RESULTS = True
  MODELS = ['SC', 'FC', 'demos'] # Can add FCgsr too
  PERMUTE = True # If True, permute the labels (for permutation test)
  print("Permute: ", PERMUTE)

  # Set random seed for reproducibility
  np.random.seed(RANDOM_STATE) 
  random_ints = np.random.randint(0, 1000, N_REPEATS + N_SPLITS) 

  # Load data (No standardization yet)
  # X_dict = {model: np.load(f'data/training_data/aligned/X_{model}_control_moderate.npy') for model in MODELS}
  X_dict = {'SC': np.load(f'data/training_data/aligned/X_SC_control_moderate.npy'),
            'FC': np.load(f'data/training_data/aligned/X_FC_control_moderate.npy'),
            # 'FCgsr': np.load(f'data/training_data/aligned/X_FCgsr_control_moderate.npy'),
            'demos': np.load(f'data/training_data/aligned/X_demos_control_moderate.npy')
            }
  y = np.load(f'data/training_data/aligned/y_aligned_control_moderate.npy')
  site_data = np.load(f'data/training_data/aligned/site_location_control_moderate.npy')
  sex_data = X_dict['demos'][:, 3].astype(int) # Sex is the 4th column in demos data (0=female, 1=male)


  print({model: X_dict[model].shape for model in MODELS}, "y:", y.shape)
  print("\n")

  print("Running logistic regression on matrices and ensemble (stratified)...")
  print("Baseline subjects with cahalan=='control' or cahalan=='moderate'")
  print("All subjects")
  print("\n")

  start = time.time()
  metrics, ensemble_metrics = create_model_and_metrics(X_dict=X_dict, y=y, site_data=site_data, num_splits=N_SPLITS, num_repeats=N_REPEATS, random_ints=random_ints, permute=PERMUTE)
  end = time.time()
  print(f"Finished in {end - start} seconds\n")

  print("Creating report...")
  create_metrics_report(metrics=metrics, ensemble_metrics=ensemble_metrics, num_splits=N_SPLITS, num_repeats=N_REPEATS, permute=PERMUTE, save_to_file=SAVE_RESULTS)
  print(f"Report created successfully at results/reports/logreg_metrics/sex_test_logreg_metrics_report_control_moderate.txt\n")