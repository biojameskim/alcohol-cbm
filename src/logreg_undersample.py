"""
Name: logreg_combined.py
Purpose: Perform logistic regression on connectivity matrices (SC, FC, FCgsr), demographics, and ensemble model.
      - Load and standardize the data
      - Create a logistic regression model and metrics
      - Save the results
      - Create a report
"""

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import time
from sklearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler


def scale_data(X_train, X_test):
  """
  [scale_data] scales the input data [X] using the StandardScaler.
  Uses the parameters from train data to standardize both train and test. 
  """
  scaler = StandardScaler()
  X_scaled_train = scaler.fit_transform(X_train)
  X_scaled_test = scaler.transform(X_test)

  return X_scaled_train, X_scaled_test

def save_results(models, metrics, ensemble_metrics, simple_ensemble_metrics, control_only, male, female, permute, undersample):
  """
  [save_results] saves the resulting metrics from the logistic regression to NumPy files.
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

  # Add undersample indicator to file names
  sampling = ""
  if undersample:
    sampling = f"_undersampled"

  if permute:
    for model_type in models:
      np.save(f'results/{model_type}/permuted_logreg_{model_type}_{file_name}{sex}{sampling}_balanced_accuracies.npy', metrics[model_type]['balanced_accuracies'])
      np.save(f'results/{model_type}/permuted_logreg_{model_type}_{file_name}{sex}{sampling}_roc_aucs.npy', metrics[model_type]['roc_aucs'])
      np.save(f'results/{model_type}/permuted_logreg_{model_type}_{file_name}{sex}{sampling}_pr_aucs.npy', metrics[model_type]['pr_aucs'])
    
    np.save(f'results/ensemble/permuted_logreg_ensemble_{file_name}{sex}{sampling}_balanced_accuracies.npy', ensemble_metrics['balanced_accuracies'])
    np.save(f'results/ensemble/permuted_logreg_ensemble_{file_name}{sex}{sampling}_roc_aucs.npy', ensemble_metrics['roc_aucs'])
    np.save(f'results/ensemble/permuted_logreg_ensemble_{file_name}{sex}{sampling}_pr_aucs.npy', ensemble_metrics['pr_aucs'])

    np.save(f'results/simple_ensemble/permuted_logreg_simple_ensemble_{file_name}{sex}{sampling}_balanced_accuracies.npy', simple_ensemble_metrics['balanced_accuracies'])
    np.save(f'results/simple_ensemble/permuted_logreg_simple_ensemble_{file_name}{sex}{sampling}_roc_aucs.npy', simple_ensemble_metrics['roc_aucs'])
    np.save(f'results/simple_ensemble/permuted_logreg_simple_ensemble_{file_name}{sex}{sampling}_pr_aucs.npy', simple_ensemble_metrics['pr_aucs'])

  else:
    for model_type in models:
        np.save(f'results/{model_type}/logreg_{model_type}_{file_name}{sex}{sampling}_accuracies.npy', metrics[model_type]['accuracies'])
        np.save(f'results/{model_type}/logreg_{model_type}_{file_name}{sex}{sampling}_balanced_accuracies.npy', metrics[model_type]['balanced_accuracies'])
        np.save(f'results/{model_type}/logreg_{model_type}_{file_name}{sex}{sampling}_roc_aucs.npy', metrics[model_type]['roc_aucs'])
        np.save(f'results/{model_type}/logreg_{model_type}_{file_name}{sex}{sampling}_pr_aucs.npy', metrics[model_type]['pr_aucs'])
        np.save(f'results/{model_type}/logreg_{model_type}_{file_name}{sex}{sampling}_true_labels.npy', metrics[model_type]['all_true_labels'])
        np.save(f'results/{model_type}/logreg_{model_type}_{file_name}{sex}{sampling}_pred_probs.npy', metrics[model_type]['all_pred_probs'])
        np.save(f'results/{model_type}/logreg_{model_type}_{file_name}{sex}{sampling}_coefficients.npy', metrics[model_type]['coefficients'])
    
    np.save(f'results/ensemble/logreg_ensemble_{file_name}{sex}{sampling}_accuracies.npy', ensemble_metrics['accuracies'])
    np.save(f'results/ensemble/logreg_ensemble_{file_name}{sex}{sampling}_balanced_accuracies.npy', ensemble_metrics['balanced_accuracies'])
    np.save(f'results/ensemble/logreg_ensemble_{file_name}{sex}{sampling}_roc_aucs.npy', ensemble_metrics['roc_aucs'])
    np.save(f'results/ensemble/logreg_ensemble_{file_name}{sex}{sampling}_pr_aucs.npy', ensemble_metrics['pr_aucs'])
    np.save(f'results/ensemble/logreg_ensemble_{file_name}{sex}{sampling}_true_labels.npy', ensemble_metrics['all_true_labels'])
    np.save(f'results/ensemble/logreg_ensemble_{file_name}{sex}{sampling}_pred_probs.npy', ensemble_metrics['all_pred_probs'])
    np.save(f'results/ensemble/logreg_ensemble_{file_name}{sex}{sampling}_coefficients.npy', ensemble_metrics['coefficients'])

    np.save(f'results/simple_ensemble/logreg_simple_ensemble_{file_name}{sex}{sampling}_accuracies.npy', simple_ensemble_metrics['accuracies'])
    np.save(f'results/simple_ensemble/logreg_simple_ensemble_{file_name}{sex}{sampling}_balanced_accuracies.npy', simple_ensemble_metrics['balanced_accuracies'])
    np.save(f'results/simple_ensemble/logreg_simple_ensemble_{file_name}{sex}{sampling}_roc_aucs.npy', simple_ensemble_metrics['roc_aucs'])
    np.save(f'results/simple_ensemble/logreg_simple_ensemble_{file_name}{sex}{sampling}_pr_aucs.npy', simple_ensemble_metrics['pr_aucs'])
    np.save(f'results/simple_ensemble/logreg_simple_ensemble_{file_name}{sex}{sampling}_true_labels.npy', simple_ensemble_metrics['all_true_labels'])
    np.save(f'results/simple_ensemble/logreg_simple_ensemble_{file_name}{sex}{sampling}_pred_probs.npy', simple_ensemble_metrics['all_pred_probs'])

def create_metrics_report(metrics, ensemble_metrics, simple_ensemble_metrics, num_splits, num_repeats, control_only, male, female, permute, undersample, save_to_file=False):
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

  # Add undersample indicator to file names
  undersample_str = ""
  if undersample:
    undersample_str = f"Using undersampling to achieve class balance"

  report_lines = [
    "Results for logistic regression on matrices and ensemble:\n",
    f"Number of Splits: {num_splits}",
    f"Number of Repeats: {num_repeats}\n",
    
    f"{control_str}",
    f"{sex_str}",
    f"{undersample_str}\n",

    "Shape of matrices:\n",
    f"SC: {metrics['SC']['coefficients'].shape}",
    f"FC: {metrics['FC']['coefficients'].shape}",
    # f"FCgsr: {metrics['FCgsr']['coefficients'].shape}",
    f"Demographics: {metrics['demos']['coefficients'].shape}\n",

    "Median metrics:\n",
    f"SC Median accuracy: {np.median(metrics['SC']['accuracies'])}",
    f"FC Median accuracy: {np.median(metrics['FC']['accuracies'])}",
    # f"FCgsr Median accuracy: {np.median(metrics['FCgsr']['accuracies'])}",
    f"Demographics Median accuracy: {np.median(metrics['demos']['accuracies'])}",
    f"Ensemble Median accuracy: {np.median(ensemble_metrics['accuracies'])}",
    f"Simple Ensemble Median accuracy: {np.median(simple_ensemble_metrics['accuracies'])}\n",

    f"SC Median balanced accuracy: {np.median(metrics['SC']['balanced_accuracies'])}",
    f"FC Median balanced accuracy: {np.median(metrics['FC']['balanced_accuracies'])}",
    # f"FCgsr Median balanced accuracy: {np.median(metrics['FCgsr']['balanced_accuracies'])}",
    f"Demographics Median balanced accuracy: {np.median(metrics['demos']['balanced_accuracies'])}",
    f"Ensemble Median balanced accuracy: {np.median(ensemble_metrics['balanced_accuracies'])}",
    f"Simple Ensemble Median balanced accuracy: {np.median(simple_ensemble_metrics['balanced_accuracies'])}\n",

    f"SC Median ROC AUC: {np.median(metrics['SC']['roc_aucs'])}",
    f"FC Median ROC AUC: {np.median(metrics['FC']['roc_aucs'])}",
    # f"FCgsr Median ROC AUC: {np.median(metrics['FCgsr']['roc_aucs'])}",
    f"Demographics Median ROC AUC: {np.median(metrics['demos']['roc_aucs'])}",
    f"Ensemble Median ROC AUC: {np.median(ensemble_metrics['roc_aucs'])}",
    f"Simple Ensemble Median ROC AUC: {np.median(simple_ensemble_metrics['roc_aucs'])}\n",

    f"SC Median PR AUC: {np.median(metrics['SC']['pr_aucs'])}",
    f"FC Median PR AUC: {np.median(metrics['FC']['pr_aucs'])}",
    # f"FCgsr Median PR AUC: {np.median(metrics['FCgsr']['pr_aucs'])}",
    f"Demographics Median PR AUC: {np.median(metrics['demos']['pr_aucs'])}",
    f"Ensemble Median PR AUC: {np.median(ensemble_metrics['pr_aucs'])}",
    f"Simple Ensemble Median PR AUC: {np.median(simple_ensemble_metrics['pr_aucs'])}\n",

    "Mean metrics:\n",
    "SC:",
    f"Mean accuracy: {np.mean(metrics['SC']['accuracies'])}",
    f"Std accuracy: {np.std(metrics['SC']['accuracies'])}",
    f"Mean balanced accuracy: {np.mean(metrics['SC']['balanced_accuracies'])}",
    f"Std balanced accuracy: {np.std(metrics['SC']['balanced_accuracies'])}",
    f"Mean ROC AUC: {np.mean(metrics['SC']['roc_aucs'])}",
    f"Std ROC AUC: {np.std(metrics['SC']['roc_aucs'])}",
    f"Mean PR AUC: {np.mean(metrics['SC']['pr_aucs'])}",
    f"Std PR AUC: {np.std(metrics['SC']['pr_aucs'])}\n",

    "FC:",
    f"Mean accuracy: {np.mean(metrics['FC']['accuracies'])}",
    f"Std accuracy: {np.std(metrics['FC']['accuracies'])}",
    f"Mean balanced accuracy: {np.mean(metrics['FC']['balanced_accuracies'])}",
    f"Std balanced accuracy: {np.std(metrics['FC']['balanced_accuracies'])}",
    f"Mean ROC AUC: {np.mean(metrics['FC']['roc_aucs'])}",
    f"Std ROC AUC: {np.std(metrics['FC']['roc_aucs'])}",
    f"Mean PR AUC: {np.mean(metrics['FC']['pr_aucs'])}",
    f"Std PR AUC: {np.std(metrics['FC']['pr_aucs'])}\n",

    # "FCgsr:",
    # f"Mean accuracy: {np.mean(metrics['FCgsr']['accuracies'])}",
    # f"Std accuracy: {np.std(metrics['FCgsr']['accuracies'])}",
    # f"Mean balanced accuracy: {np.mean(metrics['FCgsr']['balanced_accuracies'])}",
    # f"Std balanced accuracy: {np.std(metrics['FCgsr']['balanced_accuracies'])}",
    # f"Mean ROC AUC: {np.mean(metrics['FCgsr']['roc_aucs'])}",
    # f"Std ROC AUC: {np.std(metrics['FCgsr']['roc_aucs'])}",
    # f"Mean PR AUC: {np.mean(metrics['FCgsr']['pr_aucs'])}",
    # f"Std PR AUC: {np.std(metrics['FCgsr']['pr_aucs'])}\n",

    "Demographics:",
    f"Mean accuracy: {np.mean(metrics['demos']['accuracies'])}",
    f"Std accuracy: {np.std(metrics['demos']['accuracies'])}",
    f"Mean balanced accuracy: {np.mean(metrics['demos']['balanced_accuracies'])}",
    f"Std balanced accuracy: {np.std(metrics['demos']['balanced_accuracies'])}",
    f"Mean ROC AUC: {np.mean(metrics['demos']['roc_aucs'])}",
    f"Std ROC AUC: {np.std(metrics['demos']['roc_aucs'])}",
    f"Mean PR AUC: {np.mean(metrics['demos']['pr_aucs'])}",
    f"Std PR AUC: {np.std(metrics['demos']['pr_aucs'])}\n",

    "Ensemble:",
    f"Mean accuracy: {np.mean(ensemble_metrics['accuracies'])}",
    f"Std accuracy: {np.std(ensemble_metrics['accuracies'])}",
    f"Mean balanced accuracy: {np.mean(ensemble_metrics['balanced_accuracies'])}",
    f"Std balanced accuracy: {np.std(ensemble_metrics['balanced_accuracies'])}",
    f"Mean ROC AUC: {np.mean(ensemble_metrics['roc_aucs'])}",
    f"Std ROC AUC: {np.std(ensemble_metrics['roc_aucs'])}",
    f"Mean PR AUC: {np.mean(ensemble_metrics['pr_aucs'])}",
    f"Std PR AUC: {np.std(ensemble_metrics['pr_aucs'])}\n",

    "Simple Ensemble:",
    f"Mean accuracy: {np.mean(simple_ensemble_metrics['accuracies'])}",
    f"Std accuracy: {np.std(simple_ensemble_metrics['accuracies'])}",
    f"Mean balanced accuracy: {np.mean(simple_ensemble_metrics['balanced_accuracies'])}",
    f"Std balanced accuracy: {np.std(simple_ensemble_metrics['balanced_accuracies'])}",
    f"Mean ROC AUC: {np.mean(simple_ensemble_metrics['roc_aucs'])}",
    f"Std ROC AUC: {np.std(simple_ensemble_metrics['roc_aucs'])}",
    f"Mean PR AUC: {np.mean(simple_ensemble_metrics['pr_aucs'])}",
    f"Std PR AUC: {np.std(simple_ensemble_metrics['pr_aucs'])}"
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

  # Add undersampling indicator to file names
  sampling = ""
  if undersample:
    sampling = f"_undersampled"  

  if save_to_file and permute:
    with open(f'results/reports/permutation_test/permuted_logreg_metrics_report_{file_name}{sex}{sampling}.txt', 'w') as report_file:
      report_file.write("\n".join(report_lines))
  elif save_to_file and not permute:
    with open(f'results/reports/logreg_metrics/logreg_metrics_report_{file_name}{sex}{sampling}.txt', 'w') as report_file:
      report_file.write("\n".join(report_lines))
  else:
    print("\n".join(report_lines))

def create_model_and_metrics(X_dict, y, site_data, num_splits, num_repeats, random_ints, permute, undersample=False):
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
                      "pr_aucs": np.empty(num_repeats), 
                      "coefficients": np.empty((num_repeats*num_splits, X_dict[model].shape[1])),
                      "all_true_labels": [],
                      "all_pred_probs": []}
    
  ensemble_metrics = {"accuracies": np.empty(num_repeats),
                      "balanced_accuracies": np.empty(num_repeats), 
                      "roc_aucs": np.empty(num_repeats),
                      "pr_aucs": np.empty(num_repeats), 
                      "coefficients": np.empty((num_repeats*num_splits, len(MODELS))),
                      "all_true_labels": [],
                      "all_pred_probs": []} 
  
  simple_ensemble_metrics = {"accuracies": np.empty(num_repeats),
                              "balanced_accuracies": np.empty(num_repeats), 
                              "roc_aucs": np.empty(num_repeats),
                              "pr_aucs": np.empty(num_repeats), 
                              "coefficients": None, # No coefficients for simple ensemble
                              "all_true_labels": [],
                              "all_pred_probs": []}

  C_values = np.logspace(-4, 4, 15)
  outer_loop_metrics = {}

  for repeat_idx in range(num_repeats):
    outer_kf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=random_ints[repeat_idx])

    for model_type in MODELS:
      outer_loop_metrics[model_type] = {"accuracies": np.empty(num_splits),
                                        "balanced_accuracies": np.empty(num_splits),
                                        "roc_aucs": np.empty(num_splits),
                                        "pr_aucs": np.empty(num_splits),
                                        "coefs": np.empty((num_splits, X_dict[model_type].shape[1]))}
    
    ensemble_outer_loop_metrics = {"accuracies": np.empty(num_splits),
                                    "balanced_accuracies": np.empty(num_splits),
                                    "roc_aucs": np.empty(num_splits),
                                    "pr_aucs": np.empty(num_splits),
                                    "coefs": np.empty((num_splits, len(MODELS)))}
    
    simple_ensemble_outer_loop_metrics = {"accuracies": np.empty(num_splits),
                                        "balanced_accuracies": np.empty(num_splits),
                                        "roc_aucs": np.empty(num_splits),
                                        "pr_aucs": np.empty(num_splits)}

    if permute:
      y = np.random.permutation(y)

    n_samples = len(y) # StratifiedKFold stratifies on y so X doesn't matter in split (hence np.zeros(n_samples) below)
    simple_site_data = np.argmax(site_data, axis=1) # reduce complexity by reducing site data to a single column
    stratification_key = [str(a) + '_' + str(b) for a, b in zip(y, simple_site_data)]

    for fold_idx, (train_index, test_index) in enumerate(outer_kf.split(np.zeros(n_samples), stratification_key)):
      base_models = [] # Store the base models for the ensemble model in this fold

      # Track original distribution of classes and sites for reporting
      if fold_idx == 0 and repeat_idx == 0:
        class0_count = np.sum(y[train_index] == 0)
        class1_count = np.sum(y[train_index] == 1)
        site_counts = np.bincount(np.argmax(site_data[train_index], axis=1))
        print(f"Original training data: Class 0: {class0_count}, Class 1: {class1_count}")
        print(f"Original site distribution: {site_counts}")

      # If undersampling, do it ONCE for ALL models to ensure consistency
      if undersample:
          # Get any feature matrix for initial undersampling
          X_temp = X_dict[MODELS[0]][train_index]
          y_train_original = y[train_index]
          site_train = site_data[train_index]
          train_site_ids = np.argmax(site_train, axis=1)
          
          # Create a feature matrix with sample indices as the first column
          # This trick allows us to track which samples are selected
          sample_indices = np.arange(len(y_train_original)).reshape(-1, 1)
          temp_X = np.hstack((sample_indices, X_temp, train_site_ids.reshape(-1, 1)))
          
          # Apply stratified undersampling
          rus = RandomUnderSampler(sampling_strategy='auto', random_state=random_ints[repeat_idx + fold_idx])
          temp_X_resampled, y_train_resampled = rus.fit_resample(temp_X, y_train_original)
          
          # Extract the indices of the samples that were kept
          resampled_indices = temp_X_resampled[:, 0].astype(int)
          
          # For reporting
          resampled_sites = train_site_ids[resampled_indices]
          
          if fold_idx == 0 and repeat_idx == 0:
              class0_count_after = np.sum(y_train_resampled == 0)
              class1_count_after = np.sum(y_train_resampled == 1)
              site_counts_after = np.bincount(resampled_sites)
              print(f"Resampled training data: Class 0: {class0_count_after}, Class 1: {class1_count_after}")
              print(f"Resampled site distribution: {site_counts_after}")

      for model_type in MODELS:
        X_train_original = X_dict[model_type][train_index]
        X_test = X_dict[model_type][test_index]
        y_train_original = y[train_index]
        y_test = y[test_index]
        
        # Use the resampled indices if undersampling
        if undersample:
            X_train = X_train_original[resampled_indices]
            y_train = y_train_original[resampled_indices]
        else:
            X_train = X_train_original
            y_train = y_train_original

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
        metrics[model_type]['all_true_labels'].extend(y_test)
        metrics[model_type]['all_pred_probs'].extend(y_prob)

        accuracy = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)

        outer_loop_metrics[model_type]['accuracies'][fold_idx] = accuracy
        outer_loop_metrics[model_type]['balanced_accuracies'][fold_idx] = bal_acc
        outer_loop_metrics[model_type]['roc_aucs'][fold_idx] = roc_auc
        outer_loop_metrics[model_type]['pr_aucs'][fold_idx] = pr_auc
        outer_loop_metrics[model_type]['coefs'][fold_idx] = pipeline.named_steps['classifier'].coef_[0] # Save the coefficients for this 5-fold split

      # Ensemble model START
      # For the ensemble model, use the same resampled indices
      if undersample:
          # We already have the resampled indices, use them to create training data for each model
          ensemble_X_trains = {model_type: X_dict[model_type][train_index][resampled_indices] for model_type in MODELS}
          ensemble_y_train = y[train_index][resampled_indices]
      else:
          # If not undersampling, use original data
          ensemble_X_trains = {model_type: X_dict[model_type][train_index] for model_type in MODELS}
          ensemble_y_train = y[train_index]
  
      # Generate out-of-fold predictions for training the meta-model
      train_preds = np.column_stack([
          cross_val_predict(
              Pipeline([
                  ('scaler', StandardScaler()),
                  ('classifier', LogisticRegressionCV(penalty='l2', Cs=C_values, cv=inner_kf, max_iter=100, n_jobs=-1))
              ]),
              ensemble_X_trains[model_type], ensemble_y_train,
              method='predict_proba', cv=inner_kf, n_jobs=-1
          )[:, 1]
          for model_type in MODELS
      ])
  
      # Train meta-model on unbiased OOF predictions
      inner_kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=random_ints[repeat_idx + fold_idx])
      ensemble_model = LogisticRegressionCV(penalty='l2', Cs=C_values, cv=inner_kf, max_iter=100, n_jobs=-1)
      ensemble_model.fit(train_preds, ensemble_y_train)

      # Use previously trained base models to generate test predictions
      test_preds = np.column_stack([
          base_models[i].predict_proba(X_dict[model_type][test_index])[:, 1]
          for i, model_type in enumerate(MODELS)
      ])

      # Generate predictions using meta-model
      y_pred = ensemble_model.predict(test_preds)
      y_prob = ensemble_model.predict_proba(test_preds)[:, 1]  # Probability of class 1

      ensemble_metrics['all_true_labels'].extend(y_test)
      ensemble_metrics['all_pred_probs'].extend(y_prob)

      accuracy = accuracy_score(y_test, y_pred)
      bal_acc = balanced_accuracy_score(y_test, y_pred)
      roc_auc = roc_auc_score(y_test, y_prob)
      pr_auc = average_precision_score(y_test, y_prob)

      ensemble_outer_loop_metrics['accuracies'][fold_idx] = accuracy
      ensemble_outer_loop_metrics['balanced_accuracies'][fold_idx] = bal_acc
      ensemble_outer_loop_metrics['roc_aucs'][fold_idx] = roc_auc
      ensemble_outer_loop_metrics['pr_aucs'][fold_idx] = pr_auc
      ensemble_outer_loop_metrics['coefs'][fold_idx] = ensemble_model.coef_[0]
      # Ensemble model END

      # Simple ensemble model START
      # Average the test predictions
      averaged_preds = np.mean([
        base_models[i].predict_proba(X_dict[model_type][test_index])[:, 1]
        for i, model_type in enumerate(MODELS)
      ], axis=0)
      y_pred = (averaged_preds >= 0.5).astype(int) # Calculate binary predictions based on averaged probabilities
      y_prob = averaged_preds

      simple_ensemble_metrics['all_true_labels'].extend(y_test)
      simple_ensemble_metrics['all_pred_probs'].extend(y_prob)

      accuracy = accuracy_score(y_test, y_pred)
      bal_acc = balanced_accuracy_score(y_test, y_pred)
      roc_auc = roc_auc_score(y_test, y_prob)
      pr_auc = average_precision_score(y_test, y_prob)

      simple_ensemble_outer_loop_metrics['accuracies'][fold_idx] = accuracy
      simple_ensemble_outer_loop_metrics['balanced_accuracies'][fold_idx] = bal_acc
      simple_ensemble_outer_loop_metrics['roc_aucs'][fold_idx] = roc_auc
      simple_ensemble_outer_loop_metrics['pr_aucs'][fold_idx] = pr_auc
      # Simple ensemble model END

    # Save (average) metrics for individual models for this repeat 
    for model_type in MODELS:
      metrics[model_type]['accuracies'][repeat_idx] = np.mean(outer_loop_metrics[model_type]['accuracies'])
      metrics[model_type]['balanced_accuracies'][repeat_idx] = np.mean(outer_loop_metrics[model_type]['balanced_accuracies'])
      metrics[model_type]['roc_aucs'][repeat_idx] = np.mean(outer_loop_metrics[model_type]['roc_aucs'])
      metrics[model_type]['pr_aucs'][repeat_idx] = np.mean(outer_loop_metrics[model_type]['pr_aucs'])
      metrics[model_type]['coefficients'][(repeat_idx*num_splits):(repeat_idx*num_splits)+num_splits, :] = outer_loop_metrics[model_type]['coefs']
    
    # Save (average) metrics for ensemble model for this repeat
    ensemble_metrics['accuracies'][repeat_idx] = np.mean(ensemble_outer_loop_metrics['accuracies'])
    ensemble_metrics['balanced_accuracies'][repeat_idx] = np.mean(ensemble_outer_loop_metrics['balanced_accuracies'])
    ensemble_metrics['roc_aucs'][repeat_idx] = np.mean(ensemble_outer_loop_metrics['roc_aucs'])
    ensemble_metrics['pr_aucs'][repeat_idx] = np.mean(ensemble_outer_loop_metrics['pr_aucs'])
    ensemble_metrics['coefficients'][(repeat_idx*num_splits):(repeat_idx*num_splits)+num_splits, :] = ensemble_outer_loop_metrics['coefs']

    # Save (average) metrics for simple ensemble model for this repeat
    simple_ensemble_metrics['accuracies'][repeat_idx] = np.mean(simple_ensemble_outer_loop_metrics['accuracies'])
    simple_ensemble_metrics['balanced_accuracies'][repeat_idx] = np.mean(simple_ensemble_outer_loop_metrics['balanced_accuracies'])
    simple_ensemble_metrics['roc_aucs'][repeat_idx] = np.mean(simple_ensemble_outer_loop_metrics['roc_aucs'])
    simple_ensemble_metrics['pr_aucs'][repeat_idx] = np.mean(simple_ensemble_outer_loop_metrics['pr_aucs'])

    if (repeat_idx + 1) % 10 == 0:
        print(f"Finished repeat {repeat_idx + 1} of {num_repeats}")
    
  return metrics, ensemble_metrics, simple_ensemble_metrics

if __name__ == "__main__":
  N_SPLITS = 5
  N_REPEATS = 100
  RANDOM_STATE = 42
  SAVE_RESULTS = True
#   MODELS = ['SC', 'FC', 'FCgsr', 'demos']
  MODELS = ['SC', 'FC', 'demos']
  PERMUTE = False # If True, permute the labels (for permutation test)
  print("Permute: ", PERMUTE)
  CONTROL_ONLY = False # Set to False to include moderate group as well
  
  # BOTH of these sex flags cannot be True at the same time
  MALE = False # If True, only include male subjects
  FEMALE = True # If True, only include female subjects

  UNDERSAMPLE = True
  print("Undersample: ", UNDERSAMPLE)

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

  # Load data (No standardization yet)
  # X_dict = {model: np.load(f'data/training_data/aligned/X_{model}_{file_name}{sex}.npy') for model in MODELS}
  X_dict = {'SC': np.load(f'data/training_data/aligned/X_SC_{file_name}{sex}.npy'),
            'FC': np.load(f'data/training_data/aligned/X_FC_{file_name}{sex}.npy'),
            # 'FCgsr': np.load(f'data/training_data/aligned/X_FCgsr_{file_name}{sex}.npy'),
            'demos': np.load(f'data/training_data/aligned/X_demos_{file_name}{sex}.npy')
            }
  y = np.load(f'data/training_data/aligned/y_aligned_{file_name}{sex}.npy')
  site_data = np.load(f'data/training_data/aligned/site_location_{file_name}{sex}.npy')
  print({model: X_dict[model].shape for model in MODELS}, "y:", y.shape)
  print("\n")

  sampling = ""
  if UNDERSAMPLE:
    sampling = f"_undersampled"

  print("Running logistic regression on matrices and ensemble (stratified)...")
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
  metrics, ensemble_metrics, simple_ensemble_metrics = create_model_and_metrics(X_dict=X_dict, y=y, site_data=site_data, num_splits=N_SPLITS, num_repeats=N_REPEATS, random_ints=random_ints, permute=PERMUTE, undersample=UNDERSAMPLE)
  end = time.time()
  print(f"Finished in {end - start} seconds\n")

  print("Saving results...")
  save_results(models=MODELS, metrics=metrics, ensemble_metrics=ensemble_metrics, simple_ensemble_metrics=simple_ensemble_metrics, control_only=CONTROL_ONLY, male=MALE, female=FEMALE, permute=PERMUTE, undersample=UNDERSAMPLE)
  print("Results saved successfully\n")

  print("Creating report...")
  create_metrics_report(metrics=metrics, ensemble_metrics=ensemble_metrics, simple_ensemble_metrics=simple_ensemble_metrics, num_splits=N_SPLITS, num_repeats=N_REPEATS, control_only=CONTROL_ONLY, male=MALE, female=FEMALE, permute=PERMUTE, undersample=UNDERSAMPLE, save_to_file=SAVE_RESULTS)
  print(f"Report created successfully at results/reports/logreg_metrics/stratified_logreg_metrics_report_{file_name}{sex}{sampling}.txt\n")