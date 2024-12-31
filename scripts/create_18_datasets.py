#!/usr/bin/env python3

"""
Name: align_training_data.py
Purpose: Align the SC, FC, FCgsr, and demographic data based on COMMON subjects and group assignments.
        This is used for training the logistic regression models (SC, FC, FCgsr, demographics, and ensemble).
"""

import pandas as pd
import numpy as np

from process_conn_matrices import load_and_flatten_SC_matrices, load_FC_matrices_and_subject_ids, flatten_FC_matrices

# Set CONTROL_ONLY to True to only include subjects who are "control" at baseline
# Set CONTROL_ONLY to False to include subjects who are "control" or "moderate" at baseline
CONTROL_ONLY = False

# SC matrices
SC_matrices, SC_subject_ids = load_and_flatten_SC_matrices('data/tractography_subcortical (SC)')
# Create SC matrices DataFrame
SC_df = pd.DataFrame(SC_matrices)
SC_df['subject'] = SC_subject_ids

# FC matrices
baseline_FC_matrices, baseline_FCgsr_matrices, FC_subject_ids = load_FC_matrices_and_subject_ids(
path_to_FC='data/FC/NCANDA_FC.mat', 
path_to_FCgsr='data/FC/NCANDA_FCgsr.mat',
path_to_demography='data/FC/NCANDA_demos.csv'
)
FC_matrices = flatten_FC_matrices(baseline_FC_matrices)
FCgsr_matrices = flatten_FC_matrices(baseline_FCgsr_matrices)
# Create FC matrices DataFrame
FC_df = pd.DataFrame(FC_matrices)
FC_df['subject'] = FC_subject_ids
# Create FCgsr matrices DataFrame
FCgsr_df = pd.DataFrame(FCgsr_matrices)
FCgsr_df['subject'] = FC_subject_ids

# Save the matrices with subject ids
# pd.to_csv = SC_df.to_csv('data/data_with_subject_ids/SC_matrices_with_subjects.csv', index=False)
# pd.to_csv = FC_df.to_csv('data/data_with_subject_ids/FC_matrices_with_subjects.csv', index=False)
# pd.to_csv = FCgsr_df.to_csv('data/data_with_subject_ids/FCgsr_matrices_with_subjects.csv', index=False)

# Demographic data
if CONTROL_ONLY:
        demos_df = pd.read_csv('data/data_with_subject_ids/control_demos_with_subjects.csv')
else:
        demos_df = pd.read_csv('data/data_with_subject_ids/control_moderate_demos_with_subjects.csv')

# Identify common subjects
# group_subjects = np.load('data/subjects_before18baseline_heavyafter18.npy', allow_pickle=True)
group_subjects = np.load('data/subjects_complement.npy', allow_pickle=True)

# Filter each DataFrame by the common subjects and sort them based on subject ids (so that they are in the same order)
filtered_SC_df = SC_df[SC_df['subject'].isin(group_subjects)].sort_values('subject')
filtered_FC_df = FC_df[FC_df['subject'].isin(group_subjects)].sort_values('subject')
filtered_FCgsr_df = FCgsr_df[FCgsr_df['subject'].isin(group_subjects)].sort_values('subject')
filtered_demos_df = demos_df[demos_df['subject'].isin(group_subjects)].sort_values('subject')

# Get subjects with group assignments
if CONTROL_ONLY:
        subjects_with_groups_df = pd.read_csv('data/csv/control_subjects_with_groups.csv')
else:
        subjects_with_groups_df = pd.read_csv('data/csv/control_moderate_subjects_with_groups.csv')

subjects_with_groups_df = subjects_with_groups_df.drop(columns=['cahalan', 'visit'])

# Merge on filtered_sc_df to get group assignments
filtered_SC_df = filtered_SC_df.merge(subjects_with_groups_df, on='subject')
filtered_SC_df.sort_values('subject', inplace=True) # Sort based on subject ids once more to ensure order is preserved after merge
y_aligned = filtered_SC_df['group'].values

# Save the final aligned datasets
filtered_SC_df = filtered_SC_df.drop(columns=['subject', 'group'])
filtered_FC_df = filtered_FC_df.drop(columns=['subject'])
filtered_FCgsr_df = filtered_FCgsr_df.drop(columns=['subject'])
filtered_demos_df = filtered_demos_df.drop(columns=['subject'])
print("Shapes of the aligned datasets:", filtered_SC_df.shape, filtered_FC_df.shape, filtered_FCgsr_df.shape, filtered_demos_df.shape)

if CONTROL_ONLY:
    file_name = 'control'
else:
    file_name = 'control_moderate'

# Save the aligned training datasets
# np.save(f'data/18data/before18baseline_X_SC_{file_name}.npy', filtered_SC_df.values)
# np.save(f'data/18data/before18baseline_X_FC_{file_name}.npy', filtered_FC_df.values)
# np.save(f'data/18data/before18baseline_X_FCgsr_{file_name}.npy', filtered_FCgsr_df.values)
# np.save(f'data/18data/before18baseline_X_demos_{file_name}.npy', filtered_demos_df.values)
# # Save the aligned target labels
# np.save(f'data/18data/before18baseline_y_aligned_{file_name}.npy', y_aligned)

# Save the aligned training datasets
np.save(f'data/18data/comp_X_SC_{file_name}.npy', filtered_SC_df.values)
np.save(f'data/18data/comp_X_FC_{file_name}.npy', filtered_FC_df.values)
np.save(f'data/18data/comp_X_FCgsr_{file_name}.npy', filtered_FCgsr_df.values)
np.save(f'data/18data/comp_X_demos_{file_name}.npy', filtered_demos_df.values)
# Save the aligned target labels
np.save(f'data/18data/comp_y_aligned_{file_name}.npy', y_aligned)