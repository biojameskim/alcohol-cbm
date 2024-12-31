#!/usr/bin/env python3

"""
Name: extract_demo_data.py
Purpose: 
"""

import pandas as pd
import numpy as np

df = pd.read_csv('data/csv/cahalan_plus_drugs.csv')

# Set CONTROL_ONLY to True to only include subjects who are "control" at baseline
# Set CONTROL_ONLY to False to include subjects who are "control" or "moderate" at baseline
CONTROL_ONLY = False

if CONTROL_ONLY:
    print("Control ONLY")
    # Filter for subjects that are "control" on visit 0
    subjects_visit_0 = df[(df['visit'] == 0) & (df['cahalan'] == 'control')]
else:
    print("Control and Moderate")
    # Filter for subjects that are "control" or "moderate" on visit 0
    subjects_visit_0 = df[(df['visit'] == 0) & ((df['cahalan'] == 'control') | (df['cahalan'] == 'moderate'))]

# Get unique subject IDs for those subjects on visit 0
unique_subjects = subjects_visit_0['subject'].unique()

# Initialize lists to hold the subject information
group = []
age_at_visit0 = []
age_at_event = []
num_visits = []

# Loop through each unique subject and check their subsequent visits
for subject in unique_subjects:
    subject_data = df[df['subject'] == subject]
    visit0_age = subject_data[subject_data['visit'] == 0]['visit_age'].values[0]
    ages = subject_data['visit_age']

    # Count the number of follow-up visits for this subject
    visit_count = len(subject_data) # Number of visits
    num_visits.append(visit_count)
    
    if any(subject_data['cahalan'].isin(['heavy', 'heavy_with_binging'])):
        group.append(1)
        heavy_visit_age = ages[subject_data['cahalan'].isin(['heavy', 'heavy_with_binging'])].values[0]
        age_at_event.append(heavy_visit_age)
    else:
        group.append(0)
        last_visit_age = ages.values[-1]
        age_at_event.append(last_visit_age)
    
    age_at_visit0.append(visit0_age)

# Create a new DataFrame with the results
results = pd.DataFrame({
    'subject': unique_subjects,
    'group': group,
    'age_at_visit0': age_at_visit0,
    'age_at_event': age_at_event,
    'num_visits': num_visits
})

# Filter the original DataFrame to keep only relevant columns
if CONTROL_ONLY:
    print("Control ONLY")
    df = df[(df['visit'] == 0) & (df['cahalan'] == 'control')]
else:
    print("Control and Moderate")
    df = df[(df['visit'] == 0) & ((df['cahalan'] == 'control') | (df['cahalan'] == 'moderate'))]

# Merge the results with the filtered DataFrame
df = df[['subject', 'visit', 'cahalan']].merge(results, on='subject')

# Load in the common subjects array (from aligned dataset) (control moderate)
if CONTROL_ONLY:
    common_subjects = np.load('data/demo_analysis/control_common_subjects.npy', allow_pickle=True)
else:
    common_subjects = np.load('data/demo_analysis/control_moderate_common_subjects.npy', allow_pickle=True)

# Filter the main DataFrame to include only the subjects present in the common_subjects
common_df = df[df['subject'].isin(common_subjects)]

# Get race data
demos_df = pd.read_csv('data/csv/demographics_short.csv')
race_mapping = {
    'African-American/Black': 'African-American/Black',
    'African-American_Caucasian': 'Other',
    'Asian': 'Asian',
    'Asian_Pacific_Islander': 'Other',
    'Asian_White': 'Other',
    'Caucasian/White': 'Caucasian/White',
    'Native American/American Indian': 'Other',
    'NativeAmerican_Caucasian': 'Other',
    'Pacific Islander': 'Other',
    'Pacific_Islander_Caucasian': 'Other'
}
# Apply the mapping to the 'race_label' column
demos_df['race_label'] = demos_df['race_label'].map(race_mapping)

demos_df = demos_df.drop_duplicates(subset=['subject'])
df_merged = pd.merge(common_df, demos_df[['subject', 'sex', 'race_label', 'site', 'scanner_model']], on='subject', how='left')

# Save to a new CSV file
if CONTROL_ONLY:
    df_merged.to_csv('data/demo_analysis/aligned_ages_control_subjects_with_groups.csv', index=False)
else:
    df_merged.to_csv('data/demo_analysis/aligned_ages_cm_subjects_with_groups.csv', index=False)
