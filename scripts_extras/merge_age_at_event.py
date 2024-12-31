import pandas as pd
import numpy as np

df1 = pd.read_csv('data/demo_analysis/new_aligned_ages_cm_subjects_with_groups.csv')
df2 = pd.read_csv('data/18data/aligned_ages_cm_subjects_with_groups.csv')

# Merge the two DataFrames on the 'subject' column, giving priority to df2 for 'age_at_event'
df_merged = pd.merge(df1, df2[['subject', 'age_at_event']], on='subject', how='left', suffixes=('', '_correct'))

# Replace blanks or NaNs in 'age_at_event' with the corresponding values from 'age_at_event_correct'
df_merged['age_at_event'] = df_merged['age_at_event'].fillna(df_merged['age_at_event_correct'])

# Drop the helper column 'age_at_event_correct'
df_merged.drop(columns=['age_at_event_correct'], inplace=True)

# Save the updated DataFrame back to a CSV file
df_merged.to_csv('data/aligned_ages_cm_subjects_with_groups.csv', index=False)