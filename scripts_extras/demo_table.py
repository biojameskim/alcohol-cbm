import pandas as pd

# Load the CSV file
df = pd.read_csv('data/demo_analysis/aligned_ages_cm_subjects_with_groups.csv')

# Define the summary table
summary_table = {}

# Categorical Distributions
categorical_columns = ['sex', 'race_label', 'site', 'scanner_model', 'cahalan']

for column in categorical_columns:
    summary_table[column] = df.groupby('group')[column].value_counts().unstack().fillna(0)

# Continuous Variables Summary
continuous_columns = ['age_at_visit0', 'age_at_event', 'num_visits']

continuous_summary = df.groupby('group')[continuous_columns].agg(['mean', 'median', 'std']).round(2)

# Combine the categorical and continuous summaries
summary_table_combined = pd.concat([summary_table[column] for column in categorical_columns] + [continuous_summary], axis=1)

# Reshape the DataFrame to be more readable
summary_table_reshaped = summary_table_combined.stack(level=0).unstack(level=0).fillna('')

# Display the summary table
print(summary_table_combined)

summary_table_combined.to_csv('data/demo_analysis/demo_table.csv', index=True)