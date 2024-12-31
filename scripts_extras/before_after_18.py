# import pandas as pd


import pandas as pd
import numpy as np

################################################################
# Load the main DataFrame (all subjects with ages and groups)
# main_df = pd.read_csv('data/ages_cm_subjects_with_groups.csv')

# # Assuming subjects_df is a NumPy array containing the subject identifiers
# subjects_array = np.load('data/common_subjects.npy', allow_pickle=True)

# # Filter the main DataFrame to include only the subjects present in the subjects_array
# filtered_main_df = main_df[main_df['subject'].isin(subjects_array)]

# # Save the filtered DataFrame to a new CSV file (optional)
# filtered_main_df.to_csv('data/aligned_ages_cm_subjects_with_groups.csv', index=False)

# print(filtered_main_df.head())  # Display the first few rows of the filtered DataFrame

#######################################################################

# Load the CSV file created by the previous script
df = pd.read_csv('data/demo_analysis/aligned_ages_cm_subjects_with_groups.csv')

# Filter for subjects in group 1
group1_subjects = df[df['group'] == 1]
# Filter for subjects in group 0 (never engage in heavy or heavy_with_binging drinking)
group0_subjects = df[df['group'] == 0]

# Further filter for subjects with age_at_visit0 < 18 and age_at_event > 18
filtered_subjects1 = group1_subjects[(group1_subjects['age_at_visit0'] < 18) & (group1_subjects['age_at_event'] > 18)]
filtered_subjects0 = group0_subjects[(group0_subjects['age_at_visit0'] < 18) & (group0_subjects['age_at_event'] > 18)]
filtered_subjects = pd.concat([filtered_subjects1, filtered_subjects0], axis=0)
# np.save('data/subjects_before18baseline_heavyafter18', filtered_subjects['subject'].values)

filtered_subjects1_complement = group1_subjects[(group1_subjects['age_at_visit0'] > 18) | (group1_subjects['age_at_event'] < 18)]
filtered_subjects0_complement = group0_subjects[(group0_subjects['age_at_visit0'] > 18) | (group0_subjects['age_at_event'] < 18)]
filtered_subjects_complement = pd.concat([filtered_subjects1_complement, filtered_subjects0_complement], axis=0)
# np.save('data/subjects_complement.npy', filtered_subjects_complement['subject'].values)

first_visit_over_18_group1 = group1_subjects[group1_subjects['age_at_visit0'] > 18]
first_visit_over_18_group0 = group0_subjects[group0_subjects['age_at_visit0'] > 18]

# Find the last visit (follow-up) age for each subject
last_visit_ages_group1 = group1_subjects['age_at_event'].values
last_visit_ages_group0 = group0_subjects['age_at_event'].values

# Check how many subjects have their last follow-up age < 18 and >= 18
last_visit_under_18_group1 = last_visit_ages_group1[last_visit_ages_group1 < 18].shape[0]
last_visit_over_18_group1 = last_visit_ages_group1[last_visit_ages_group1 >= 18].shape[0]

last_visit_under_18_group0 = last_visit_ages_group0[last_visit_ages_group0 < 18].shape[0]
last_visit_over_18_group0 = last_visit_ages_group0[last_visit_ages_group0 >= 18].shape[0]


# Count the number of subjects that meet the criteria
num_subjects = filtered_subjects1.shape[0]
num_subjects2 = filtered_subjects0.shape[0]

print("N dataset:", df.shape[0])
print("N group 1:", group1_subjects.shape[0])
print("N group 0:", group0_subjects.shape[0])
print("\n")
print("Group 1:")
print("age_at_visit0 < 18:", group1_subjects[(group1_subjects['age_at_visit0'] < 18)].shape[0])
print(f'age_at_visit0 < 18 and age_at_event > 18: {num_subjects}')
print("age_at_visit0 > 18:", first_visit_over_18_group1.shape[0])
print(f"last visit before age 18: {last_visit_under_18_group1}")
print(f"last visit after age 18: {last_visit_over_18_group1}")

print("\n")
print("Group 0:")
print("age_at_visit0 < 18:", group0_subjects[(group0_subjects['age_at_visit0'] < 18)].shape[0])
print(f'age_at_visit0 < 18 and age_at_event > 18: {num_subjects2}')
print("age_at_visit0 > 18:", first_visit_over_18_group0.shape[0])
print(f"last visit before age 18: {last_visit_under_18_group0}")
print(f"last visit after age 18: {last_visit_over_18_group0}")


