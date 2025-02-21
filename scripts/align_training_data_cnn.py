import pandas as pd
import numpy as np

from process_conn_matrices_cnn import load_SC_matrices, load_FC_matrices

CONTROL_ONLY = False

 # SC matrices
SC_matrices, SC_subject_ids = load_SC_matrices('data/tractography_subcortical (SC)')
SC_df = pd.DataFrame({
        'conn_matrix': list(SC_matrices),  # Convert to list to treat each as a single object
        'subject': SC_subject_ids
    })

# FC matrices
FC_matrices, FCgsr_matrices, FC_subject_ids = load_FC_matrices(
path_to_FC='data/FC/NCANDA_FC.mat', 
path_to_FCgsr='data/FC/NCANDA_FCgsr.mat',
path_to_demography='data/FC/NCANDA_demos.csv'
)

FC_df = pd.DataFrame({
        'conn_matrix': list(FC_matrices),  # Convert to list to treat each as a single object
        'subject': FC_subject_ids
    })
FCgsr_df = pd.DataFrame({
        'conn_matrix': list(FCgsr_matrices),  # Convert to list to treat each as a single object
        'subject': FC_subject_ids
    })

# Demographic data (Just to get the same common subjects as the log reg)
if CONTROL_ONLY:
        demos_df = pd.read_csv('data/data_with_subject_ids/control_demos_with_subjects.csv')
else:
        demos_df = pd.read_csv('data/data_with_subject_ids/control_moderate_demos_with_subjects.csv')

# Identify common subjects
common_subjects = set(SC_df['subject']).intersection(FC_df['subject'], FCgsr_df['subject'], demos_df['subject'])
print(len(common_subjects))

# Filter each DataFrame by the common subjects and sort them based on subject ids (so that they are in the same order)
filtered_SC_df = SC_df[SC_df['subject'].isin(common_subjects)].sort_values('subject')
filtered_FC_df = FC_df[FC_df['subject'].isin(common_subjects)].sort_values('subject')
filtered_FCgsr_df = FCgsr_df[FCgsr_df['subject'].isin(common_subjects)].sort_values('subject')

if CONTROL_ONLY:
    file_name = 'control'
else:
    file_name = 'control_moderate'

# Save common subjects to a NumPy array
np.save(f'data/demo_analysis/{file_name}_common_subjects_cnn.npy', np.array(list(common_subjects)))

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

# Save the aligned datasets with subject ids
filtered_SC_df.to_csv(f'data/data_with_subject_ids/cnn/aligned/{file_name}_SC_with_subjects.csv', index=False)
filtered_FC_df.to_csv(f'data/data_with_subject_ids/cnn/aligned/{file_name}_FC_with_subjects.csv', index=False)
filtered_FCgsr_df.to_csv(f'data/data_with_subject_ids/cnn/aligned/{file_name}_FCgsr_with_subjects.csv', index=False)

X_SC = np.stack(filtered_SC_df['conn_matrix'].tolist())
X_FC = np.stack(filtered_FC_df['conn_matrix'].tolist())
X_FCgsr = np.stack(filtered_FCgsr_df['conn_matrix'].tolist())

# Save the aligned training datasets
np.save(f'data/training_data/cnn/aligned/X_SC_{file_name}.npy', X_SC)
np.save(f'data/training_data/cnn/aligned/X_FC_{file_name}.npy', X_FC)
np.save(f'data/training_data/cnn/aligned/X_FCgsr_{file_name}.npy', X_FCgsr)
# Save the aligned target labels
np.save(f'data/training_data/cnn/aligned/y_aligned_{file_name}.npy', y_aligned)

print("Shape of aligned SC matrices: ", X_SC.shape)
print("Shape of aligned FC matrices: ", X_FC.shape)
print("Shape of aligned FCgsr matrices: ", X_FCgsr.shape)
print("Shape of aligned target labels: ", y_aligned.shape)
