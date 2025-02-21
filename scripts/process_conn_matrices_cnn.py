import os
import scipy.io
import numpy as np
import pandas as pd

def load_SC_matrices(directory):
    """
    Load and flatten the SC matrices from the given directory.
    Returns: [SC_matrices]: A 2D array of SC matrices
                [subject_ids]: An array of matching (in order of the SC matrices) subject IDs
    """
    SC_matrices = []
    subject_ids = []

    for filename in os.listdir(directory):
        if filename.endswith('baseline.mat'): # Only extract baseline matrices
            file_path = os.path.join(directory, filename)
            mat_contents = scipy.io.loadmat(file_path)
            SC_matrix = mat_contents['matrix'] # Extract the connectivity matrix from the .mat file
            
            SC_matrices.append(SC_matrix)
            subject_id = "NCANDA_" + filename.split('_')[1]  # Extract subject number, e.g., 'NCANDA_S00033'
            subject_ids.append(subject_id)

    return np.array(SC_matrices), np.array(subject_ids)

def load_FC_matrices(path_to_FC, path_to_FCgsr, path_to_demography):
    """
    Load the FC matrices (with and without gsr) and extract the subject IDs.
    [path_to_FC]: Path to the FC .mat file
    [path_to_FCgsr]: Path to the FCgsr .mat file
    [path_to_demography]: Path to the demography csv file. The ordering of the subjects in the FC/FCgsr matrices is based on this file.
    Returns: [baseline_FC_matrices]: A 2D array of baseline FC matrices
             [baseline_FCgsr_matrices]: A 2D array of baseline FCgsr matrices
             [subject_ids]: An array of matching (in order of the baseline FC matrices) subject IDs
    """
    FC_mat_contents = scipy.io.loadmat(path_to_FC)
    FC_cell_array = FC_mat_contents['FC']
    FC_matrices = FC_cell_array[0] # This is an array of FC matrices for all subjects (contains duplicate subjects based on visit)

    # Same process for FCgsr matrices
    FCgsr_mat_contents = scipy.io.loadmat(path_to_FCgsr)
    FCgsr_cell_array = FCgsr_mat_contents['FC']
    FCgsr_matrices = FCgsr_cell_array[0]

    demos = pd.read_csv(path_to_demography)
    baseline_subjects = demos[(demos['visit'] == 'baseline')] # Filter for baseline subjects
    baseline_indices = baseline_subjects.index

    baseline_FC_matrices = FC_matrices[baseline_indices] # extract FC matrices for just the baseline
    baseline_FCgsr_matrices = FCgsr_matrices[baseline_indices] # extract FCgsr matrices for just the baseline

    # Extract matching subject ids, e.g., 'NCANDA_S00033'
    subject_ids = np.array(baseline_subjects['subject'])

    return baseline_FC_matrices, baseline_FCgsr_matrices, subject_ids


def get_X_y(conn_matrices, subject_ids, subjects_with_groups_df):
    """
    Prepare the feature matrix X and target labels (groups) y for the logistic regression model.
    [conn_matrices] is a 2D array of flattened FC/SC matrices for all subjects
    [subjects] is a 1D array of subject IDs (e.g., "NCANDA_S00033")
    [subjects_with_groups_df] is a DataFrame with each subject and their group label
    Returns: A tuple of (X, y) where X is a 2D array and y is a 1D array
    """
    # Create a DataFrame with each matrix stored as an object
    conn_matrices_df = pd.DataFrame({
        'conn_matrix': list(conn_matrices),  # Convert to list to treat each as a single object
        'subject': subject_ids
    })

    conn_matrices_df = conn_matrices_df.drop_duplicates(subset=['subject']) # drop all duplicates (sanity check- should be none)
    conn_matrices_with_groups = conn_matrices_df.merge(subjects_with_groups_df, on='subject') # Merge with the group labels

    X = np.stack(conn_matrices_with_groups['conn_matrix'].values)
    y = conn_matrices_with_groups['group'].values  # Group labels

    return X, y

if __name__ == "__main__":
    CONTROL_ONLY = False

    if CONTROL_ONLY:
        subjects_with_groups_df = pd.read_csv('data/csv/control_subjects_with_groups.csv')
        file_name = 'control'
    else:
        subjects_with_groups_df = pd.read_csv('data/csv/control_moderate_subjects_with_groups.csv')
        file_name = 'control_moderate'

    # SC matrices
    SC_matrices, SC_subject_ids = load_SC_matrices('data/tractography_subcortical (SC)')
    X_SC, y_SC = get_X_y(SC_matrices, SC_subject_ids, subjects_with_groups_df)
    np.save(f'data/training_data/cnn/unaligned/X_SC_{file_name}.npy', X_SC)
    np.save(f'data/training_data/cnn/unaligned/y_SC_{file_name}.npy', y_SC)

    # FC matrices
    FC_matrices, FCgsr_matrices, FC_subject_ids = load_FC_matrices(
    path_to_FC='data/FC/NCANDA_FC.mat', 
    path_to_FCgsr='data/FC/NCANDA_FCgsr.mat',
    path_to_demography='data/FC/NCANDA_demos.csv'
    )

    X_FC, y_FC = get_X_y(FC_matrices, FC_subject_ids, subjects_with_groups_df)
    X_FCgsr, y_FCgsr = get_X_y(FCgsr_matrices, FC_subject_ids, subjects_with_groups_df)
    
    np.save(f'data/training_data/cnn/unaligned/X_FC_{file_name}.npy', X_FC)
    np.save(f'data/training_data/cnn/unaligned/y_FC_{file_name}.npy', y_FC)
    np.save(f'data/training_data/cnn/unaligned/X_FCgsr_{file_name}.npy', X_FCgsr)
    np.save(f'data/training_data/cnn/unaligned/y_FCgsr_{file_name}.npy', y_FCgsr)


# TODO --> Now I have just the X and the y. But I have to filter out the "aligned" ones.
# Maybe i'm doing too much repetitive code. this is essentially the same thing as before but not flattening
# Perhaps appending the matrices isn't the best appraoch. Is there a way to just build it in the 3rd dim?